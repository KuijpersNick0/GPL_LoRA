from sentence_transformers import SentenceTransformer, models
import logging
from peft import LoraConfig, TaskType, get_peft_model
import torch
import transformers
from transformers import is_torch_npu_available
from torch import nn, Tensor, device
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import List, Dict, Literal, Tuple, Iterable, Type, Union, Callable, Optional
from tqdm.autonotebook import trange
import os

logger = logging.getLogger(__name__)

class LoRaSentenceTransformer(SentenceTransformer):
    def __init__(self, model_name_or_path: str):
        super(LoRaSentenceTransformer, self).__init__(model_name_or_path)
        
    # Custom training loop for LoRa
    def LoRa_fit(  
        self, 
        train_objectives: Iterable[Tuple[DataLoader, nn.Module]], 
        evaluator=None,
        epochs=1,
        steps_per_epoch=None,
        scheduler="WarmupLinear",
        warmup_steps=10000,
        optimizer_class=torch.optim.AdamW,
        optimizer_params={"lr": 2e-5},
        weight_decay=0.01,
        evaluation_steps=0,
        output_path=None,
        save_best_model=True,
        max_grad_norm=1,
        use_amp=False,
        callback=None,
        show_progress_bar=True,
        checkpoint_path=None,
        checkpoint_save_steps=500,
        checkpoint_save_total_limit=0,
        ):
        """
        Train the model with the given training objective using LoRa (PEFT) technique.
        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to ensure equal training with each dataset.

        :param model: The SentenceTransformer model with LoRa (PEFT) modifications.
        :param train_objectives: Tuples of (DataLoader, LossFunction). Pass more than one for multi-task learning.
        :param epochs: Number of epochs for training.
        :param steps_per_epoch: Number of training steps per epoch. If set to None (default), one epoch is equal to the DataLoader size from train_objectives.
        :param scheduler: Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts.
        :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from 0 up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero.
        :param optimizer_class: Optimizer.
        :param optimizer_params: Optimizer parameters.
        :param weight_decay: Weight decay for model parameters.
        :param evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps.
        :param output_path: Storage path for the model and evaluation files.
        :param save_best_model: If True, the best model (according to evaluator) is stored at output_path.
        :param max_grad_norm: Used for gradient normalization.
        :param use_amp: Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0.
        :param callback: Callback function that is invoked after each evaluation. It must accept the following three parameters in this order: `score`, `epoch`, `steps`.
        :param show_progress_bar: If True, output a tqdm progress bar.
        :param checkpoint_path: Folder to save checkpoints during training.
        :param checkpoint_save_steps: Will save a checkpoint after so many steps.
        :param checkpoint_save_total_limit: Total number of checkpoints to store.
        """ 
        if use_amp:
            if is_torch_npu_available():
                scaler = torch.npu.amp.GradScaler()
            else:
                scaler = torch.cuda.amp.GradScaler()

        device = self.device
        self.to(device)

        dataloaders = [dataloader for dataloader, _ in train_objectives]
        loss_models = [loss for _, loss in train_objectives]

        # Use smart batching
        for dataloader in dataloaders:
            dataloader.collate_fn = self.smart_batching_collate
        
        # Move loss models to device
        for loss_model in loss_models:
            loss_model.to(device)

        best_score = -9999999

        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])

        num_train_steps = int(steps_per_epoch * epochs)

        # Prepare optimizers and schedulers
        optimizers = []
        schedulers = []
        for loss_model in loss_models:
            param_optimizer = list(loss_model.named_parameters())

            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                    "weight_decay": weight_decay,
                },
                {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            ]

            optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
            scheduler_obj = self._get_scheduler(
                optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps
            )

            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)

        global_step = 0
        data_iterators = [iter(dataloader) for dataloader in dataloaders]

        num_train_objectives = len(train_objectives)

        skip_scheduler = False
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0

            for loss_model in loss_models:
                loss_model.zero_grad()
                loss_model.train()

            for _ in trange(steps_per_epoch, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):
                for train_idx in range(num_train_objectives):
                    loss_model = loss_models[train_idx]
                    optimizer = optimizers[train_idx]
                    scheduler = schedulers[train_idx]
                    data_iterator = data_iterators[train_idx]

                    try:
                        data = next(data_iterator)
                    except StopIteration:
                        data_iterator = iter(dataloaders[train_idx])
                        data_iterators[train_idx] = data_iterator
                        data = next(data_iterator)

                    features, labels = data
                    labels = labels.to(device)
                    features = list(map(lambda batch: self.batch_to_device(batch, device), features))

                    if use_amp:
                        with torch.autocast(device_type=device.type):
                            loss_value = loss_model(features, labels)

                        scale_before_step = scaler.get_scale()
                        scaler.scale(loss_value).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()

                        skip_scheduler = scaler.get_scale() != scale_before_step
                    else:
                        loss_value = loss_model(features, labels)
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        optimizer.step()

                    optimizer.zero_grad()

                    if not skip_scheduler:
                        scheduler.step()

                training_steps += 1
                global_step += 1

                if evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_during_training(
                        evaluator, output_path, save_best_model, epoch, training_steps, callback
                    )

                    for loss_model in loss_models:
                        loss_model.zero_grad()
                        loss_model.train()

                if (
                    checkpoint_path is not None
                    and checkpoint_save_steps is not None
                    and checkpoint_save_steps > 0
                    and global_step % checkpoint_save_steps == 0
                ):
                    self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)

            self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1, callback)

        if evaluator is None and output_path is not None:  # No evaluator, but output path: save final model version
            # model.save(output_path)
            self.save_pretrained("outputs_lora")

        if checkpoint_path is not None:
            # model._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)
            self.save_pretrained("outputs_lora " + str(global_step) + " steps")
          
    def batch_to_device(self, batch, target_device: device):
        """
        send a pytorch batch to a device (CPU/GPU)
        """
        for key in batch:
            if isinstance(batch[key], Tensor):
                batch[key] = batch[key].to(target_device)
        return batch
    
    @staticmethod
    def _get_scheduler(optimizer, scheduler: str, warmup_steps: int, t_total: int):
        """
        Returns the correct learning rate scheduler. Available scheduler: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        """
        scheduler = scheduler.lower()
        if scheduler == "constantlr":
            return transformers.get_constant_schedule(optimizer)
        elif scheduler == "warmupconstant":
            return transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
        elif scheduler == "warmuplinear":
            return transformers.get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
            )
        elif scheduler == "warmupcosine":
            return transformers.get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
            )
        elif scheduler == "warmupcosinewithhardrestarts":
            return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
            )
        else:
            raise ValueError("Unknown scheduler {}".format(scheduler))
        
    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps, callback):
            """Runs evaluation during the training"""
            eval_path = output_path
            if output_path is not None:
                os.makedirs(output_path, exist_ok=True)
                eval_path = os.path.join(output_path, "eval")
                os.makedirs(eval_path, exist_ok=True)

            if evaluator is not None:
                score = evaluator(self, output_path=eval_path, epoch=epoch, steps=steps)
                if callback is not None:
                    callback(score, epoch, steps)
                if score > self.best_score:
                    self.best_score = score
                    if save_best_model:
                        self.save(output_path)
    
def directly_loadable_by_sbert(model: SentenceTransformer):
    loadable_by_sbert = True
    try:
        texts = [
            "This is an input text",
        ]
        model.encode(texts)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            raise e
        else:
            loadable_by_sbert = False
    return loadable_by_sbert


def load_lora(model_name_or_path, pooling=None, max_seq_length=None):
    model = LoRaSentenceTransformer(model_name_or_path)
    
    peft_config = None

    ## Check whether SBERT can load the checkpoint and use it
    loadable_by_sbert = directly_loadable_by_sbert(model)
    if loadable_by_sbert:
        ## Loadable by SBERT directly
        ## Mainly two cases: (1) The checkpoint is in SBERT-format (e.g. "bert-base-nli-mean-tokens"); (2) it is in HF-format but the last layer can provide a hidden state for each token (e.g. "bert-base-uncased")
        ## NOTICE: Even for (2), there might be some checkpoints (e.g. "princeton-nlp/sup-simcse-bert-base-uncased") that uses a linear layer on top of the CLS token embedding to get the final dense representation. In this case, setting `--pooling` to a specify pooling method will misuse the checkpoint. This is why we recommend to use SBERT-format if possible
        ## Setting pooling if needed
        if pooling is not None:
            logger.warning(
                f"Trying setting pooling method manually (`--pooling={pooling}`). Recommand to use a checkpoint in SBERT-format and leave the `--pooling=None`: This is less likely to misuse the pooling"
            )
            last_layer: models.Pooling = model[-1]
            assert (
                type(last_layer) == models.Pooling
            ), f"The last layer is not a pooling layer and thus `--pooling={pooling}` cannot work. Please try leaving `--pooling=None` as in the default setting"
            # We here change the pooling by building the whole SBERT model again, which is safer and more maintainable than setting the attributes of the Pooling module
            word_embedding_model = models.Transformer(
                model_name_or_path, max_seq_length=max_seq_length
            )
            pooling_model = models.Pooling(
                word_embedding_model.get_word_embedding_dimension(),
                pooling_mode=pooling,
            )
            print(word_embedding_model)
            print(pooling_model)
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    else:
        ## Not directly loadable by SBERT
        ## Mainly one case: The last layer is a linear layer (e.g. "facebook/dpr-question_encoder-single-nq-base")
        raise NotImplementedError(
            "This checkpoint cannot be directly loadable by SBERT. Please transform it into SBERT-format first and then try again. Please pay attention to its last layer"
        )

    ## Setting max_seq_length if needed
    if max_seq_length is not None:
        first_layer: models.Transformer = model[0]
        assert (
            type(first_layer) == models.Transformer
        ), "Unknown error, please report this"
        assert hasattr(
            first_layer, "max_seq_length"
        ), "Unknown error, please report this"
        setattr(
            first_layer, "max_seq_length", max_seq_length
        )  # Set the maximum-sequence length
        logger.info(f"Set max_seq_length={max_seq_length}")
 
    # Use a default LoRA configuration
    peft_config = LoraConfig(
        r=1,
        lora_alpha=1,
        bias="none",
        # task_type=TaskType.FEATURE_EXTRACTION,
        # inference_mode=False,
        target_modules=["key", "query", "value"],
    )
    
    # # Layers before Lora
    # print("/n Printing layers /n")
    # for n, m in model.named_modules():
    #     print(n, type(m))
    #     print()
    # print("/n End of printing layers /n")
        
        
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Layers after Lora
    # print("/n Printing layers /n")
    # for n, m in model.named_modules():
    #     print(n, type(m))
    #     print()
    # print("/n End of printing layers /n")
    
    # # Layers applying Lora
    # print("\nTargeted Modules:")
    # for module in model.targeted_module_names:
    #     print(module)
    # print() 
    
    return model
