import os
import json
import logging
import numpy as np
import re
from typing import List
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from sentence_transformers import SentenceTransformer
import sentence_transformers 
from sbert import load_sbert 

logger = logging.getLogger(__name__)

def evaluate(
    data_path: str,
    output_dir: str,
    model_path: str,
    max_seq_length: int = 350,
    score_function: str = "dot",
    pooling: str = None,
    sep: str = " ",
    k_values: List[int] = [1, 3, 5, 10, 20, 100],
    split: str = "test",
    gpl_steps: int = 100,
):

    ndcgs = []
    _maps = []
    recalls = []
    precisions = []
    mrrs = []
 
    logger.info(f"Evaluating model: {model_path}")
    
    model: SentenceTransformer = load_sbert(model_path, pooling, max_seq_length)

    pooling_module: sentence_transformers.models.Pooling = model._last_module()
    assert type(pooling_module) == sentence_transformers.models.Pooling
    pooling_mode = pooling_module.get_pooling_mode_str()
    logger.info(
        f"Running evaluation with setting: max_seq_length = {max_seq_length}, score_function = {score_function}, split = {split} and pooling: {pooling_mode}"
    )

    data_paths = []
    if "cqadupstack" in data_path:
        data_paths = [
            os.path.join(data_path, sub_dataset)
            for sub_dataset in [
                "android",
                "english",
                "gaming",
                "gis",
                "mathematica",
                "physics",
                "programmers",
                "stats",
                "tex",
                "unix",
                "webmasters",
                "wordpress",
            ]
        ]
    else:
        data_paths.append(data_path)

    for data_path in data_paths:
        try:
            corpus, queries, qrels = GenericDataLoader(data_path).load(split=split)
        except ValueError as e:
            missing_files = re.search(
                r"File (.*) not present! Please provide accurate file.", str(e)
            )
            if missing_files:
                raise ValueError(
                    f"Missing evaluation data files ({missing_files.groups()}). "
                    f"Please put them under {data_path} or set `do_evaluation`=False."
                )
            else:
                raise e

        sbert = models.SentenceBERT(sep=sep)
        sbert.q_model = model
        sbert.doc_model = model

        model_dres = DRES(sbert, batch_size=16)
        assert score_function in ["dot", "cos_sim"]
        retriever = EvaluateRetrieval(
            model_dres, score_function=score_function, k_values=k_values
        )  # or "dot" for dot-product
        results = retriever.retrieve(corpus, queries)

        #### Evaluate your retrieval using NDCG@k, MAP@K ...
        ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
            qrels, results, k_values
        )
        mrr = EvaluateRetrieval.evaluate_custom(qrels, results, k_values, metric="mrr")
        ndcgs.append(ndcg)
        _maps.append(_map)
        recalls.append(recall)
        precisions.append(precision)
        mrrs.append(mrr)

    ndcg = {k: np.mean([score[k] for score in ndcgs]) for k in ndcg}
    _map = {k: np.mean([score[k] for score in _maps]) for k in _map}
    recall = {k: np.mean([score[k] for score in recalls]) for k in recall}
    precision = {k: np.mean([score[k] for score in precisions]) for k in precision}
    mrr = {k: np.mean([score[k] for score in mrrs]) for k in mrr}

    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, f"results_{gpl_steps}.json")
    with open(result_path, "w") as f:
        json.dump(
            {
                "model_paths": model_path,
                "gpl_steps": gpl_steps,
                "data_path": data_path,
                "ndcg": ndcg,
                "map": _map,
                "recall": recall,
                "precicion": precision,
                "mrr": mrr,
            },
            f,
            indent=4,
        )
    logger.info(f"Saved evaluation results to {result_path}")
    
    
if __name__ == "__main__":
    checkpoint_dir = "C:/Users/Siemens/Documents/TFE_Nick_Kuijpers/LoRa_GPL/GPL_LoRA/output/Siemens/distilbert-base-uncased/checkpoints" 
    # Iterate over all files in the directory
    for filename in os.listdir(checkpoint_dir):
        # Check if the file represents a model checkpoint
        if filename.isdigit():  # Assuming model checkpoints are named with numbers
            model_path = os.path.join(checkpoint_dir, filename)
            evaluate(
                data_path="C:/Users/Siemens/Documents/TFE_Nick_Kuijpers/LoRa_GPL/GPL_LoRA/Siemens",
                output_dir="C:/Users/Siemens/Documents/TFE_Nick_Kuijpers/LoRa_GPL/GPL_LoRA/evaluation/LoRa_r8_a32/Siemens/distilbert-base-uncased",
                model_path=model_path,
                max_seq_length=350,
                score_function="dot",
                pooling=None, 
                split="test",
                gpl_steps=filename,
            ) 