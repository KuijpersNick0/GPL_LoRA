This work is originally forked from https://github.com/UKPLab/gpl. 
For my Master Thesis I adapted the Sentence-Transformer with LoRA and used the same framework as the original GPL publication.
 
# Generative Pseudo Labeling (GPL)
GPL is an unsupervised domain adaptation method for training dense retrievers. It is based on query generation and pseudo labeling with powerful cross-encoders. To train a domain-adapted model, it needs only the unlabeled target corpus and can achieve significant improvement over zero-shot models.

For more information, checkout the publication:
- [GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval](https://arxiv.org/abs/2112.07577) (NAACL 2022)

## Installation
- Use the same libraries as in the environment.yml file
- Make sure the [correct version of PyTorch](https://pytorch.org/get-started/locally/) has been installed according to your CUDA version.

## How does GPL work?
1. GPL first use a seq2seq (we use [BeIR/query-gen-msmarco-t5-base-v1](https://huggingface.co/BeIR/query-gen-msmarco-t5-base-v1) by default) model to generate `queries_per_passage` queries for each passage in the unlabeled corpus. The query-passage pairs are viewed as **positive examples** for training.
    > Result files (under path `$path_to_generated_data`): (1) `${qgen}-qrels/train.tsv`, (2) `${qgen}-queries.jsonl` and also (3) `corpus.jsonl` (copied from `$evaluation_data/`);
2. Then, it runs negative mining with the generated queries as input on the target corpus. The mined passages will be viewed as **negative examples** for training. One can specify any dense retrievers ([SBERT](https://github.com/UKPLab/sentence-transformers) or [Huggingface/transformers](https://github.com/huggingface/transformers) checkpoints, we use [msmarco-distilbert-base-v3](sentence-transformers/msmarco-distilbert-base-v3) + [msmarco-MiniLM-L-6-v3](https://huggingface.co/sentence-transformers/msmarco-MiniLM-L-6-v3) by default) or BM25 to the argument `retrievers` as the negative miner.
    > Result file (under path `$path_to_generated_data`): hard-negatives.jsonl;
3. Finally, it does pseudo labeling with the powerful cross-encoders (we use [cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) by default.) on the query-passage pairs that we have so far (for both positive and negative examples).
    > Result file (under path `$path_to_generated_data`): `gpl-training-data.tsv`. It contains (`gpl_steps` * `batch_size_gpl`) tuples in total.

# LoRA 

LoRA (Low-Rank Adaptation) is a method designed to enhance the training efficiency and adaptability of Large Language Models (LLM), such as Sentence-Transformers, by introducing low-rank adaptations. This technique is particularly useful in scenarios where computational resources are limited or when fine-tuning large models on specific tasks is necessary. Here's a brief explanation of how LoRA works:

## Key Concepts

1. **Low-Rank Matrices**: The core idea of LoRA is to approximate the weight updates in a neural network using low-rank matrices. Instead of updating the full weight matrix during training, LoRA inserts low-rank matrices into the architecture, which significantly reduces the number of parameters to be updated.

2. **Parameter Efficiency**: By decomposing the weight updates into low-rank matrices, LoRA reduces the number of trainable parameters. This makes the training process more parameter-efficient, which is especially advantageous when dealing with large models.

3. **Adaptation Process**: During fine-tuning, LoRA introduces additional trainable low-rank matrices into each layer of the pre-trained model. These matrices are trained on the specific task, allowing the model to adapt to new data without modifying the original weights extensively.

## Workflow

1. **Initialize Low-Rank Matrices**: For each layer in the pre-trained model, initialize low-rank matrices that will capture the weight updates.

2. **Insert Low-Rank Adaptations**: Insert the low-rank matrices into the model architecture. These matrices are designed to approximate the weight updates needed for fine-tuning.

3. **Fine-Tuning**: Train the model on the target task using the low-rank matrices. During this phase, only the low-rank matrices are updated, while the original model weights remain unchanged.

4. **Inference**: During inference, the modified model, which now includes the low-rank adaptations, is used to make predictions. The adaptations allow the model to perform well on the specific task it was fine-tuned for.

## Benefits

- **Reduced Computational Cost**: LoRA significantly lowers the computational requirements for fine-tuning large models.
- **Efficiency**: It allows for efficient training and adaptation with fewer parameters.
- **Flexibility**: LoRA can be applied to various types of neural networks and tasks, making it a versatile approach for model adaptation.


# Structure of the Code

## GPL Folder

The `GPL` folder primarily contains the code from the original GPL publication. It includes all the necessary scripts and tools for creating new queries, generating pseudo-labels, performing negative mining, and training new Sentence-Transformers, including those enhanced with LoRA.

**Main Addition:**
- **`toolkit/LoRA.py`**: This file is the main addition to the GPL folder. It introduces the new structure for training Sentence-Transformers enhanced with LoRA. 
- To use this file, you need to modify the `train.py` script to utilize a LoRA-enhanced Sentence-Transformer instead of the original "simple" Sentence-Transformer as described in the original GPL publication and code.

## Python Folder

The `Python` folder contains all the scripts and tools related to evaluating models, benchmarking, displaying results, and preparing data. 

**Key Scripts:**
- **Evaluation and Benchmarking**: Scripts to evaluate the performance of trained models and compare benchmarks.
- **Data Preparation**: Various scripts to format and preprocess data correctly before it can be used with the GPL algorithm.
  - **`tsv_clean.py`**: A specific script needed when new data requires formatting after a GPL run. This is particularly useful since BEIR, the benchmarking library, was originally a Linux-based library.

By organizing the code in this manner, we ensure that all processes from data preparation to model evaluation are streamlined and accessible. The additions and modifications made enhance the original GPL framework, enabling the integration of LoRA for improved Sentence-Transformer training.

For detailed instructions on how to run the scripts and use the new functionalities, please refer to the comments and documentation within each script.




