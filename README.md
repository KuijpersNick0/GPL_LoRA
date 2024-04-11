Skeleton GPL is forked from https://github.com/UKPLab/gpl. 
Adapting GPL with LoRA now for my thesis.

Will update this README later.

# Generative Pseudo Labeling (GPL)
GPL is an unsupervised domain adaptation method for training dense retrievers. It is based on query generation and pseudo labeling with powerful cross-encoders. To train a domain-adapted model, it needs only the unlabeled target corpus and can achieve significant improvement over zero-shot models.

For more information, checkout the publication:
- [GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval](https://arxiv.org/abs/2112.07577) (NAACL 2022)

## Installation
One can either install GPL via `pip`
```bash
pip install gpl
```
or via `git clone`
```bash
git clone https://github.com/UKPLab/gpl.git && cd gpl
pip install -e .
```
> Meanwhile, please make sure the [correct version of PyTorch](https://pytorch.org/get-started/locally/) has been installed according to your CUDA version.

## How does GPL work?
The workflow of GPL is shown as follows:
![](imgs/GPL.png)
1. GPL first use a seq2seq (we use [BeIR/query-gen-msmarco-t5-base-v1](https://huggingface.co/BeIR/query-gen-msmarco-t5-base-v1) by default) model to generate `queries_per_passage` queries for each passage in the unlabeled corpus. The query-passage pairs are viewed as **positive examples** for training.
    > Result files (under path `$path_to_generated_data`): (1) `${qgen}-qrels/train.tsv`, (2) `${qgen}-queries.jsonl` and also (3) `corpus.jsonl` (copied from `$evaluation_data/`);
2. Then, it runs negative mining with the generated queries as input on the target corpus. The mined passages will be viewed as **negative examples** for training. One can specify any dense retrievers ([SBERT](https://github.com/UKPLab/sentence-transformers) or [Huggingface/transformers](https://github.com/huggingface/transformers) checkpoints, we use [msmarco-distilbert-base-v3](sentence-transformers/msmarco-distilbert-base-v3) + [msmarco-MiniLM-L-6-v3](https://huggingface.co/sentence-transformers/msmarco-MiniLM-L-6-v3) by default) or BM25 to the argument `retrievers` as the negative miner.
    > Result file (under path `$path_to_generated_data`): hard-negatives.jsonl;
3. Finally, it does pseudo labeling with the powerful cross-encoders (we use [cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) by default.) on the query-passage pairs that we have so far (for both positive and negative examples).
    > Result file (under path `$path_to_generated_data`): `gpl-training-data.tsv`. It contains (`gpl_steps` * `batch_size_gpl`) tuples in total.

