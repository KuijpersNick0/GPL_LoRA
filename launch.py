import gpl
import torch

# Emptyy torch cache
torch.cuda.empty_cache()

dataset = 'scifact'
gpl.train(
    path_to_generated_data=f"generated/{dataset}",
    base_ckpt="distilbert-base-uncased",  
    # intfloat/e5-small-v2
    # distilbert-base-uncased
    # base_ckpt='GPL/msmarco-distilbert-margin-mse',  
    # The starting checkpoint of the experiments in the paper
    gpl_score_function="dot",
    # Note that GPL uses MarginMSE loss, which works with dot-product
    batch_size_gpl=32,
    gpl_steps=1000,
    new_size=-1,
    # Resize the corpus to `new_size` (|corpus|) if needed. When set to None (by default), the |corpus| will be the full size. When set to -1, the |corpus| will be set automatically: If QPP * |corpus| <= 250K, |corpus| will be the full size; else QPP will be set 3 and |corpus| will be set to 250K / 3
    queries_per_passage=-1,
    # Number of Queries Per Passage (QPP) in the query generation step. When set to -1 (by default), the QPP will be chosen automatically: If QPP * |corpus| <= 250K, then QPP will be set to 250K / |corpus|; else QPP will be set 3 and |corpus| will be set to 250K / 3
    output_dir=f"output/{dataset}",
    evaluation_data=f"./{dataset}",
    evaluation_output=f"evaluation/{dataset}",
    generator="BeIR/query-gen-msmarco-t5-base-v1",
    retrievers=["msmarco-distilbert-base-v3", "msmarco-MiniLM-L-6-v3"],
    retriever_score_functions=["cos_sim", "cos_sim"],
    # Note that these two retriever model work with cosine-similarity
    cross_encoder="cross-encoder/ms-marco-MiniLM-L-6-v2",
    qgen_prefix="qgen",
    # This prefix will appear as part of the (folder/file) names for query-generation results: For example, we will have "qgen-qrels/" and "qgen-queries.jsonl" by default.
    do_evaluation=False,
    use_amp=True   
)