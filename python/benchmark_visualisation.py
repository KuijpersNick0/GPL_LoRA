import os
import json
import matplotlib.pyplot as plt

# Function to read JSON files and extract evaluation metrics
def read_evaluation_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    gpl_steps = int(data["gpl_steps"])
    ndcg = data["ndcg"]["NDCG@3"]
    map_ = data["map"]["MAP@3"]
    recall = data["recall"]["Recall@3"]
    precision = data["precicion"]["P@3"]
    mrr = data["mrr"]["MRR@3"]
    return gpl_steps, ndcg, map_, recall, precision, mrr

# Function to plot a single metric over different training steps
def plot_single_metric_over_steps(metric_name, steps, metric_data):
    plt.figure(figsize=(10, 6))
    plt.plot(steps, metric_data, marker='o', linestyle='None')
    plt.xlabel("Training Steps")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} at k=3 over Training Steps")
    plt.grid(True)
    plt.show()

# Path to the directory containing evaluation JSON files
evaluation_dir = "C:/Users/Siemens/Documents/TFE_Nick_Kuijpers/LoRa_GPL/GPL_LoRA/evaluation/KV/LoRa_r8_a128/fiqa_scifact/GPL/fiqa-tsdae-msmarco-distilbert-margin-mse"

# List to store data for plotting
steps = []
ndcg_data = []
map_data = []
recall_data = []
precision_data = []
mrr_data = []


# Iterate over each JSON file in the directory
for filename in os.listdir(evaluation_dir):
    if filename.endswith(".json"):
        json_file = os.path.join(evaluation_dir, filename)
        gpl_steps, ndcg, map_, recall, precision, mrr = read_evaluation_json(json_file)
        steps.append(gpl_steps)
        ndcg_data.append(ndcg)
        map_data.append(map_)
        recall_data.append(recall)
        precision_data.append(precision)
        mrr_data.append(mrr)

# Plot NDCG@10
plot_single_metric_over_steps("NDCG@3", steps, ndcg_data)

# Plot MAP@10
plot_single_metric_over_steps("MAP@3", steps, map_data)

# Plot Recall@10
plot_single_metric_over_steps("Recall@10", steps, recall_data)

# Plot Precision@10
plot_single_metric_over_steps("Precision@10", steps, precision_data)

# Plot MRR@10
plot_single_metric_over_steps("MRR@10", steps, mrr_data)