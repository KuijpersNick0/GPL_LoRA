import os
import json
import matplotlib.pyplot as plt

# Function to read JSON files and extract evaluation metrics
def read_evaluation_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    gpl_steps = int(data["gpl_steps"])
    ndcg = data["ndcg"]["NDCG@1"]
    map_ = data["map"]["MAP@10"]
    recall = data["recall"]["Recall@10"]
    precision = data["precicion"]["P@10"]
    mrr = data["mrr"]["MRR@10"]
    return gpl_steps, ndcg, map_, recall, precision, mrr

# Function to plot a single metric for multiple models
def plot_metric_over_steps(metric_name, models, max_steps=20000):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Color for each model
    plt.figure(figsize=(10, 6))
    for i, (model_name, model_path) in enumerate(models.items()):
            step_metric_pairs = []

            # Iterate over each JSON file in the directory
            for filename in os.listdir(model_path):
                if filename.endswith(".json"):
                    json_file = os.path.join(model_path, filename)
                    gpl_steps, ndcg, map_, recall, precision, mrr = read_evaluation_json(json_file)
                    current_metric = {
                        "NDCG@1": ndcg,
                        "MAP@10": map_,
                        "Recall@10": recall,
                        "Precision@10": precision,
                        "MRR@10": mrr
                    }
                    if gpl_steps <= max_steps:  # Only include data up to the specified step limit
                        step_metric_pairs.append((gpl_steps, current_metric[metric_name]))

            # Sort data by steps before plotting
            step_metric_pairs.sort()  # Default sorts by first element of tuple, which is gpl_steps
            steps, metric_data = zip(*step_metric_pairs)  # Unzip into separate lists

            # Plot metric for current model
            plt.plot(steps, metric_data, marker='o', linestyle='-', color=colors[i % len(colors)], label=f"{model_name}")

    plt.xlabel("Training Steps")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} at k=1 over Training Steps")
    plt.grid(True)
    plt.legend()
    plt.show()

# Dictionary of model names and their corresponding directories
models = {
    "Model1_with_r4": "C:/Users/Siemens/Documents/TFE_Nick_Kuijpers/LoRa_GPL/GPL_LoRA/evaluation/LoRa_r4_a32/Siemens/distilbert-base-uncased",
    "Model2_r8": "C:/Users/Siemens/Documents/TFE_Nick_Kuijpers/LoRa_GPL/GPL_LoRA/evaluation/LoRa_r8_a32/Siemens/distilbert-base-uncased",
    "Model3_r16": "C:/Users/Siemens/Documents/TFE_Nick_Kuijpers/LoRa_GPL/GPL_LoRA/evaluation/LoRa_r16_a32/Siemens/distilbert-base-uncased",
    # "Model4_r32": "C:/Users/Siemens/Documents/TFE_Nick_Kuijpers/LoRa_GPL/GPL_LoRA/evaluation/LoRa_r32_a32/scifact/distilbert-base-uncased",
    # "Model5_r64": "C:/Users/Siemens/Documents/TFE_Nick_Kuijpers/LoRa_GPL/GPL_LoRA/evaluation/LoRa_r64_a32/scifact/distilbert-base-uncased"
    }

# Call function to plot metrics for all models
plot_metric_over_steps('NDCG@1', models)
