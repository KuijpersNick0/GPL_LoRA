import json
import matplotlib.pyplot as plt
import numpy as np

def plot_loss_over_steps(json_file_path, step_interval=100):
    # Read the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Extract steps and loss values
    steps = data[0]['steps']
    loss = data[0]['loss']
    
    # Subset data to reduce number of points
    steps_subset = steps[::step_interval]
    loss_subset = loss[::step_interval]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(steps_subset, loss_subset, color='blue', linewidth=1, marker='o', markersize=2, label='Loss')
    
    # Set labels and title
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Loss over Steps')
    
    # Add grid
    plt.grid(True)
    
    # Show legend
    plt.legend()
    
    # Show plot
    plt.show()
 
json_file_path = 'C:/Users/Siemens/Documents/TFE_Nick_Kuijpers/LoRa_GPL/GPL_LoRA/output/scifact/distilbert-base-uncased/training_data_50000.json'  
plot_loss_over_steps(json_file_path)
