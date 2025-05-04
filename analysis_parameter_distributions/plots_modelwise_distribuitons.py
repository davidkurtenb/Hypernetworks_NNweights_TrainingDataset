import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from collections import defaultdict
from collections import Counter


# Load the data
file_path = '/homes/dkurtenb/projects/HyperGENAI/training_dataset_build/outputs/lenet5_onevall_10000_FINAL/outputs/pickle_files/lenet5_onevall_wghts_bymodel_20250502_102356.pkl'
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Create dictionary to store weights by class
class_weights = defaultdict(list)


for x in range(len(data)):
    class_nm_split = data[x]['model'].split('_')[:-1]
    class_nm = '_'.join(class_nm_split)

    tensor = data[x]['parameters_flat']
    class_weights[class_nm].extend(tensor.numpy())

# Set up the plot parameters
plt.figure(figsize=(15, 10))
weight_range = (-0.2, 0.2) 
bins = 500
save_dir = '/homes/dkurtenb/projects/HyperGENAI/training_dataset_build/outputs/lenet5_onevall_10000_FINAL/distribution_comp' 
os.makedirs(save_dir, exist_ok=True)

# Create a single figure with subplots for comparison
num_classes = len(class_weights)
rows = int(np.ceil(num_classes / 2))


plt.figure(figsize=(16, 4 * rows))

for i, (class_name, weights) in enumerate(class_weights.items()):
    weights_array = np.array(weights)
    filtered = weights_array[(weights_array >= weight_range[0]) & (weights_array <= weight_range[1])]
    
    if len(filtered) == 0:
        continue
    
    plt.subplot(rows, 2, i+1)
    plt.hist(filtered, bins=bins, alpha=0.7)
    plt.title(f'Class: {class_name}')
    plt.grid(alpha=0.3)
    
    # Add mini statistics
    plt.text(0.05, 0.95, f"Mean: {np.mean(filtered):.4f}, StdDev: {np.std(filtered):.4f}", 
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6))

plt.tight_layout()
plt.show()
plt.savefig(os.path.join(save_dir, 'param_distro_modelwise_byclass.png'), dpi=300)
plt.close()

print(f"Plots saved to: {save_dir}")