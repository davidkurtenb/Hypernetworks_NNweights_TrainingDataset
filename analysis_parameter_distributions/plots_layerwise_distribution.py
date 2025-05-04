#################################################################
#         PROD CODE - Distribution Comparison Layerwise
#################################################################

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from matplotlib.colors import to_rgba
import seaborn as sns
from datetime import datetime

#pickle file with all hdf5 params by layer
file_path ='/homes/dkurtenb/projects/HyperGENAI/training_dataset_build/outputs/lenet5_onevall_10000_FINAL/outputs/pickle_files/lenet5_onevall_wghts_bylayer_class_20250502_102427.pkl'


with open(file_path, 'rb') as file:
    results = pickle.load(file)

dt = datetime.now()
dt_str = dt.strftime('%Y%m%d_%H%M%S')
output_dir = '/homes/dkurtenb/projects/HyperGENAI/training_dataset_build/outputs/lenet5_onevall_10000_FINAL/distribution_comp/layer_distros'
os.makedirs(output_dir, exist_ok=True)

# Extract class names
class_names=[]
class_names = [item['class'] for item in results]
print(f"Classes found: {class_names}")

# Find common layers across all classes
common_layers = set()
for item in results:
    common_layers.update(item['parameters'].keys())
common_layers = list(common_layers)
print(f"Found {len(common_layers)} layers")

# Assign a distinct color to each class
colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
class_colors = {class_name: color for class_name, color in zip(class_names, colors)}

# Set up plotting style
plt.style.use('ggplot')
sns.set_context("talk")

# Function to create histograms for each layer
def plot_layer_histograms(layer_name):
    plt.figure(figsize=(14, 8))
    
    # Dictionary to track min/max values across all classes for this layer
    min_val = float('inf')
    max_val = float('-inf')
    
    # First pass to find overall min/max values for consistent bins
    for item in results:
        class_name = item['class']
        if layer_name in item['parameters']:
            tensor = item['parameters'][layer_name]
            if isinstance(tensor, list):
                tensor = np.concatenate([np.array(t).flatten() for t in tensor])
            elif hasattr(tensor, 'detach'):
                tensor = tensor.detach().numpy()
            min_val = min(min_val, np.min(tensor))
            max_val = max(max_val, np.max(tensor))
    
    # Create histogram bins
    bins = np.linspace(min_val, max_val, 50)
    
    # Plot each class
    for item in results:
        class_name = item['class']
        if layer_name in item['parameters']:
            tensor = item['parameters'][layer_name]
            if isinstance(tensor, list):
                tensor = np.concatenate([np.array(t).flatten() for t in tensor])
            elif hasattr(tensor, 'detach'):
                tensor = tensor.detach().numpy()

            # Create histogram with alpha for better visibility when overlapping
            color = class_colors[class_name]
            alpha = 0.7  # Adjust transparency
            
            plt.hist(tensor, bins=bins, alpha=alpha, label=class_name, 
                     color=to_rgba(color, alpha), edgecolor='black', linewidth=0.5)
    
    # Add summary statistics to the plot
    stat_text = []
    for item in results:
        class_name = item['class']
        if layer_name in item['parameters']:
            tensor = item['parameters'][layer_name]
            if isinstance(tensor, list):
                tensor = np.concatenate([np.array(t).flatten() for t in tensor])
            elif hasattr(tensor, 'detach'):
                tensor = tensor.detach().numpy()
            
            mean = np.mean(tensor)
            std = np.std(tensor)
            stat_text.append(f"{class_name}: mean={mean:.4f}, std={std:.4f}")
    
    # Add title and labels
    plt.title(f'Parameter Distribution for {layer_name}')
    plt.xlabel('Parameter Value')
    plt.ylabel('Frequency')
    plt.legend(loc='best')
    
    # Add stats as text
    plt.figtext(0.5, 0.01, '\n'.join(stat_text), ha='center', fontsize=10, 
                bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    # Save the plot
    clean_layer_name = layer_name.replace('/', '_')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'histogram_{clean_layer_name}.png'), dpi=300)
    plt.close()

# Create dual plots showing both standard and log-scale histograms
def plot_dual_histograms(layer_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Find min/max values
    min_val = float('inf')
    max_val = float('-inf')
    for item in results:
        if layer_name in item['parameters']:
            tensor = item['parameters'][layer_name]
            if isinstance(tensor, list):
                tensor = np.concatenate([np.array(t).flatten() for t in tensor])
            elif hasattr(tensor, 'detach'):
                tensor = tensor.detach().numpy()
            
            min_val = min(min_val, np.min(tensor))
            max_val = max(max_val, np.max(tensor))
    
    bins = np.linspace(min_val, max_val, 50)
    
    # Plot regular histogram on left
    for item in results:
        class_name = item['class']
        if layer_name in item['parameters']:
            tensor = item['parameters'][layer_name]
            if isinstance(tensor, list):
                tensor = np.concatenate([np.array(t).flatten() for t in tensor])
            elif hasattr(tensor, 'detach'):
                tensor = tensor.detach().numpy()
            
            color = class_colors[class_name]
            ax1.hist(tensor, bins=bins, alpha=0.7, label=class_name, 
                     color=to_rgba(color, 0.7), edgecolor='black', linewidth=0.5)
    
    ax1.set_title(f'Linear Scale: {layer_name}')
    ax1.set_xlabel('Parameter Value')
    ax1.set_ylabel('Frequency')
    
    # Plot log-scale histogram on right
    for item in results:
        class_name = item['class']
        if layer_name in item['parameters']:
            tensor = item['parameters'][layer_name]
            if isinstance(tensor, list):
                tensor = np.concatenate([np.array(t).flatten() for t in tensor])
            elif hasattr(tensor, 'detach'):
                tensor = tensor.detach().numpy()
            
            color = class_colors[class_name]
            ax2.hist(tensor, bins=bins, alpha=0.7, label=class_name, 
                     color=to_rgba(color, 0.7), edgecolor='black', linewidth=0.5)
    
    ax2.set_title(f'Log Scale: {layer_name}')
    ax2.set_xlabel('Parameter Value')
    ax2.set_yscale('log')
    ax2.legend(loc='best')
    
    # Add summary statistics
    stat_texts = []
    for item in results:
        class_name = item['class']
        if layer_name in item['parameters']:
            tensor = item['parameters'][layer_name]
            if isinstance(tensor, list):
                tensor = np.concatenate([np.array(t).flatten() for t in tensor])
            elif hasattr(tensor, 'detach'):
                tensor = tensor.detach().numpy()
            
            mean = np.mean(tensor)
            std = np.std(tensor)
            stat_texts.append(f"{class_name}: mean={mean:.4f}, std={std:.4f}")
    
    plt.figtext(0.5, 0.01, '\n'.join(stat_texts), ha='center', fontsize=10, 
                bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    # Save the plot
    clean_layer_name = layer_name.replace('/', '_')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Make room for the text at the bottom
    plt.savefig(os.path.join(output_dir, f'dual_histogram_{clean_layer_name}.png'), dpi=300)
    plt.close()

# Optional: Create a density plot for comparing distributions
def plot_density(layer_name):
    plt.figure(figsize=(12, 8))
    
    # For each class, create a density plot
    for item in results:
        class_name = item['class']
        if layer_name in item['parameters']:
            tensor = item['parameters'][layer_name]
            if isinstance(tensor, list):
                tensor = np.concatenate([np.array(t).flatten() for t in tensor])
            elif hasattr(tensor, 'detach'):
                tensor = tensor.detach().numpy()
            
            # Use kernel density estimation
            sns.kdeplot(tensor, label=class_name, color=class_colors[class_name], fill=True, alpha=0.3)
    
    plt.title(f'Parameter Density for {layer_name}')
    plt.xlabel('Parameter Value')
    plt.ylabel('Density')
    plt.legend()
    
    # Save the plot
    clean_layer_name = layer_name.replace('/', '_')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'density_{clean_layer_name}.png'), dpi=300)
    plt.close()

# Create a summary plot of all layers for a specific class
def plot_class_summary(class_name):
    # Find the class data
    class_data = next((item for item in results if item['class'] == class_name), None)
    if not class_data:
        print(f"Class {class_name} not found")
        return
    
    # Get all layers for this class
    layers = list(class_data['parameters'].keys())
    num_layers = len(layers)
    
    # Create a grid of plots (4 layers per row)
    cols = min(5, num_layers)
    rows = (num_layers + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, layer in enumerate(layers):
        if i < len(axes):
            tensor = class_data['parameters'][layer]
            if isinstance(tensor, list):
                tensor = np.concatenate([np.array(t).flatten() for t in tensor])
            elif hasattr(tensor, 'detach'):
                tensor = tensor.detach().numpy()
            
            axes[i].hist(tensor, bins=30, color=class_colors[class_name], alpha=0.7)
            axes[i].set_title(layer)
            
            # Add basic stats
            mean = np.mean(tensor)
            std = np.std(tensor)
            axes[i].text(0.05, 0.95, f"Mean: {mean:.4f}\nStd: {std:.4f}", 
                         transform=axes[i].transAxes, va='top', fontsize=8,
                         bbox={"facecolor":"white", "alpha":0.8, "pad":3})
    
    # Hide unused subplots
    for i in range(num_layers, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Parameter Distributions for {class_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(output_dir, f'class_summary_{class_name}.png'), dpi=300)
    plt.close()

# Create summary plots for each class
# Create plots for each layer
for i, layer in enumerate(common_layers):
    print(f"Processing layer {i+1}/{len(common_layers)}: {layer}")
    plot_layer_histograms(layer)
    plot_dual_histograms(layer)
    plot_density(layer)

print(f"All plots saved to {output_dir}")

# Create summary plots for each class
for class_name in class_names:
    plot_class_summary(class_name)