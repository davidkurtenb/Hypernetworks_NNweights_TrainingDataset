#############################################################
#          PROD CODE
############################################################

import numpy as np
import pickle
from scipy.stats import entropy
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
from scipy.spatial.distance import pdist, squareform, jensenshannon
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import os

def flatten_data(file):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    flattened_data = []  

    for i in range(len(data)):
        flattened_params = {
            'class': data[i]['class'],
            'flattened_params': {
                'conv2d_layers': np.concatenate(data[i]['parameters']['conv2d']),
                'conv2d_1_layers': np.concatenate(data[i]['parameters']['conv2d_1']),
                'conv2d_2_layers': np.concatenate(data[i]['parameters']['conv2d_2']),
                'dense_layers': np.concatenate(data[i]['parameters']['dense']),
                'dense_1_layers': np.concatenate(data[i]['parameters']['dense_1'])
            }
        }
        flattened_data.append(flattened_params)


    print(f"Processed {len(flattened_data)} classes")

    return flattened_data

def analyze_layer_distributions(models, layer_name,dir_name,log_file):

    def log_print(message):
        print(message)
        log_file.write(message + '\n')


    log_print(f"\n===== ANALYZING {layer_name} DISTRIBUTIONS =====")
    
    layer_params = []
    model_classes = []
    
    for model in models:
        layer_params.append(np.array(model['flattened_params'][layer_name]))
        model_classes.append(model['class'])

    #print(len(layer_params))
    #for x in layer_params:
    #    print(len(x))
    #print(model_classes)
    layer_params = np.array(layer_params)
    log_print(f"Shape of layer parameters: {layer_params.shape}")
    
    num_bins = 30
    param_histograms = []
    
    all_values = layer_params.flatten()
    global_min = np.min(all_values)
    global_max = np.max(all_values)
    
    for i, params in enumerate(layer_params):
        hist, _ = np.histogram(params, bins=num_bins, range=(global_min, global_max), density=True)
        hist = hist + 1e-10
        hist = hist / np.sum(hist)
        param_histograms.append(hist)
    
    js_distances = np.zeros((len(param_histograms), len(param_histograms)))
    
    for i in range(len(param_histograms)):
        for j in range(i+1, len(param_histograms)):
            js_dist = jensenshannon(param_histograms[i], param_histograms[j])
            js_distances[i, j] = js_dist
            js_distances[j, i] = js_dist
    
    js_flat = js_distances[np.triu_indices_from(js_distances, k=1)]
    avg_js = np.mean(js_flat)
    std_js = np.std(js_flat)
    min_js = np.min(js_flat)
    max_js = np.max(js_flat)
    
    log_print(f"Average JS divergence: {avg_js:.4f}")
    log_print(f"Standard deviation of JS divergence: {std_js:.4f}")
    log_print(f"Min JS divergence: {min_js:.4f}")
    log_print(f"Max JS divergence: {max_js:.4f}")
    
    #heatmap divergence
    plt.figure(figsize=(10, 8))
    sns.heatmap(js_distances, cmap='YlGnBu', xticklabels=model_classes, yticklabels=model_classes, 
                annot=True, fmt='.3f')
    plt.title(f'Jensen-Shannon Divergence for {layer_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(dir_name,f'{layer_name}_js_divergence.png'))
    plt.close()
    
    
    #parameter distributions
    plt.figure(figsize=(12, 8))
    bin_edges = np.linspace(global_min, global_max, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    """
    for i, hist in enumerate(param_histograms):
        plt.plot(bin_centers, hist, label=model_classes[i])
    
    plt.title(f'Parameter Distributions for {layer_name}')
    plt.xlabel('Parameter Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{layer_name}_distributions.png')
    plt.close()
    """
    max_js_val = 0
    min_js_val = float('inf')
    max_js_pair = (0, 0)
    min_js_pair = (0, 0)
    
    for i in range(len(param_histograms)):
        for j in range(i+1, len(param_histograms)):
            if js_distances[i, j] > max_js_val:
                max_js_val = js_distances[i, j]
                max_js_pair = (i, j)
            if js_distances[i, j] < min_js_val:
                min_js_val = js_distances[i, j]
                min_js_pair = (i, j)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    i, j = max_js_pair
    plt.plot(bin_centers, param_histograms[i], label=model_classes[i])
    plt.plot(bin_centers, param_histograms[j], label=model_classes[j])
    plt.title(f'Most Different (JS={max_js_val:.3f})')
    plt.xlabel('Parameter Value')
    plt.ylabel('Probability Density')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    i, j = min_js_pair
    plt.plot(bin_centers, param_histograms[i], label=model_classes[i])
    plt.plot(bin_centers, param_histograms[j], label=model_classes[j])
    plt.title(f'Most Similar (JS={min_js_val:.3f})')
    plt.xlabel('Parameter Value')
    plt.ylabel('Probability Density')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(dir_name,f'{layer_name}_extreme_pairs.png'))
    plt.close()
    
    log_print("Performing PCA for clustering...")
    pca = PCA(n_components=min(8, len(model_classes)-1))
    pca_result = pca.fit_transform(layer_params)
    
    log_print(f"PCA result shape: {pca_result.shape}")
    log_print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    log_print("Finding optimal number of clusters...")
    silhouette_scores = []
    K_range = range(2, min(6, len(model_classes)))
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(pca_result)
        score = silhouette_score(pca_result, cluster_labels)
        silhouette_scores.append(score)
        log_print(f"K={k}, Silhouette Score: {score:.4f}")
    
    if silhouette_scores:
        best_k = K_range[np.argmax(silhouette_scores)]
        log_print(f"Best number of clusters: {best_k}")
        
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(pca_result)
        
        cluster_counts = np.bincount(cluster_labels)
        for i, count in enumerate(cluster_counts):
            log_print(f"Cluster {i}: {count} models")
            classes_in_cluster = [model_classes[j] for j in range(len(cluster_labels)) if cluster_labels[j] == i]
            log_print(f"  Classes in cluster {i}: {', '.join(classes_in_cluster)}")
            
        plt.figure(figsize=(10, 8))
        
        if pca_result.shape[1] <= 2:
            if pca_result.shape[1] == 1:
                plot_data = np.column_stack((pca_result, np.zeros(pca_result.shape[0])))
            else:
                plot_data = pca_result
                
            for i in range(best_k):
                cluster_points = plot_data[cluster_labels == i]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}')
            
            for i, (x, y) in enumerate(plot_data):
                plt.annotate(model_classes[i], (x, y), textcoords="offset points", 
                             xytext=(0, 5), ha='center')
        else:
            plot_data = pca_result[:, :2]
            
            for i in range(best_k):
                cluster_points = plot_data[cluster_labels == i]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}')
            
            for i, (x, y) in enumerate(plot_data):
                plt.annotate(model_classes[i], (x, y), textcoords="offset points", 
                             xytext=(0, 5), ha='center')
        
        plt.title(f'K-means Clustering of {layer_name} (k={best_k})')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(dir_name,f'{layer_name}_clusters.png'))
        plt.close()
        
        plt.figure(figsize=(8, 5))
        plt.plot(K_range, silhouette_scores, 'o-')
        plt.axvline(best_k, color='r', linestyle='--', label=f'Best k={best_k}')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title(f'Optimal Cluster Count for {layer_name}')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(dir_name,f'{layer_name}_silhouette.png'))
        plt.close()
        
        cluster_matrix = np.zeros((len(model_classes), best_k))
        for i, label in enumerate(cluster_labels):
            cluster_matrix[i, label] = 1
            
        plt.figure(figsize=(8, 6))
        sns.heatmap(cluster_matrix, cmap='binary', xticklabels=[f'Cluster {i}' for i in range(best_k)], 
                    yticklabels=model_classes)
        plt.title(f'Cluster Membership for {layer_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(dir_name,f'{layer_name}_membership.png'))
        plt.close()
    
    # Calculate distribution-based entropy metrics
    all_params = layer_params.flatten()
    
    # Calculate overall distribution entropy
    hist, _ = np.histogram(all_params, bins=100, density=True)
    hist = hist + 1e-10
    hist = hist / np.sum(hist)
    overall_entropy = entropy(hist)
    
    log_print(f"Overall parameter distribution entropy: {overall_entropy:.4f}")
    
    # Calculate per-class entropy
    class_entropies = []
    for i, params in enumerate(layer_params):
        hist, _ = np.histogram(params, bins=100, density=True)
        hist = hist + 1e-10
        hist = hist / np.sum(hist)
        class_entropy = entropy(hist)
        class_entropies.append((model_classes[i], class_entropy))
    
    plt.figure(figsize=(10, 6))
    class_names = [item[0] for item in class_entropies]
    entropy_values = [item[1] for item in class_entropies]
    
    plt.bar(class_names, entropy_values)
    plt.axhline(overall_entropy, color='r', linestyle='--', label=f'Overall: {overall_entropy:.3f}')
    plt.xlabel('Class')
    plt.ylabel('Entropy')
    plt.title(f'Parameter Distribution Entropy by Class for {layer_name}')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dir_name,f'{layer_name}_entropy.png'))
    plt.close()
    
    log_print("Class-wise entropy values:")
    for class_name, class_entropy in class_entropies:
        log_print(f"  {class_name}: {class_entropy:.4f}")
    
    distances = pdist(layer_params, metric='euclidean')
    dist_matrix = squareform(distances)
    
    avg_dist = np.mean(distances)
    std_dist = np.std(distances)
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    
    cos_sim = cosine_similarity(layer_params)
    upper_indices = np.triu_indices_from(cos_sim, k=1)
    cos_values = cos_sim[upper_indices]
    avg_sim = np.mean(cos_values)
    std_sim = np.std(cos_values)
    min_sim = np.min(cos_values)
    max_sim = np.max(cos_values)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(dist_matrix, cmap='viridis', xticklabels=model_classes, yticklabels=model_classes, 
                annot=True, fmt='.2f')
    plt.title(f'Pairwise Euclidean Distance for {layer_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(dir_name,f'{layer_name}_distance_matrix.png'))
    plt.close()
    
    return {
        'avg_js': avg_js,
        'std_js': std_js,
        'min_js': min_js,
        'max_js': max_js,
        'overall_entropy': overall_entropy,
        'class_entropies': dict(class_entropies),
        'js_distances': js_distances,
        'model_classes': model_classes,
        'avg_dist': avg_dist,
        'std_dist': std_dist,
        'min_dist': min_dist,
        'max_dist': max_dist,
        'avg_sim': avg_sim,
        'best_k': best_k if silhouette_scores else None,
        'silhouette_scores': silhouette_scores if silhouette_scores else None,
        'cluster_labels': cluster_labels.tolist() if silhouette_scores else None
    }

def compare_layer_distributions(layer_stats,dir_name,log_file):

    def log_print(message):
        print(message)
        log_file.write(message + '\n')

    log_print("\n===== COMPARING DISTRIBUTIONS ACROSS LAYERS =====")
    
    layer_names = list(layer_stats.keys())
    
    avg_js_values = [layer_stats[layer]['avg_js'] for layer in layer_names]
    max_js_values = [layer_stats[layer]['max_js'] for layer in layer_names]
    min_js_values = [layer_stats[layer]['min_js'] for layer in layer_names]
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(layer_names))
    width = 0.3
    
    plt.bar(x - width, min_js_values, width, label='Min JS Divergence')
    plt.bar(x, avg_js_values, width, label='Avg JS Divergence')
    plt.bar(x + width, max_js_values, width, label='Max JS Divergence')
    
    plt.xlabel('Layer')
    plt.ylabel('Jensen-Shannon Divergence')
    plt.title('Distribution Divergence Across Layers')
    plt.xticks(x, layer_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dir_name,'layer_js_comparison.png'))
    plt.close()
    
    best_k_values = []
    silhouette_values = []
    
    for layer in layer_names:
        if layer_stats[layer]['best_k'] is not None:
            best_k_values.append(layer_stats[layer]['best_k'])
            silhouette_values.append(max(layer_stats[layer]['silhouette_scores']))
        else:
            best_k_values.append(0)
            silhouette_values.append(0)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(layer_names, best_k_values)
    plt.xlabel('Layer')
    plt.ylabel('Optimal Number of Clusters')
    plt.title('Best K Value by Layer')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    plt.bar(layer_names, silhouette_values)
    plt.xlabel('Layer')
    plt.ylabel('Best Silhouette Score')
    plt.title('Clustering Quality by Layer')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(dir_name,'layer_clustering_comparison.png'))
    plt.close()
    
    overall_entropies = [layer_stats[layer]['overall_entropy'] for layer in layer_names]
    
    plt.figure(figsize=(10, 6))
    plt.bar(layer_names, overall_entropies)
    plt.xlabel('Layer')
    plt.ylabel('Entropy')
    plt.title('Parameter Distribution Entropy Across Layers')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_name,'layer_entropy_comparison.png'))
    plt.close()
    
    avg_dist_values = [layer_stats[layer]['avg_dist'] for layer in layer_names]
    
    plt.figure(figsize=(10, 6))
    plt.bar(layer_names, avg_dist_values)
    plt.xlabel('Layer')
    plt.ylabel('Average Euclidean Distance')
    plt.title('Average Pairwise Distance Across Layers')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_name,'layer_distance_comparison.png'))
    plt.close()
    
    plt.figure(figsize=(15, 10))
    
    norm_js = [val / max(avg_js_values) for val in avg_js_values]
    norm_entropy = [val / max(overall_entropies) for val in overall_entropies]
    norm_dist = [val / max(avg_dist_values) for val in avg_dist_values]
    norm_silhouette = [val / max(silhouette_values) if max(silhouette_values) > 0 else 0 for val in silhouette_values]
    
    plt.subplot(2, 2, 1)
    plt.bar(layer_names, norm_js)
    plt.title('Normalized JS Divergence')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 2)
    plt.bar(layer_names, norm_entropy)
    plt.title('Normalized Entropy')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 3)
    plt.bar(layer_names, norm_dist)
    plt.title('Normalized Distance')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 4)
    plt.bar(layer_names, norm_silhouette)
    plt.title('Normalized Silhouette Score')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(dir_name,'layer_metrics_comparison.png'))
    plt.close()
    
    log_print("\nSummary of metrics across layers:")
    for i, layer in enumerate(layer_names):
        log_print(f"\n{layer}:")
        log_print(f"  Average JS Divergence: {avg_js_values[i]:.4f}")
        log_print(f"  Min/Max JS Divergence: {min_js_values[i]:.4f}/{max_js_values[i]:.4f}")
        log_print(f"  Distribution Entropy: {overall_entropies[i]:.4f}")
        log_print(f"  Average Distance: {avg_dist_values[i]:.4f}")
        if best_k_values[i] > 0:
            log_print(f"  Optimal Clusters: {best_k_values[i]} (Silhouette={silhouette_values[i]:.4f})")
    
    most_diverse_idx = np.argmax(avg_js_values)
    least_diverse_idx = np.argmin(avg_js_values)
    
    best_clustering_idx = np.argmax(silhouette_values) if max(silhouette_values) > 0 else -1
    
    log_print(f"\nMost diverse layer (by JS): {layer_names[most_diverse_idx]} (JS={avg_js_values[most_diverse_idx]:.4f})")
    log_print(f"Least diverse layer (by JS): {layer_names[least_diverse_idx]} (JS={avg_js_values[least_diverse_idx]:.4f})")
    
    if best_clustering_idx >= 0:
        log_print(f"Best clustering layer: {layer_names[best_clustering_idx]} " +
              f"(Silhouette={silhouette_values[best_clustering_idx]:.4f}, K={best_k_values[best_clustering_idx]})")
    

    model_classes = layer_stats[layer_names[0]]['model_classes']
    n_classes = len(model_classes)
    n_layers = len(layer_names)
    
    cluster_assignment_matrix = np.zeros((n_classes, n_layers))
    
    for i, layer in enumerate(layer_names):
        if layer_stats[layer]['cluster_labels'] is not None:
            for j, label in enumerate(layer_stats[layer]['cluster_labels']):
                cluster_assignment_matrix[j, i] = label
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(cluster_assignment_matrix, cmap='tab10', annot=True, fmt='.0f',
                xticklabels=layer_names, yticklabels=model_classes)
    plt.title('Cluster Assignments Across Layers')
    plt.xlabel('Layer')
    plt.ylabel('Class')
    plt.tight_layout()
    plt.savefig(os.path.join(dir_name,'layer_cluster_assignments.png'))
    plt.close()
    
    js_div_by_layer = []
    for layer in layer_names:
        js_div = layer_stats[layer]['js_distances']
        upper_indices = np.triu_indices_from(js_div, k=1)
        js_div_by_layer.append(js_div[upper_indices])
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(js_div_by_layer, labels=layer_names)
    plt.ylabel('JS Divergence')
    plt.title('Distribution of JS Divergences by Layer')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_name,'js_divergence_boxplot.png'))
    plt.close()

def main(models,dir_name):
    
    os.makedirs(dir_name, exist_ok=True)

    log_file = open(os.path.join(dir_name, 'analysis_report.txt'), 'w')

    def log_print(message):
        print(message)
        log_file.write(message + '\n')

    layer_names = list(models[0]['flattened_params'].keys())
    log_print(f"Found {len(layer_names)} layers: {', '.join(layer_names)}")
    
    layer_stats = {}
    for layer_name in layer_names:
        stats = analyze_layer_distributions(models, layer_name, dir_name, log_file)
        layer_stats[layer_name] = stats
    
    compare_layer_distributions(layer_stats, dir_name, log_file)
    
    log_print("\nAnalysis complete!")
    log_file.close()
    
    print("\nAnalysis complete!")

file_path =  '/homes/dkurtenb/projects/HyperGENAI/training_dataset_build/outputs/lenet5_onevall_10000_FINAL/outputs/pickle_files/lenet5_onevall_wghts_bylayer_class_20250502_102427.pkl'
save_dir = '/homes/dkurtenb/projects/HyperGENAI/training_dataset_build/outputs/lenet5_onevall_10000_FINAL/distribution_comp/layerwise_analysis_comparison' 
flat_data= flatten_data(file_path)
main(flat_data, save_dir)