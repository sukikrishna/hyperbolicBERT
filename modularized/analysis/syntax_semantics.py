"""
Analysis of syntactic vs semantic information in BERT's representations.

This module implements methods to analyze how syntactic and semantic information
is represented across BERT layers, comparing hyperbolic and Euclidean geometries.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
import seaborn as sns
import os

from utils.hyperbolic_utils import (
    init_poincare_ball,
    euclidean_to_poincare,
    poincare_distance,
    mobius_addition,
    mobius_matrix_multiplication
)

def compute_nn_overlap(embeddings, syntax_distances, semantic_distances, k=10):
    """
    Compute the overlap between syntactic and semantic nearest neighbors.
    
    Args:
        embeddings: Tensor of shape (n, dim) with word embeddings
        syntax_distances: Matrix of shape (n, n) with syntactic distances
        semantic_distances: Matrix of shape (n, n) with semantic distances
        k: Number of nearest neighbors to consider
        
    Returns:
        float: Jaccard similarity between syntactic and semantic nearest neighbors
    """
    n = embeddings.shape[0]
    jaccard_similarities = []
    
    for i in range(n):
        # Get indices of k nearest syntactic neighbors (excluding self)
        syntax_indices = torch.argsort(syntax_distances[i])
        syntax_nn = set(syntax_indices[1:k+1].cpu().numpy())  # exclude self
        
        # Get indices of k nearest semantic neighbors (excluding self)
        semantic_indices = torch.argsort(semantic_distances[i])
        semantic_nn = set(semantic_indices[1:k+1].cpu().numpy())  # exclude self
        
        # Compute Jaccard similarity
        intersection = len(syntax_nn.intersection(semantic_nn))
        union = len(syntax_nn.union(semantic_nn))
        jaccard = intersection / union if union > 0 else 0
        jaccard_similarities.append(jaccard)
    
    return np.mean(jaccard_similarities)

def analyze_distance_distributions(syntax_distances, semantic_distances):
    """
    Analyze the distributions of syntactic and semantic distances.
    
    Args:
        syntax_distances: Matrix of shape (n, n) with syntactic distances
        semantic_distances: Matrix of shape (n, n) with semantic distances
        
    Returns:
        dict: Statistics of distance distributions
    """
    # Flatten distance matrices (excluding self-distances)
    n = syntax_distances.shape[0]
    
    # Get upper triangular part (excluding diagonal)
    indices = torch.triu_indices(n, n, 1)
    syntax_flat = syntax_distances[indices[0], indices[1]].cpu().numpy()
    semantic_flat = semantic_distances[indices[0], indices[1]].cpu().numpy()
    
    # Compute statistics
    stats = {
        'syntax': {
            'mean': np.mean(syntax_flat),
            'std': np.std(syntax_flat),
            'min': np.min(syntax_flat),
            'max': np.max(syntax_flat),
            'median': np.median(syntax_flat)
        },
        'semantic': {
            'mean': np.mean(semantic_flat),
            'std': np.std(semantic_flat),
            'min': np.min(semantic_flat),
            'max': np.max(semantic_flat),
            'median': np.median(semantic_flat)
        }
    }
    
    # Compute correlation
    correlation, p_value = spearmanr(syntax_flat, semantic_flat)
    stats['correlation'] = {
        'spearman': correlation,
        'p_value': p_value
    }
    
    return stats

def analyze_nearest_neighbor_distances(embeddings, syntax_distances, semantic_distances, k=10):
    """
    Analyze how distances to nearest neighbors differ between syntax and semantics.
    
    Args:
        embeddings: Tensor of shape (n, dim) with word embeddings
        syntax_distances: Matrix of shape (n, n) with syntactic distances
        semantic_distances: Matrix of shape (n, n) with semantic distances
        k: Number of nearest neighbors to consider
        
    Returns:
        dict: Statistics about nearest neighbor distances
    """
    n = embeddings.shape[0]
    syntax_nn_distances = []
    semantic_nn_distances = []
    
    for i in range(n):
        # Get distances to k nearest syntactic neighbors (excluding self)
        syntax_indices = torch.argsort(syntax_distances[i])
        syntax_nn_dist = syntax_distances[i, syntax_indices[1:k+1]].cpu().numpy()
        syntax_nn_distances.append(syntax_nn_dist)
        
        # Get distances to k nearest semantic neighbors (excluding self)
        semantic_indices = torch.argsort(semantic_distances[i])
        semantic_nn_dist = semantic_distances[i, semantic_indices[1:k+1]].cpu().numpy()
        semantic_nn_distances.append(semantic_nn_dist)
    
    # Compute average distance to nearest neighbors
    avg_syntax_nn_dist = np.mean([np.mean(dist) for dist in syntax_nn_distances])
    avg_semantic_nn_dist = np.mean([np.mean(dist) for dist in semantic_nn_distances])
    
    return {
        'avg_syntax_nn_distance': avg_syntax_nn_dist,
        'avg_semantic_nn_distance': avg_semantic_nn_dist,
        'ratio': avg_syntax_nn_dist / avg_semantic_nn_dist
    }

def extract_syntactic_semantic_distances(model, tokenizer, sentences, device="cuda", batch_size=8):
    """
    Extract syntactic and semantic distances from BERT layers.
    
    Args:
        model: A BERT model with output_hidden_states=True
        tokenizer: BERT tokenizer
        sentences: List of sentences to analyze
        device: Device to run computations on
        batch_size: Batch size for processing
        
    Returns:
        dict: Syntactic and semantic distances for each layer
    """
    model.eval()
    num_layers = model.config.num_hidden_layers + 1  # +1 for embeddings
    
    # Initialize hyperbolic manifold
    ball = init_poincare_ball()
    
    # Results will contain distances for each layer
    results = {
        'euclidean': [[] for _ in range(num_layers)],
        'hyperbolic': [[] for _ in range(num_layers)]
    }
    
    # Process sentences in batches
    for batch_start in tqdm(range(0, len(sentences), batch_size), desc="Extracting distances"):
        batch_sentences = sentences[batch_start:batch_start + batch_size]
        
        # Tokenize sentences
        inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states
            
            # Process each layer
            for layer_idx in range(num_layers):
                layer_hidden_states = hidden_states[layer_idx]
                
                # Extract token embeddings (exclude special tokens)
                attention_mask = inputs['attention_mask']
                token_embeddings = []
                
                for i, sent_length in enumerate(attention_mask.sum(dim=1)):
                    # Get embeddings for actual tokens (exclude [CLS], [SEP], and padding)
                    sent_embeddings = layer_hidden_states[i, 1:sent_length-1]
                    token_embeddings.append(sent_embeddings)
                
                # Compute distances for each sentence
                for sent_embeddings in token_embeddings:
                    if sent_embeddings.shape[0] < 3:  # Skip sentences that are too short
                        continue
                    
                    # Compute Euclidean distances
                    n = sent_embeddings.shape[0]
                    euclidean_distances = torch.zeros((n, n), device=device)
                    for i in range(n):
                        for j in range(i+1, n):
                            euclidean_distances[i, j] = torch.norm(sent_embeddings[i] - sent_embeddings[j])
                            euclidean_distances[j, i] = euclidean_distances[i, j]
                    
                    # Project to Poincaré ball and compute hyperbolic distances
                    poincare_embeddings = euclidean_to_poincare(sent_embeddings, ball)
                    hyperbolic_distances = torch.zeros((n, n), device=device)
                    for i in range(n):
                        for j in range(i+1, n):
                            hyperbolic_distances[i, j] = poincare_distance(
                                poincare_embeddings[i].unsqueeze(0),
                                poincare_embeddings[j].unsqueeze(0),
                                ball
                            )
                            hyperbolic_distances[j, i] = hyperbolic_distances[i, j]
                    
                    # Store distances
                    results['euclidean'][layer_idx].append(euclidean_distances.cpu())
                    results['hyperbolic'][layer_idx].append(hyperbolic_distances.cpu())
    
    return results

def analyze_dependency_vs_semantic_correlation(distances, dependency_trees, layer_idx):
    """
    Analyze correlation between dependency distances and semantic distances.
    
    Args:
        distances: Dictionary with syntactic and semantic distances for multiple sentences
        dependency_trees: List of dependency trees (one per sentence)
        layer_idx: Layer index to analyze
        
    Returns:
        dict: Correlation statistics
    """
    euclidean_corrs = []
    hyperbolic_corrs = []
    
    for i, (euclidean_dists, hyperbolic_dists, dep_tree) in enumerate(zip(
        distances['euclidean'][layer_idx], 
        distances['hyperbolic'][layer_idx], 
        dependency_trees
    )):
        # Convert dependency tree to distance matrix
        n = euclidean_dists.shape[0]
        dep_distances = torch.zeros((n, n))
        
        # Assuming dep_tree is a list of (parent, child) tuples or an adjacency matrix
        if isinstance(dep_tree, list):
            # Convert to adjacency matrix
            adj_matrix = torch.zeros((n, n))
            for parent, child in dep_tree:
                if parent < n and child < n:  # Ensure indices are valid
                    adj_matrix[parent, child] = 1
                    adj_matrix[child, parent] = 1  # Undirected graph
        else:
            adj_matrix = dep_tree
        
        # Compute all-pairs shortest paths
        dep_distances = adj_matrix.clone()
        dep_distances[dep_distances == 0] = float('inf')
        dep_distances.fill_diagonal_(0)
        
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dep_distances[i, k] + dep_distances[k, j] < dep_distances[i, j]:
                        dep_distances[i, j] = dep_distances[i, k] + dep_distances[k, j]
        
        # Replace inf with max distance + 1
        max_dist = dep_distances[dep_distances != float('inf')].max()
        dep_distances[dep_distances == float('inf')] = max_dist + 1
        
        # Compute correlations
        # Flatten matrices (upper triangular, excluding diagonal)
        indices = torch.triu_indices(n, n, 1)
        dep_flat = dep_distances[indices[0], indices[1]].numpy()
        euc_flat = euclidean_dists[indices[0], indices[1]].numpy()
        hyp_flat = hyperbolic_dists[indices[0], indices[1]].numpy()
        
        # Compute Spearman correlations
        euc_corr, _ = spearmanr(dep_flat, euc_flat)
        hyp_corr, _ = spearmanr(dep_flat, hyp_flat)
        
        euclidean_corrs.append(euc_corr)
        hyperbolic_corrs.append(hyp_corr)
    
    return {
        'euclidean_correlation': np.mean(euclidean_corrs),
        'hyperbolic_correlation': np.mean(hyperbolic_corrs),
        'hyperbolic_advantage': np.mean(hyperbolic_corrs) - np.mean(euclidean_corrs)
    }

def visualize_embeddings_pca(embeddings, labels=None, title="PCA Visualization", save_path=None):
    """
    Visualize embeddings using PCA.
    
    Args:
        embeddings: Embeddings of shape (n, dim)
        labels: Optional labels for coloring points
        title: Plot title
        save_path: Path to save the plot, if None, plot will be displayed
    """
    # Apply PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings.cpu().numpy())
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    if labels is not None:
        # Color by labels
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                label=f'Class {label}',
                alpha=0.7
            )
        plt.legend()
    else:
        # Single color
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
    
    plt.title(title)
    plt.xlabel(f'PCA 1 (Variance: {pca.explained_variance_ratio_[0]:.2f})')
    plt.ylabel(f'PCA 2 (Variance: {pca.explained_variance_ratio_[1]:.2f})')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def visualize_embeddings_poincare(embeddings, ball, labels=None, title="Poincaré Disk Visualization", save_path=None):
    """
    Visualize embeddings in the Poincaré disk.
    
    Args:
        embeddings: Embeddings in the Poincaré ball of shape (n, dim)
        ball: Poincaré ball manifold
        labels: Optional labels for coloring points
        title: Plot title
        save_path: Path to save the plot, if None, plot will be displayed
    """
    # Apply PCA to reduce to 2D while preserving as much variance as possible
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings.cpu().numpy())
    
    # Convert back to PyTorch
    embeddings_2d = torch.tensor(embeddings_2d)
    
    # Project to Poincaré disk
    poincare_embeddings = euclidean_to_poincare(embeddings_2d, ball)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Draw boundary of Poincaré disk
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--')
    plt.gca().add_patch(circle)
    
    if labels is not None:
        # Color by labels
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            plt.scatter(
                poincare_embeddings[mask, 0].cpu().numpy(),
                poincare_embeddings[mask, 1].cpu().numpy(),
                label=f'Class {label}',
                alpha=0.7
            )
        plt.legend()
    else:
        # Single color
        plt.scatter(
            poincare_embeddings[:, 0].cpu().numpy(),
            poincare_embeddings[:, 1].cpu().numpy(),
            alpha=0.7
        )
    
    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.axis('equal')
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_distance_distributions(layer_results, model_name, layer_idx, save_dir=None):
    """
    Plot distributions of Euclidean and hyperbolic distances.
    
    Args:
        layer_results: Dictionary with distances for a specific layer
        model_name: Name of the BERT model for the plot title
        layer_idx: Layer index for the plot title
        save_dir: Directory to save the plot, if None, plot will be displayed
    """
    euclidean_distances = torch.cat([tensor.flatten() for tensor in layer_results['euclidean'][layer_idx]])
    hyperbolic_distances = torch.cat([tensor.flatten() for tensor in layer_results['hyperbolic'][layer_idx]])
    
    # Remove self-distances (zeros)
    euclidean_distances = euclidean_distances[euclidean_distances > 0]
    hyperbolic_distances = hyperbolic_distances[hyperbolic_distances > 0]
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(euclidean_distances.numpy(), kde=True)
    plt.title('Euclidean Distances')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    sns.histplot(hyperbolic_distances.numpy(), kde=True)
    plt.title('Hyperbolic Distances')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    
    plt.suptitle(f'Distance Distributions for {model_name} (Layer {layer_idx})', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'distance_distributions_{model_name}_layer_{layer_idx}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_nn_jaccard_similarity(jaccard_similarities, model_name, save_dir=None):
    """
    Plot Jaccard similarity between syntactic and semantic nearest neighbors across layers.
    
    Args:
        jaccard_similarities: List of Jaccard similarities for each layer
        model_name: Name of the BERT model for the plot title
        save_dir: Directory to save the plot, if None, plot will be displayed
    """
    num_layers = len(jaccard_similarities)
    layer_indices = list(range(num_layers))
    
    plt.figure(figsize=(10, 6))
    plt.plot(layer_indices, jaccard_similarities, 'o-')
    plt.title(f'Jaccard Similarity between Syntactic and Semantic Nearest Neighbors ({model_name})')
    plt.xlabel('Layer')
    plt.ylabel('Jaccard Similarity')
    plt.xticks(layer_indices)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'nn_jaccard_similarity_{model_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_dependency_correlation(correlation_results, model_name, save_dir=None):
    """
    Plot correlation between embedding distances and dependency distances across layers.
    
    Args:
        correlation_results: Dictionary with correlation scores for each layer
        model_name: Name of the BERT model for the plot title
        save_dir: Directory to save the plot, if None, plot will be displayed
    """
    num_layers = len(correlation_results['euclidean_correlation'])
    layer_indices = list(range(num_layers))
    
    plt.figure(figsize=(10, 6))
    plt.plot(layer_indices, correlation_results['euclidean_correlation'], 'o-', label='Euclidean')
    plt.plot(layer_indices, correlation_results['hyperbolic_correlation'], 'o-', label='Hyperbolic')
    plt.plot(layer_indices, correlation_results['hyperbolic_advantage'], 'o-', color='green', 
             label='Hyperbolic Advantage')
    
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.title(f'Correlation with Dependency Tree Distances ({model_name})')
    plt.xlabel('Layer')
    plt.ylabel('Spearman Correlation')
    plt.xticks(layer_indices)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'dependency_correlation_{model_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def analyze_syntax_semantics(model, tokenizer, dataset, device="cuda", batch_size=8, save_dir=None):
    """
    Analyze how syntactic and semantic information is represented across BERT layers.
    
    Args:
        model: A BERT model with output_hidden_states=True
        tokenizer: BERT tokenizer
        dataset: Dictionary with 'sentences' and optional 'dependency_trees'
        device: Device to run computations on
        batch_size: Batch size for processing
        save_dir: Directory to save plots, if None, plots will be displayed
        
    Returns:
        dict: Dictionary with analysis results
    """
    # Extract data
    sentences = dataset['sentences']
    dependency_trees = dataset.get('dependency_trees', None)
    
    # Extract distances from BERT layers
    distances = extract_syntactic_semantic_distances(
        model, tokenizer, sentences, device, batch_size
    )
    
    # Initialize results
    num_layers = len(distances['euclidean'])
    jaccard_similarities = []
    distance_stats = []
    nn_distance_stats = []
    
    # Initialize Poincaré ball for visualization
    ball = init_poincare_ball()
    
    # Process each layer
    for layer_idx in range(num_layers):
        # Sample a sentence for visualization
        sample_idx = 0
        while sample_idx < len(distances['euclidean'][layer_idx]):
            if distances['euclidean'][layer_idx][sample_idx].shape[0] >= 5:
                break
            sample_idx += 1
        
        if sample_idx < len(distances['euclidean'][layer_idx]):
            euclidean_distances = distances['euclidean'][layer_idx][sample_idx]
            hyperbolic_distances = distances['hyperbolic'][layer_idx][sample_idx]
            
            # Plot distance distributions for this layer
            if save_dir:
                plot_distance_distributions(
                    distances, model.config.model_type, layer_idx,
                    os.path.join(save_dir, 'distributions')
                )
            
            # Analyze overlaps between syntactic and semantic neighbors
            n = euclidean_distances.shape[0]
            
            # For simplicity, use Euclidean distances for semantic neighbors 
            # and hyperbolic distances for syntactic neighbors
            # This is a simplification - in practice, you would use specialized probes
            dummy_embeddings = torch.eye(n)  # Placeholder
            jaccard = compute_nn_overlap(
                dummy_embeddings, hyperbolic_distances, euclidean_distances
            )
            jaccard_similarities.append(jaccard)
            
            # Analyze distance distributions
            stats = analyze_distance_distributions(
                hyperbolic_distances, euclidean_distances
            )
            distance_stats.append(stats)
            
            # Analyze nearest neighbor distances
            nn_stats = analyze_nearest_neighbor_distances(
                dummy_embeddings, hyperbolic_distances, euclidean_distances
            )
            nn_distance_stats.append(nn_stats)
    
    # Plot Jaccard similarity across layers
    if save_dir:
        plot_nn_jaccard_similarity(
            jaccard_similarities, model.config.model_type,
            os.path.join(save_dir, 'jaccard')
        )
    
    # Analyze correlation with dependency trees if available
    dependency_results = None
    if dependency_trees:
        dependency_results = {
            'euclidean_correlation': [],
            'hyperbolic_correlation': [],
            'hyperbolic_advantage': []
        }
        
        for layer_idx in range(num_layers):
            corr = analyze_dependency_vs_semantic_correlation(
                distances, dependency_trees, layer_idx
            )
            dependency_results['euclidean_correlation'].append(corr['euclidean_correlation'])
            dependency_results['hyperbolic_correlation'].append(corr['hyperbolic_correlation'])
            dependency_results['hyperbolic_advantage'].append(corr['hyperbolic_advantage'])
        
        # Plot dependency correlation
        if save_dir:
            plot_dependency_correlation(
                dependency_results, model.config.model_type,
                os.path.join(save_dir, 'dependency')
            )
    
    return {
        'distances': distances,
        'jaccard_similarities': jaccard_similarities,
        'distance_stats': distance_stats,
        'nn_distance_stats': nn_distance_stats,
        'dependency_results': dependency_results
    }