"""
Analysis of hyperbolicity in BERT's representations across layers.

This module provides functions to measure and analyze how hyperbolic BERT's
representations are at different layers, using multiple hyperbolicity metrics.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from tqdm import tqdm

from utils.hyperbolic_utils import (
    init_poincare_ball,
    euclidean_to_poincare,
    poincare_distance,
    compute_delta_hyperbolicity,
    compute_tree_likeness,
    estimate_curvature
)

def compute_layer_hyperbolicity(
    model,
    tokenizer,
    sentences,
    device="cuda",
    batch_size=8,
    delta_sample_size=50,
    methods=['delta', 'curvature', 'tree_likeness']
):
    """
    Compute hyperbolicity measures for each layer of BERT.
    
    Args:
        model: A BERT model with output_hidden_states=True
        tokenizer: BERT tokenizer
        sentences: List of sentences to analyze
        device: Device to run computations on
        batch_size: Batch size for processing
        delta_sample_size: Number of sentences to sample for delta-hyperbolicity
        methods: List of hyperbolicity measures to compute
        
    Returns:
        dict: Dictionary mapping method names to lists of hyperbolicity values for each layer
    """
    model.eval()
    results = {method: [] for method in methods}
    num_layers = model.config.num_hidden_layers + 1  # +1 for embeddings
    
    # Process sentences in batches
    for batch_start in tqdm(range(0, len(sentences), batch_size), desc="Processing batches"):
        batch_sentences = sentences[batch_start:batch_start + batch_size]
        
        # Tokenize sentences
        inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            # hidden_states includes embeddings + all layers
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
                
                # Concatenate all token embeddings
                all_embeddings = torch.cat(token_embeddings, dim=0)
                
                # If too many embeddings, sample a subset for efficiency
                if all_embeddings.shape[0] > 1000:
                    indices = torch.randperm(all_embeddings.shape[0])[:1000]
                    sampled_embeddings = all_embeddings[indices]
                else:
                    sampled_embeddings = all_embeddings
                
                # Compute hyperbolicity measures
                for method in methods:
                    if method == 'delta':
                        # Sample even fewer points for delta-hyperbolicity (computationally expensive)
                        if sampled_embeddings.shape[0] > delta_sample_size:
                            delta_indices = torch.randperm(sampled_embeddings.shape[0])[:delta_sample_size]
                            delta_embeddings = sampled_embeddings[delta_indices]
                        else:
                            delta_embeddings = sampled_embeddings
                        
                        # Compute pairwise Euclidean distances
                        n = delta_embeddings.shape[0]
                        distances = torch.zeros((n, n), device=device)
                        for i in range(n):
                            for j in range(i+1, n):
                                distances[i, j] = torch.norm(delta_embeddings[i] - delta_embeddings[j])
                                distances[j, i] = distances[i, j]
                        
                        # Compute delta-hyperbolicity
                        delta = compute_delta_hyperbolicity(distances)
                        
                        # Add to results if this is the first batch, otherwise average
                        if len(results[method]) <= layer_idx:
                            results[method].append(delta)
                        else:
                            results[method][layer_idx] = (results[method][layer_idx] + delta) / 2
                    
                    elif method == 'curvature':
                        # Estimate curvature
                        curvature = estimate_curvature(sampled_embeddings.cpu())
                        
                        # Add to results if this is the first batch, otherwise average
                        if len(results[method]) <= layer_idx:
                            results[method].append(curvature)
                        else:
                            results[method][layer_idx] = (results[method][layer_idx] + curvature) / 2
                    
                    elif method == 'tree_likeness':
                        # Compute tree-likeness score
                        ball = init_poincare_ball()
                        poincare_embeddings = euclidean_to_poincare(sampled_embeddings, ball)
                        tree_likeness = compute_tree_likeness(poincare_embeddings, ball)
                        
                        # Add to results if this is the first batch, otherwise average
                        if len(results[method]) <= layer_idx:
                            results[method].append(tree_likeness)
                        else:
                            results[method][layer_idx] = (results[method][layer_idx] + tree_likeness) / 2
    
    return results

def analyze_dependency_distances(
    model,
    tokenizer,
    sentences,
    dependency_trees,
    device="cuda",
    batch_size=8
):
    """
    Analyze how well hyperbolic distances correlate with dependency tree distances
    across different layers.
    
    Args:
        model: A BERT model with output_hidden_states=True
        tokenizer: BERT tokenizer
        sentences: List of sentences to analyze
        dependency_trees: List of dependency trees (one per sentence)
        device: Device to run computations on
        batch_size: Batch size for processing
        
    Returns:
        dict: Dictionary with correlation scores for each layer
    """
    model.eval()
    correlations = {
        'euclidean': [],  # Correlation with Euclidean distances
        'hyperbolic': []  # Correlation with hyperbolic distances
    }
    
    num_layers = model.config.num_hidden_layers + 1  # +1 for embeddings
    ball = init_poincare_ball()
    
    # Process sentences in batches
    for batch_start in tqdm(range(0, len(sentences), batch_size), desc="Processing dependency correlations"):
        batch_sentences = sentences[batch_start:batch_start + batch_size]
        batch_trees = dependency_trees[batch_start:batch_start + batch_size]
        
        # Tokenize sentences
        inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states
            
            # Process each layer
            for layer_idx in range(num_layers):
                layer_hidden_states = hidden_states[layer_idx]
                
                # Compute correlations for each sentence in the batch
                batch_euclidean_corrs = []
                batch_hyperbolic_corrs = []
                
                for i, (tree, sent_length) in enumerate(zip(batch_trees, inputs['attention_mask'].sum(dim=1))):
                    # Get sentence embeddings (exclude [CLS], [SEP], and padding)
                    sent_embeddings = layer_hidden_states[i, 1:sent_length-1]
                    
                    # Skip if too short
                    if sent_embeddings.shape[0] < 2:
                        continue
                    
                    # Compute dependency distances from the tree
                    dep_distances = compute_dependency_distances(tree)
                    
                    # Compute Euclidean pairwise distances
                    n = sent_embeddings.shape[0]
                    euclidean_distances = torch.zeros((n, n), device=device)
                    for j in range(n):
                        for k in range(j+1, n):
                            euclidean_distances[j, k] = torch.norm(sent_embeddings[j] - sent_embeddings[k])
                            euclidean_distances[k, j] = euclidean_distances[j, k]
                    
                    # Project to Poincaré ball and compute hyperbolic distances
                    poincare_embeddings = euclidean_to_poincare(sent_embeddings, ball)
                    hyperbolic_distances = torch.zeros((n, n), device=device)
                    for j in range(n):
                        for k in range(j+1, n):
                            hyperbolic_distances[j, k] = poincare_distance(
                                poincare_embeddings[j].unsqueeze(0),
                                poincare_embeddings[k].unsqueeze(0),
                                ball
                            )
                            hyperbolic_distances[k, j] = hyperbolic_distances[j, k]
                    
                    # Compute Spearman correlations
                    # Flatten the upper triangular part of the matrices
                    triu_indices = torch.triu_indices(n, n, 1)
                    dep_distances_flat = dep_distances[triu_indices[0], triu_indices[1]].cpu().numpy()
                    euclidean_flat = euclidean_distances[triu_indices[0], triu_indices[1]].cpu().numpy()
                    hyperbolic_flat = hyperbolic_distances[triu_indices[0], triu_indices[1]].cpu().numpy()
                    
                    # Compute Spearman correlations
                    euclidean_corr, _ = spearmanr(dep_distances_flat, euclidean_flat)
                    hyperbolic_corr, _ = spearmanr(dep_distances_flat, hyperbolic_flat)
                    
                    batch_euclidean_corrs.append(euclidean_corr)
                    batch_hyperbolic_corrs.append(hyperbolic_corr)
                
                # Average correlations for this batch and layer
                if batch_euclidean_corrs and batch_hyperbolic_corrs:
                    euclidean_avg = np.mean(batch_euclidean_corrs)
                    hyperbolic_avg = np.mean(batch_hyperbolic_corrs)
                    
                    # Add to results if this is the first batch, otherwise average
                    if len(correlations['euclidean']) <= layer_idx:
                        correlations['euclidean'].append(euclidean_avg)
                        correlations['hyperbolic'].append(hyperbolic_avg)
                    else:
                        correlations['euclidean'][layer_idx] = (correlations['euclidean'][layer_idx] + euclidean_avg) / 2
                        correlations['hyperbolic'][layer_idx] = (correlations['hyperbolic'][layer_idx] + hyperbolic_avg) / 2
    
    return correlations

def compute_dependency_distances(dependency_tree):
    """
    Compute the shortest path distances in a dependency tree.
    
    Args:
        dependency_tree: A representation of a dependency tree 
                         (list of tuples (parent_idx, child_idx) or adjacency matrix)
    
    Returns:
        torch.Tensor: Matrix of pairwise dependency distances
    """
    # Convert dependency tree to adjacency matrix if it's not already
    if isinstance(dependency_tree, list):
        n = max(max(parent, child) for parent, child in dependency_tree) + 1
        adj_matrix = torch.zeros((n, n), dtype=torch.float)
        for parent, child in dependency_tree:
            adj_matrix[parent, child] = 1
            adj_matrix[child, parent] = 1  # Undirected graph
    else:
        adj_matrix = dependency_tree
        n = adj_matrix.shape[0]
    
    # Compute all-pairs shortest paths using Floyd-Warshall algorithm
    distances = adj_matrix.clone()
    distances[distances == 0] = float('inf')
    distances.fill_diagonal_(0)
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if distances[i, k] + distances[k, j] < distances[i, j]:
                    distances[i, j] = distances[i, k] + distances[k, j]
    
    # Replace inf with max distance + 1 for numerical stability
    max_dist = distances[distances != float('inf')].max()
    distances[distances == float('inf')] = max_dist + 1
    
    return distances

def plot_hyperbolicity_measures(hyperbolicity_results, model_name, save_path=None):
    """
    Plot hyperbolicity measures across BERT layers.
    
    Args:
        hyperbolicity_results: Dictionary with hyperbolicity measures for each layer
        model_name: Name of the BERT model for the plot title
        save_path: Path to save the plot, if None, plot will be displayed
    """
    num_layers = len(hyperbolicity_results['delta']) if 'delta' in hyperbolicity_results else \
                len(next(iter(hyperbolicity_results.values())))
    layer_indices = list(range(num_layers))
    
    plt.figure(figsize=(12, 8))
    
    if 'delta' in hyperbolicity_results:
        plt.subplot(2, 2, 1)
        plt.plot(layer_indices, hyperbolicity_results['delta'], 'o-', label='δ-hyperbolicity')
        plt.title('δ-hyperbolicity across layers')
        plt.xlabel('Layer')
        plt.ylabel('δ value')
        plt.xticks(layer_indices)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
    
    if 'curvature' in hyperbolicity_results:
        plt.subplot(2, 2, 2)
        plt.plot(layer_indices, hyperbolicity_results['curvature'], 'o-', color='orange', label='Curvature')
        plt.title('Estimated curvature across layers')
        plt.xlabel('Layer')
        plt.ylabel('Curvature')
        plt.xticks(layer_indices)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
    
    if 'tree_likeness' in hyperbolicity_results:
        plt.subplot(2, 2, 3)
        plt.plot(layer_indices, hyperbolicity_results['tree_likeness'], 'o-', color='green', label='Tree-likeness')
        plt.title('Tree-likeness measure across layers')
        plt.xlabel('Layer')
        plt.ylabel('Tree-likeness score')
        plt.xticks(layer_indices)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
    
    plt.subplot(2, 2, 4)
    # Plot relative changes for easier comparison
    for method, values in hyperbolicity_results.items():
        # Normalize to [0, 1] range
        min_val = min(values)
        max_val = max(values)
        if max_val > min_val:
            normalized = [(v - min_val) / (max_val - min_val) for v in values]
        else:
            normalized = [0.5] * len(values)
        plt.plot(layer_indices, normalized, 'o-', label=method)
    
    plt.title('Normalized hyperbolicity measures')
    plt.xlabel('Layer')
    plt.ylabel('Normalized value')
    plt.xticks(layer_indices)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.suptitle(f'Hyperbolicity Analysis for {model_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_dependency_correlation(correlation_results, model_name, save_path=None):
    """
    Plot correlation between embedding distances and dependency tree distances across layers.
    
    Args:
        correlation_results: Dictionary with correlation scores for each layer
        model_name: Name of the BERT model for the plot title
        save_path: Path to save the plot, if None, plot will be displayed
    """
    num_layers = len(correlation_results['euclidean'])
    layer_indices = list(range(num_layers))
    
    plt.figure(figsize=(10, 6))
    plt.plot(layer_indices, correlation_results['euclidean'], 'o-', label='Euclidean')
    plt.plot(layer_indices, correlation_results['hyperbolic'], 'o-', label='Hyperbolic')
    
    # Plot the difference (hyperbolic advantage)
    advantage = [h - e for h, e in zip(correlation_results['hyperbolic'], correlation_results['euclidean'])]
    plt.plot(layer_indices, advantage, 'o-', color='green', label='Hyperbolic Advantage')
    
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.title(f'Correlation with Dependency Tree Distances ({model_name})')
    plt.xlabel('Layer')
    plt.ylabel('Spearman Correlation')
    plt.xticks(layer_indices)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def measure_hyperbolicity(model, tokenizer, dataset, device="cuda", batch_size=8, save_dir=None):
    """
    Measure hyperbolicity of BERT representations and analyze correlation with dependency trees.
    
    Args:
        model: A BERT model with output_hidden_states=True
        tokenizer: BERT tokenizer
        dataset: Dictionary with 'sentences' and 'dependency_trees'
        device: Device to run computations on
        batch_size: Batch size for processing
        save_dir: Directory to save plots, if None, plots will be displayed
        
    Returns:
        dict: Dictionary with hyperbolicity and correlation results
    """
    # Extract data
    sentences = dataset['sentences']
    dependency_trees = dataset.get('dependency_trees', None)
    
    # Compute hyperbolicity measures
    hyperbolicity_results = compute_layer_hyperbolicity(
        model, tokenizer, sentences, device, batch_size
    )
    
    # Plot results
    if save_dir is not None:
        import os
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"hyperbolicity_{model.config.model_type}.png")
    else:
        save_path = None
    
    plot_hyperbolicity_measures(hyperbolicity_results, model.config.model_type, save_path)
    
    # Analyze dependency correlations if trees are provided
    correlation_results = None
    if dependency_trees is not None:
        correlation_results = analyze_dependency_distances(
            model, tokenizer, sentences, dependency_trees, device, batch_size
        )
        
        # Plot correlation results
        if save_dir is not None:
            save_path = os.path.join(save_dir, f"dependency_correlation_{model.config.model_type}.png")
        else:
            save_path = None
        
        plot_dependency_correlation(correlation_results, model.config.model_type, save_path)
    
    return {
        'hyperbolicity': hyperbolicity_results,
        'correlation': correlation_results
    }