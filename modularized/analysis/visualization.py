"""
Visualization utilities for hyperbolic spaces and analysis results.

This module provides functions for visualizing embeddings in hyperbolic space,
as well as various metrics and findings from the hyperbolic analysis of BERT.
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Arc
from matplotlib.path import Path
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import torch

from utils.hyperbolic_utils import init_poincare_ball, euclidean_to_poincare, poincare_distance

def visualize_poincare_disk(embeddings, ball, labels=None, title="Poincaré Disk Visualization", 
                           colors=None, annotations=None, figsize=(10, 8), save_path=None):
    """
    Visualize embeddings in the Poincaré disk.
    
    Args:
        embeddings: Embeddings in the Poincaré ball (or Euclidean space to be mapped)
        ball: Poincaré ball manifold
        labels: Optional labels for coloring points
        title: Plot title
        colors: Optional color map for labels
        annotations: Optional list of strings to annotate points
        figsize: Figure size
        save_path: Path to save the plot, if None, plot will be displayed
    
    Returns:
        matplotlib figure
    """
    # Convert to numpy for processing
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    
    # Apply PCA to reduce to 2D if needed
    if embeddings.shape[1] > 2:
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
    else:
        embeddings_2d = embeddings
    
    # Convert back to PyTorch
    embeddings_2d = torch.tensor(embeddings_2d)
    
    # Project to Poincaré disk if not already there
    try:
        # Check if points are already in the disk
        norm = torch.norm(embeddings_2d, dim=1, p=2)
        if torch.any(norm >= 1):
            # Points are not in the disk, project them
            poincare_embeddings = euclidean_to_poincare(embeddings_2d, ball)
        else:
            # Points are already in the disk
            poincare_embeddings = embeddings_2d
    except:
        # If there's any error, just project to be safe
        poincare_embeddings = euclidean_to_poincare(embeddings_2d, ball)
    
    # Create plot
    fig = plt.figure(figsize=figsize)
    
    # Draw boundary of Poincaré disk
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--')
    plt.gca().add_patch(circle)
    
    # Draw geodesic grid lines (optional)
    for r in np.linspace(0.25, 0.75, 3):
        grid_circle = plt.Circle((0, 0), r, fill=False, color='gray', linestyle=':', alpha=0.5)
        plt.gca().add_patch(grid_circle)
    
    for angle in np.linspace(0, np.pi, 6):
        endpoint_x = np.cos(angle)
        endpoint_y = np.sin(angle)
        plt.plot([0, endpoint_x], [0, endpoint_y], color='gray', linestyle=':', alpha=0.5)
        plt.plot([0, -endpoint_x], [0, -endpoint_y], color='gray', linestyle=':', alpha=0.5)
    
    if labels is not None:
        # Color by labels
        unique_labels = np.unique(labels)
        
        if colors is None:
            # Generate colors
            if len(unique_labels) <= 10:
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            else:
                colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(
                poincare_embeddings[mask, 0].cpu().numpy(),
                poincare_embeddings[mask, 1].cpu().numpy(),
                label=f'Class {label}',
                color=colors[i],
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
    
    # Add annotations if provided
    if annotations is not None:
        for i, text in enumerate(annotations):
            if i < len(poincare_embeddings):
                plt.annotate(
                    text,
                    (poincare_embeddings[i, 0].item(), poincare_embeddings[i, 1].item()),
                    fontsize=9,
                    alpha=0.7
                )
    
    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.axis('equal')
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.grid(False)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return None
    else:
        return fig

def plot_hyperbolicity_metrics(metrics, model_name, figsize=(12, 8), save_path=None):
    """
    Plot hyperbolicity metrics across layers.
    
    Args:
        metrics: Dictionary with hyperbolicity metrics for each layer
        model_name: Name of the model
        figsize: Figure size
        save_path: Path to save the plot, if None, plot will be displayed
    
    Returns:
        matplotlib figure
    """
    # Extract metrics
    num_layers = len(next(iter(metrics.values())))
    layer_indices = list(range(num_layers))
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Plot metrics
    metric_names = list(metrics.keys())
    metric_colors = plt.cm.tab10(np.linspace(0, 1, len(metric_names)))
    
    # Plot individual metrics
    for i, (metric_name, values) in enumerate(metrics.items()):
        if i < 3:  # First 3 subplots for individual metrics
            ax = axes[i]
            ax.plot(layer_indices, values, 'o-', color=metric_colors[i], label=metric_name)
            ax.set_title(f'{metric_name.replace("_", " ").title()} across Layers')
            ax.set_xlabel('Layer')
            ax.set_ylabel(metric_name.replace('_', ' ').title())
            ax.set_xticks(layer_indices)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
    
    # Normalized comparison plot
    ax = axes[3]
    for i, (metric_name, values) in enumerate(metrics.items()):
        # Normalize to [0, 1] range for easier comparison
        min_val = min(values)
        max_val = max(values)
        if max_val > min_val:
            normalized = [(v - min_val) / (max_val - min_val) for v in values]
        else:
            normalized = [0.5] * len(values)
        ax.plot(layer_indices, normalized, 'o-', color=metric_colors[i], label=metric_name)
    
    ax.set_title('Normalized Metrics Comparison')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Normalized Value')
    ax.set_xticks(layer_indices)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.suptitle(f'Hyperbolicity Metrics for {model_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return None
    else:
        return fig

def create_syntax_semantics_visualization(syntax_distances, semantic_distances, tokens, 
                                         title="Syntax vs Semantics", figsize=(18, 6), save_path=None):
    """
    Create a visualization comparing syntactic and semantic spaces.
    
    Args:
        syntax_distances: Matrix of syntactic distances
        semantic_distances: Matrix of semantic distances
        tokens: List of tokens
        title: Plot title
        figsize: Figure size
        save_path: Path to save the plot, if None, plot will be displayed
    
    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot syntax heatmap
    im1 = axes[0].imshow(syntax_distances, cmap='viridis')
    axes[0].set_title('Syntactic Distances')
    axes[0].set_xticks(range(len(tokens)))
    axes[0].set_yticks(range(len(tokens)))
    axes[0].set_xticklabels(tokens, rotation=45, ha='right')
    axes[0].set_yticklabels(tokens)
    plt.colorbar(im1, ax=axes[0])
    
    # Plot semantics heatmap
    im2 = axes[1].imshow(semantic_distances, cmap='viridis')
    axes[1].set_title('Semantic Distances')
    axes[1].set_xticks(range(len(tokens)))
    axes[1].set_yticks(range(len(tokens)))
    axes[1].set_xticklabels(tokens, rotation=45, ha='right')
    axes[1].set_yticklabels(tokens)
    plt.colorbar(im2, ax=axes[1])
    
    # Plot difference heatmap
    difference = syntax_distances - semantic_distances
    im3 = axes[2].imshow(difference, cmap='coolwarm', vmin=-np.max(np.abs(difference)), vmax=np.max(np.abs(difference)))
    axes[2].set_title('Difference (Syntax - Semantics)')
    axes[2].set_xticks(range(len(tokens)))
    axes[2].set_yticks(range(len(tokens)))
    axes[2].set_xticklabels(tokens, rotation=45, ha='right')
    axes[2].set_yticklabels(tokens)
    plt.colorbar(im3, ax=axes[2])
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return None
    else:
        return fig

def plot_correlation_heatmap(correlation_matrix, labels, title="Feature Correlation", 
                           figsize=(10, 8), save_path=None):
    """
    Plot a correlation heatmap.
    
    Args:
        correlation_matrix: Matrix of correlation coefficients
        labels: Labels for the features
        title: Plot title
        figsize: Figure size
        save_path: Path to save the plot, if None, plot will be displayed
    
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient')
    
    # Add ticks and labels
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    
    # Add title
    ax.set_title(title)
    
    # Add correlation values to cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                          ha='center', va='center', color='black' if abs(correlation_matrix[i, j]) < 0.7 else 'white')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return None
    else:
        return fig

def create_hyperbolicity_evolution_graph(metrics_by_finetuning_step, metric_name, model_name, 
                                       figsize=(10, 6), save_path=None):
    """
    Create a graph showing how hyperbolicity evolves during fine-tuning.
    
    Args:
        metrics_by_finetuning_step: List of hyperbolicity metrics for each fine-tuning step
        metric_name: Name of the metric to plot
        model_name: Name of the model
        figsize: Figure size
        save_path: Path to save the plot, if None, plot will be displayed
    
    Returns:
        matplotlib figure
    """
    num_steps = len(metrics_by_finetuning_step)
    steps = list(range(num_steps))
    
    num_layers = len(metrics_by_finetuning_step[0][metric_name])
    
    fig = plt.figure(figsize=figsize)
    
    # Create a colormap for different layers
    colors = plt.cm.viridis(np.linspace(0, 1, num_layers))
    
    # Plot evolution for each layer
    for layer_idx in range(num_layers):
        layer_values = [metrics[metric_name][layer_idx] for metrics in metrics_by_finetuning_step]
        plt.plot(steps, layer_values, 'o-', color=colors[layer_idx], label=f'Layer {layer_idx}')
    
    plt.xlabel('Fine-tuning Step')
    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.title(f'Evolution of {metric_name.replace("_", " ").title()} During Fine-tuning ({model_name})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return None
    else:
        return fig

def plot_task_performance_vs_hyperbolicity(task_performance, hyperbolicity_metrics, metric_name,
                                         model_name, figsize=(10, 6), save_path=None):
    """
    Plot relationship between task performance and hyperbolicity.
    
    Args:
        task_performance: List of performance metrics for different models/layers
        hyperbolicity_metrics: Corresponding hyperbolicity metrics
        metric_name: Name of the hyperbolicity metric
        model_name: Name of the model
        figsize: Figure size
        save_path: Path to save the plot, if None, plot will be displayed
    
    Returns:
        matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    
    # Scatter plot
    plt.scatter(hyperbolicity_metrics, task_performance, alpha=0.7)
    
    # Add trend line
    z = np.polyfit(hyperbolicity_metrics, task_performance, 1)
    p = np.poly1d(z)
    plt.plot(sorted(hyperbolicity_metrics), p(sorted(hyperbolicity_metrics)), 'r--', alpha=0.7)
    
    plt.xlabel(metric_name.replace('_', ' ').title())
    plt.ylabel('Task Performance')
    plt.title(f'Task Performance vs {metric_name.replace("_", " ").title()} ({model_name})')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Calculate and display correlation
    correlation = np.corrcoef(hyperbolicity_metrics, task_performance)[0, 1]
    plt.annotate(f'Correlation: {correlation:.3f}', xy=(0.05, 0.95), xycoords='axes fraction', 
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return None
    else:
        return fig

def visualize_all_hyperbolicity_results(results, model_name, output_dir):
    """
    Create a comprehensive set of visualizations for hyperbolicity analysis results.
    
    Args:
        results: Dictionary with hyperbolicity analysis results
        model_name: Name of the model
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories
    metrics_dir = os.path.join(output_dir, 'metrics')
    disk_dir = os.path.join(output_dir, 'poincare_disk')
    corr_dir = os.path.join(output_dir, 'correlations')
    
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(disk_dir, exist_ok=True)
    os.makedirs(corr_dir, exist_ok=True)
    
    # Plot hyperbolicity metrics
    plot_hyperbolicity_metrics(
        results['hyperbolicity'],
        model_name,
        save_path=os.path.join(metrics_dir, f'hyperbolicity_metrics_{model_name}.png')
    )
    
    # Plot correlation with dependency trees if available
    if results.get('correlation'):
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(results['correlation']['euclidean'])), results['correlation']['euclidean'], 'o-', label='Euclidean')
        plt.plot(range(len(results['correlation']['hyperbolic'])), results['correlation']['hyperbolic'], 'o-', label='Hyperbolic')
        plt.plot(range(len(results['correlation']['hyperbolic'])), 
                [h - e for h, e in zip(results['correlation']['hyperbolic'], results['correlation']['euclidean'])], 
                'o-', color='green', label='Hyperbolic Advantage')
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.title(f'Correlation with Dependency Tree Distances ({model_name})')
        plt.xlabel('Layer')
        plt.ylabel('Spearman Correlation')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.savefig(os.path.join(corr_dir, f'dependency_correlation_{model_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create a summary visualization
    plt.figure(figsize=(12, 10))
    
    # Plot normalized hyperbolicity metrics
    plt.subplot(2, 1, 1)
    for metric_name, values in results['hyperbolicity'].items():
        # Normalize to [0, 1] range
        min_val = min(values)
        max_val = max(values)
        if max_val > min_val:
            normalized = [(v - min_val) / (max_val - min_val) for v in values]
        else:
            normalized = [0.5] * len(values)
        plt.plot(range(len(values)), normalized, 'o-', label=metric_name)
    
    plt.title('Normalized Hyperbolicity Measures')
    plt.xlabel('Layer')
    plt.ylabel('Normalized Value')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot hyperbolic advantage if available
    if results.get('correlation'):
        plt.subplot(2, 1, 2)
        advantage = [h - e for h, e in zip(results['correlation']['hyperbolic'], results['correlation']['euclidean'])]
        plt.bar(range(len(advantage)), advantage, color=['g' if x > 0 else 'r' for x in advantage])
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        plt.title('Hyperbolic Advantage for Syntactic Structure')
        plt.xlabel('Layer')
        plt.ylabel('Correlation Difference (Hyperbolic - Euclidean)')
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.suptitle(f'Hyperbolicity Analysis Summary for {model_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, f'hyperbolicity_summary_{model_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def visualize_dependency_tree_poincare(sentence, tree, embeddings, ball, title=None, figsize=(12, 10), save_path=None):
    """
    Visualize a dependency tree in the Poincaré disk, with edges shown as geodesics.
    
    Args:
        sentence: List of tokens in the sentence
        tree: Dependency tree as a list of (parent_idx, child_idx) pairs
        embeddings: Embeddings in the Poincaré ball
        ball: Poincaré ball manifold
        title: Plot title
        figsize: Figure size
        save_path: Path to save the plot, if None, plot will be displayed
    
    Returns:
        matplotlib figure
    """
    # Convert to numpy for processing
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    
    # Apply PCA to reduce to 2D if needed
    if embeddings.shape[1] > 2:
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
    else:
        embeddings_2d = embeddings
    
    # Convert back to PyTorch
    embeddings_2d = torch.tensor(embeddings_2d)
    
    # Project to Poincaré disk if not already there
    try:
        # Check if points are already in the disk
        norm = torch.norm(embeddings_2d, dim=1, p=2)
        if torch.any(norm >= 1):
            # Points are not in the disk, project them
            poincare_embeddings = euclidean_to_poincare(embeddings_2d, ball)
        else:
            # Points are already in the disk
            poincare_embeddings = embeddings_2d
    except:
        # If there's any error, just project to be safe
        poincare_embeddings = euclidean_to_poincare(embeddings_2d, ball)
    
    # Create plot
    fig = plt.figure(figsize=figsize)
    
    # Draw boundary of Poincaré disk
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--')
    plt.gca().add_patch(circle)
    
    # Draw nodes
    plt.scatter(
        poincare_embeddings[:, 0].cpu().numpy(),
        poincare_embeddings[:, 1].cpu().numpy(),
        s=100,
        alpha=0.8
    )
    
    # Add token labels
    for i, token in enumerate(sentence):
        plt.annotate(
            token,
            (poincare_embeddings[i, 0].item(), poincare_embeddings[i, 1].item()),
            fontsize=12,
            weight='bold',
            alpha=0.9,
            ha='center',
            va='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7)
        )
    
    # Draw edges as geodesics
    for parent, child in tree:
        x1, y1 = poincare_embeddings[parent, 0].item(), poincare_embeddings[parent, 1].item()
        x2, y2 = poincare_embeddings[child, 0].item(), poincare_embeddings[child, 1].item()
        
        # Calculate parameters for the geodesic (circular arc)
        # For points in Poincaré disk, the geodesic is either a diameter or a circular arc
        # perpendicular to the boundary of the disk
        
        # If points are almost collinear with origin, draw a straight line
        if abs(x1 * y2 - x2 * y1) < 1e-5:
            plt.plot([x1, x2], [y1, y2], 'k-', alpha=0.6)
        else:
            # Calculate the center and radius of the circle containing the geodesic
            # Formula from "Poincaré's Disk Model: A guide to the mysteries of the hyperbolic plane"
            
            # Calculate determinant
            den = 2 * (x1 * y2 - x2 * y1)
            
            # Calculate center of the circle
            cx = ((x1**2 + y1**2) * y2 - (x2**2 + y2**2) * y1) / den
            cy = ((x2**2 + y2**2) * x1 - (x1**2 + y1**2) * x2) / den
            
            # Calculate radius
            r = math.sqrt((x1 - cx)**2 + (y1 - cy)**2)
            
            # Calculate angles for arc
            theta1 = math.atan2(y1 - cy, x1 - cx) * 180 / math.pi
            theta2 = math.atan2(y2 - cy, x2 - cx) * 180 / math.pi
            
            # Ensure theta2 > theta1
            if theta2 < theta1:
                theta2 += 360
            
            # Draw arc
            arc = Arc((cx, cy), 2*r, 2*r, angle=0, theta1=theta1, theta2=theta2, color='k', alpha=0.6)
            plt.gca().add_patch(arc)
    
    if title:
        plt.title(title)
    else:
        plt.title("Dependency Tree in Poincaré Disk")
    
    plt.axis('equal')
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.grid(False)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return None
    else:
        return fig

def visualize_layer_progression(embeddings_by_layer, ball, tokens=None, layers_to_show=None, 
                              figsize=(15, 10), save_path=None):
    """
    Visualize how token embeddings evolve across layers in the Poincaré disk.
    
    Args:
        embeddings_by_layer: List of embeddings for each layer
        ball: Poincaré ball manifold
        tokens: List of tokens to annotate points
        layers_to_show: List of layer indices to visualize
        figsize: Figure size
        save_path: Path to save the plot, if None, plot will be displayed
    
    Returns:
        matplotlib figure
    """
    num_layers = len(embeddings_by_layer)
    
    if layers_to_show is None:
        # Show a few representative layers
        if num_layers <= 6:
            layers_to_show = list(range(num_layers))
        else:
            # Show first, last, and some intermediate layers
            layers_to_show = [0, num_layers//4, num_layers//2, 3*num_layers//4, num_layers-1]
    
    # Number of layers to visualize
    n_plots = len(layers_to_show)
    
    # Calculate grid dimensions
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Create visualizations for selected layers
    for i, layer_idx in enumerate(layers_to_show):
        if i < len(axes):
            ax = axes[i]
            plt.sca(ax)
            
            embeddings = embeddings_by_layer[layer_idx]
            
            # Convert to numpy for processing
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()
            
            # Apply PCA to reduce to 2D if needed
            if embeddings.shape[1] > 2:
                pca = PCA(n_components=2)
                embeddings_2d = pca.fit_transform(embeddings)
            else:
                embeddings_2d = embeddings
            
            # Convert back to PyTorch
            embeddings_2d = torch.tensor(embeddings_2d)
            
            # Project to Poincaré disk
            poincare_embeddings = euclidean_to_poincare(embeddings_2d, ball)
            
            # Draw boundary of Poincaré disk
            circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--')
            ax.add_patch(circle)
            
            # Draw points
            ax.scatter(
                poincare_embeddings[:, 0].cpu().numpy(),
                poincare_embeddings[:, 1].cpu().numpy(),
                alpha=0.7
            )
            
            # Add token annotations if provided
            if tokens is not None:
                for j, token in enumerate(tokens):
                    if j < len(poincare_embeddings):
                        ax.annotate(
                            token,
                            (poincare_embeddings[j, 0].item(), poincare_embeddings[j, 1].item()),
                            fontsize=9,
                            alpha=0.7
                        )
            
            ax.set_title(f"Layer {layer_idx}")
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_aspect('equal')
            ax.axis('off')
    
    # Hide unused subplots
    for i in range(len(layers_to_show), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle("Evolution of Embeddings Across BERT Layers", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return None
    else:
        return fig