"""
Experiment script for analyzing syntactic vs semantic representations.

This script investigates how syntactic and semantic information is represented
across BERT layers, comparing hyperbolic and Euclidean geometries.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from analysis.syntax_semantics import analyze_syntax_semantics
from data.processors import UDTreebankProcessor, GlueProcessor, create_dataloader, SyntaxSemanticsDataset
from utils.hyperbolic_utils import init_poincare_ball, euclidean_to_poincare, poincare_distance
from visualization import visualize_poincare_disk, create_syntax_semantics_visualization, plot_correlation_heatmap

# Set up logging
logger = logging.getLogger(__name__)

def run_syntax_semantics_analysis(model, tokenizer, config, args):
    """
    Run analysis of syntactic and semantic information across BERT layers.
    
    Args:
        model: BERT model
        tokenizer: BERT tokenizer
        config: Configuration dictionary
        args: Command line arguments
    """
    logger.info("Starting syntax vs semantics analysis...")
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, "syntax_semantics_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load UD treebank data for syntactic analysis
    logger.info(f"Loading UD treebank data from {args.data_dir}...")
    ud_processor = UDTreebankProcessor(
        base_path=os.path.join(args.data_dir, "ud_treebanks"),
        language=config.get("language", "en"),
        split=config.get("split", "train")
    )
    
    # Load GLUE data for semantic analysis
    logger.info(f"Loading GLUE data from {args.data_dir}...")
    glue_processor = GlueProcessor(
        base_path=os.path.join(args.data_dir, "glue"),
        task_name=config.get("glue_task", "sst2"),
        split=config.get("split", "train")
    )
    
    # Get datasets with sentences and annotations
    ud_data = ud_processor.get_dataset(max_sentences=config.get("max_sentences", 100))
    glue_data = glue_processor.get_dataset(max_examples=config.get("max_sentences", 100))
    
    # Create a combined dataset for analysis
    combined_dataset = {
        'sentences': ud_data['sentences'][:50] + glue_data['sentences'][:50],
        'dependency_trees': ud_data['dependency_trees'][:50] + [None] * len(glue_data['sentences'][:50])
    }
    
    # Analyze syntax vs semantics
    logger.info("Analyzing syntactic and semantic representations...")
    analysis_results = analyze_syntax_semantics(
        model=model,
        tokenizer=tokenizer,
        dataset=combined_dataset,
        device=args.device,
        batch_size=config.get("batch_size", 8),
        save_dir=output_dir
    )
    
    # Save analysis results
    logger.info(f"Saving analysis results to {output_dir}...")
    save_analysis_results(analysis_results, output_dir, model.config.model_type)
    
    # Create visualizations
    logger.info("Creating visualizations...")
    create_visualizations(model, tokenizer, combined_dataset, analysis_results, model.config.model_type, output_dir, args.device)
    
    # Generate report
    logger.info("Generating analysis report...")
    generate_analysis_report(analysis_results, model.config.model_type, output_dir)
    
    logger.info("Syntax vs semantics analysis completed.")

def save_analysis_results(results, output_dir, model_name):
    """
    Save analysis results to disk.
    
    Args:
        results: Dictionary with analysis results
        output_dir: Output directory
        model_name: Name of the BERT model
    """
    # Convert to numpy arrays for saving
    np_results = {}
    
    # Save Jaccard similarities
    np_results['jaccard_similarities'] = np.array(results['jaccard_similarities'])
    
    # Save distance statistics
    for i, stats in enumerate(results['distance_stats']):
        for metric, metric_stats in stats.items():
            for stat_name, value in metric_stats.items():
                np_results[f'dist_stats_layer{i}_{metric}_{stat_name}'] = np.array(value)
    
    # Save nearest neighbor distances
    for i, stats in enumerate(results['nn_distance_stats']):
        for key, value in stats.items():
            np_results[f'nn_stats_layer{i}_{key}'] = np.array(value)
    
    # Save dependency results if available
    if results['dependency_results'] is not None:
        for key, value in results['dependency_results'].items():
            np_results[key] = np.array(value)
    
    # Save as numpy file
    np.savez_compressed(
        os.path.join(output_dir, f"syntax_semantics_results_{model_name}.npz"),
        **np_results
    )

def create_visualizations(model, tokenizer, dataset, results, model_name, output_dir, device):
    """
    Create visualizations of syntactic and semantic spaces.
    
    Args:
        model: BERT model
        tokenizer: BERT tokenizer
        dataset: Dataset containing sentences and annotations
        results: Dictionary with analysis results
        model_name: Name of the BERT model
        output_dir: Output directory
        device: Computation device
    """
    # Create directory for visualizations
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Select a few example sentences
    example_sentences = [
        dataset['sentences'][i] for i in range(min(5, len(dataset['sentences'])))
    ]
    
    # Initialize Poincaré ball
    ball = init_poincare_ball()
    
    # Find layers with best hyperbolicity for syntax and semantics
    # For demonstration, let's use lower layers for syntax and higher layers for semantics
    num_layers = model.config.num_hidden_layers + 1
    syntax_layer = num_layers // 3  # Lower third
    semantics_layer = (2 * num_layers) // 3  # Upper third
    
    # Create PCA and t-SNE visualizations for selected layers
    for layer_idx in [0, syntax_layer, semantics_layer, num_layers - 1]:
        logger.info(f"Creating visualizations for layer {layer_idx}...")
        
        # Get embeddings for example sentences
        embeddings = []
        
        with torch.no_grad():
            for sentence in example_sentences:
                inputs = tokenizer(
                    sentence,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128
                ).to(device)
                
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[layer_idx]
                
                # Extract token embeddings (exclude special tokens)
                sent_length = inputs['attention_mask'].sum().item()
                sent_embeddings = hidden_states[0, 1:sent_length-1]
                
                embeddings.append(sent_embeddings)
        
        # Concatenate all token embeddings
        all_embeddings = torch.cat(embeddings, dim=0).cpu()
        
        # Create PCA visualization
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(all_embeddings.numpy())
        
        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
        plt.title(f'PCA of BERT Layer {layer_idx} Embeddings')
        plt.xlabel(f'PC 1 (Variance: {pca.explained_variance_ratio_[0]:.2f})')
        plt.ylabel(f'PC 2 (Variance: {pca.explained_variance_ratio_[1]:.2f})')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(viz_dir, f'pca_layer_{layer_idx}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create t-SNE visualization
        tsne = TSNE(n_components=2, perplexity=min(30, max(5, all_embeddings.shape[0] // 10)))
        embeddings_tsne = tsne.fit_transform(all_embeddings.numpy())
        
        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], alpha=0.7)
        plt.title(f't-SNE of BERT Layer {layer_idx} Embeddings')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(viz_dir, f'tsne_layer_{layer_idx}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create Poincaré disk visualization
        # Convert to Poincaré ball
        poincare_embeddings = euclidean_to_poincare(all_embeddings, ball)
        
        # Apply PCA to get 2D representation while preserving as much variance as possible
        poincare_pca = PCA(n_components=2)
        poincare_2d = poincare_pca.fit_transform(poincare_embeddings.numpy())
        poincare_2d = torch.tensor(poincare_2d)
        
        # Project to Poincaré disk
        poincare_disk = euclidean_to_poincare(poincare_2d, ball)
        
        plt.figure(figsize=(10, 8))
        
        # Draw boundary of Poincaré disk
        circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--')
        plt.gca().add_patch(circle)
        
        plt.scatter(
            poincare_disk[:, 0].numpy(),
            poincare_disk[:, 1].numpy(),
            alpha=0.7
        )
        
        plt.title(f'Poincaré Disk Visualization of BERT Layer {layer_idx}')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.axis('equal')
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(viz_dir, f'poincare_layer_{layer_idx}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create comparison visualizations across layers
    create_cross_layer_visualizations(results, model_name, viz_dir)

def create_cross_layer_visualizations(results, model_name, viz_dir):
    """
    Create visualizations comparing metrics across layers.
    
    Args:
        results: Dictionary with analysis results
        model_name: Name of the BERT model
        viz_dir: Directory to save visualizations
    """
    # Plot Jaccard similarity across layers
    jaccard_similarities = results['jaccard_similarities']
    num_layers = len(jaccard_similarities)
    layer_indices = list(range(num_layers))
    
    plt.figure(figsize=(10, 6))
    plt.plot(layer_indices, jaccard_similarities, 'o-')
    plt.title(f'Jaccard Similarity between Syntactic and Semantic Nearest Neighbors ({model_name})')
    plt.xlabel('Layer')
    plt.ylabel('Jaccard Similarity')
    plt.xticks(layer_indices)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(viz_dir, f'jaccard_similarity_across_layers.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot nearest neighbor distance statistics
    avg_syntax_distances = [stats['avg_syntax_nn_distance'] for stats in results['nn_distance_stats']]
    avg_semantic_distances = [stats['avg_semantic_nn_distance'] for stats in results['nn_distance_stats']]
    distance_ratios = [stats['ratio'] for stats in results['nn_distance_stats']]
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(layer_indices, avg_syntax_distances, 'o-', label='Syntactic')
    plt.plot(layer_indices, avg_semantic_distances, 'o-', label='Semantic')
    plt.title('Average Nearest Neighbor Distances')
    plt.xlabel('Layer')
    plt.ylabel('Distance')
    plt.xticks(layer_indices)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(layer_indices, distance_ratios, 'o-')
    plt.title('Syntactic/Semantic Distance Ratio')
    plt.xlabel('Layer')
    plt.ylabel('Ratio')
    plt.xticks(layer_indices)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f'nearest_neighbor_distances_across_layers.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot dependency correlation if available
    if results['dependency_results'] is not None:
        dep_results = results['dependency_results']
        
        plt.figure(figsize=(10, 6))
        plt.plot(layer_indices, dep_results['euclidean_correlation'], 'o-', label='Euclidean')
        plt.plot(layer_indices, dep_results['hyperbolic_correlation'], 'o-', label='Hyperbolic')
        plt.plot(layer_indices, dep_results['hyperbolic_advantage'], 'o-', color='green', 
                 label='Hyperbolic Advantage')
        
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.title(f'Correlation with Dependency Tree Distances ({model_name})')
        plt.xlabel('Layer')
        plt.ylabel('Spearman Correlation')
        plt.xticks(layer_indices)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.savefig(os.path.join(viz_dir, f'dependency_correlation_across_layers.png'), dpi=300, bbox_inches='tight')
        plt.close()

def generate_analysis_report(results, model_name, output_dir):
    """
    Generate a text report summarizing syntax-semantics analysis findings.
    
    Args:
        results: Dictionary with analysis results
        model_name: Name of the BERT model
        output_dir: Output directory
    """
    jaccard_similarities = results['jaccard_similarities']
    distance_stats = results['distance_stats']
    nn_distance_stats = results['nn_distance_stats']
    dependency_results = results['dependency_results']
    
    num_layers = len(jaccard_similarities)
    
    with open(os.path.join(output_dir, f"syntax_semantics_findings_{model_name}.txt"), 'w') as f:
        f.write(f"Syntax vs Semantics Analysis Findings for {model_name}\n")
        f.write("=" * 50 + "\n\n")
        
        # Overview of syntactic and semantic representation
        f.write("Separation of Syntactic and Semantic Information:\n")
        f.write("-" * 40 + "\n")
        
        # Analyze Jaccard similarity trend
        min_jaccard = min(jaccard_similarities)
        max_jaccard = max(jaccard_similarities)
        min_layer = jaccard_similarities.index(min_jaccard)
        max_layer = jaccard_similarities.index(max_jaccard)
        
        f.write(f"Jaccard similarity between syntactic and semantic nearest neighbors:\n")
        f.write(f"  Minimum: {min_jaccard:.4f} at Layer {min_layer}\n")
        f.write(f"  Maximum: {max_jaccard:.4f} at Layer {max_layer}\n")
        
        if min_layer < max_layer:
            f.write("  Syntax and semantics become more intertwined in higher layers.\n\n")
        else:
            f.write("  Syntax and semantics become more separated in higher layers.\n\n")
        
        # Analyze distance statistics
        f.write("Distance Distribution Characteristics:\n")
        f.write("-" * 40 + "\n")
        
        for i, stats in enumerate([distance_stats[0], distance_stats[num_layers//2], distance_stats[-1]]):
            layer_name = "Early" if i == 0 else "Middle" if i == 1 else "Late"
            layer_idx = 0 if i == 0 else num_layers//2 if i == 1 else num_layers-1
            
            f.write(f"{layer_name} Layer (Layer {layer_idx}):\n")
            
            syntax_mean = stats['syntax']['mean']
            semantic_mean = stats['semantic']['mean']
            syntax_std = stats['syntax']['std']
            semantic_std = stats['semantic']['std']
            correlation = stats['correlation']['spearman']
            
            f.write(f"  Syntactic distances: mean={syntax_mean:.4f}, std={syntax_std:.4f}\n")
            f.write(f"  Semantic distances: mean={semantic_mean:.4f}, std={semantic_std:.4f}\n")
            f.write(f"  Correlation between syntactic and semantic distances: {correlation:.4f}\n")
            
            if syntax_mean < semantic_mean:
                f.write("  Syntactic distances are generally smaller than semantic distances.\n")
            else:
                f.write("  Semantic distances are generally smaller than syntactic distances.\n")
            
            if correlation > 0.7:
                f.write("  Strong correlation between syntactic and semantic spaces.\n")
            elif correlation > 0.3:
                f.write("  Moderate correlation between syntactic and semantic spaces.\n")
            else:
                f.write("  Weak correlation between syntactic and semantic spaces.\n")
            
            f.write("\n")
        
        # Analyze nearest neighbor statistics
        f.write("Nearest Neighbor Characteristics:\n")
        f.write("-" * 40 + "\n")
        
        for i, stats in enumerate([nn_distance_stats[0], nn_distance_stats[num_layers//2], nn_distance_stats[-1]]):
            layer_name = "Early" if i == 0 else "Middle" if i == 1 else "Late"
            layer_idx = 0 if i == 0 else num_layers//2 if i == 1 else num_layers-1
            
            f.write(f"{layer_name} Layer (Layer {layer_idx}):\n")
            
            syntax_nn_dist = stats['avg_syntax_nn_distance']
            semantic_nn_dist = stats['avg_semantic_nn_distance']
            ratio = stats['ratio']
            
            f.write(f"  Average distance to syntactic nearest neighbors: {syntax_nn_dist:.4f}\n")
            f.write(f"  Average distance to semantic nearest neighbors: {semantic_nn_dist:.4f}\n")
            f.write(f"  Ratio (syntax/semantic): {ratio:.4f}\n")
            
            if ratio > 1.2:
                f.write("  Syntactic neighbors are generally farther apart than semantic neighbors.\n")
            elif ratio < 0.8:
                f.write("  Semantic neighbors are generally farther apart than syntactic neighbors.\n")
            else:
                f.write("  Syntactic and semantic neighbors are at similar distances.\n")
            
            f.write("\n")
        
        # Analyze dependency correlation if available
        if dependency_results is not None:
            f.write("Correlation with Dependency Trees:\n")
            f.write("-" * 40 + "\n")
            
            euclidean_corrs = dependency_results['euclidean_correlation']
            hyperbolic_corrs = dependency_results['hyperbolic_correlation']
            advantages = dependency_results['hyperbolic_advantage']
            
            max_euclidean_idx = np.argmax(euclidean_corrs)
            max_hyperbolic_idx = np.argmax(hyperbolic_corrs)
            max_advantage_idx = np.argmax(advantages)
            
            f.write(f"Euclidean correlation peaks at Layer {max_euclidean_idx} " 
                   f"({euclidean_corrs[max_euclidean_idx]:.4f})\n")
            f.write(f"Hyperbolic correlation peaks at Layer {max_hyperbolic_idx} "
                   f"({hyperbolic_corrs[max_hyperbolic_idx]:.4f})\n")
            f.write(f"Hyperbolic advantage peaks at Layer {max_advantage_idx} "
                   f"({advantages[max_advantage_idx]:.4f})\n\n")
            
            # Analyze trends
            early_advantage = np.mean(advantages[:num_layers//3])
            middle_advantage = np.mean(advantages[num_layers//3:2*num_layers//3])
            late_advantage = np.mean(advantages[2*num_layers//3:])
            
            f.write("Trends in hyperbolic advantage:\n")
            f.write(f"  Early layers (0-{num_layers//3-1}): {early_advantage:.4f}\n")
            f.write(f"  Middle layers ({num_layers//3}-{2*num_layers//3-1}): {middle_advantage:.4f}\n")
            f.write(f"  Late layers ({2*num_layers//3}-{num_layers-1}): {late_advantage:.4f}\n\n")
            
            if early_advantage > middle_advantage and early_advantage > late_advantage:
                f.write("Hyperbolic geometry provides the greatest benefit in early layers.\n")
            elif middle_advantage > early_advantage and middle_advantage > late_advantage:
                f.write("Hyperbolic geometry provides the greatest benefit in middle layers.\n")
            else:
                f.write("Hyperbolic geometry provides the greatest benefit in late layers.\n")
        
        # Key findings and implications
        f.write("\nKey Findings and Implications:\n")
        f.write("-" * 40 + "\n")
        
        # Analyze overall syntactic/semantic separation
        avg_jaccard = np.mean(jaccard_similarities)
        if avg_jaccard < 0.3:
            f.write("- Strong separation between syntactic and semantic spaces across layers.\n")
        elif avg_jaccard < 0.5:
            f.write("- Moderate separation between syntactic and semantic spaces across layers.\n")
        else:
            f.write("- Weak separation between syntactic and semantic spaces across layers.\n")
        
        # Analyze trends in Jaccard similarity
        first_half_jaccard = np.mean(jaccard_similarities[:num_layers//2])
        second_half_jaccard = np.mean(jaccard_similarities[num_layers//2:])
        
        if first_half_jaccard < second_half_jaccard:
            f.write("- Syntax and semantics become more integrated in higher layers.\n")
        else:
            f.write("- Syntax and semantics become more separated in higher layers.\n")
        
        # Analyze hyperbolic advantage if available
        if dependency_results is not None:
            if np.mean(advantages) > 0:
                f.write("- Hyperbolic geometry better captures syntactic structure than Euclidean geometry.\n")
                
                if np.argmax(hyperbolic_corrs) < np.argmax(euclidean_corrs):
                    f.write("- Hyperbolic geometry captures syntax earlier in the network than Euclidean geometry.\n")
                
                best_layer = max_hyperbolic_idx
                f.write(f"- Layer {best_layer} shows the strongest syntactic structure in hyperbolic space.\n")
            else:
                f.write("- Euclidean geometry captures syntactic structure as well as or better than hyperbolic geometry.\n")
        
        # Recommendations
        f.write("\nRecommendations for Model Understanding and Fine-tuning:\n")
        f.write("-" * 40 + "\n")
        
        if dependency_results is not None and np.mean(advantages) > 0:
            f.write("- Consider hyperbolic fine-tuning for syntactic tasks.\n")
            f.write(f"- Focus on representations from Layer {max_hyperbolic_idx} for parsing tasks.\n")
        
        min_ratio_idx = np.argmin([stats['ratio'] for stats in nn_distance_stats])
        max_ratio_idx = np.argmax([stats['ratio'] for stats in nn_distance_stats])
        
        f.write(f"- Layer {min_ratio_idx} may be most suitable for semantic tasks.\n")
        f.write(f"- Layer {max_ratio_idx} may be most suitable for syntactic tasks.\n")
        
        f.write("- Different geometries may be optimal for different layers and tasks.\n")
        
        f.write("\nNote: These findings are based on analysis of a subset of data and should be validated with broader experiments.\n")