"""
Experiment script for analyzing hyperbolicity across BERT layers.

This script measures how hyperbolic BERT's representations are at different layers
using multiple hyperbolicity metrics, and analyzes correlation with dependency trees.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

from analysis.hyperbolicity import measure_hyperbolicity
from data.processors import UDTreebankProcessor, create_dataloader, HyperbolicDataset

# Set up logging
logger = logging.getLogger(__name__)

def run_hyperbolicity_analysis(model, tokenizer, config, args):
    """
    Run hyperbolicity analysis experiment.
    
    Args:
        model: BERT model
        tokenizer: BERT tokenizer
        config: Configuration dictionary
        args: Command line arguments
    """
    logger.info("Starting hyperbolicity analysis across BERT layers...")
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, "hyperbolicity_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load UD treebank data for syntactic analysis
    logger.info(f"Loading UD treebank data from {args.data_dir}...")
    ud_processor = UDTreebankProcessor(
        base_path=os.path.join(args.data_dir, "ud_treebanks"),
        language=config.get("language", "en"),
        split=config.get("split", "train")
    )
    
    # Get dataset with sentences and dependency trees
    dataset = ud_processor.get_dataset(max_sentences=config.get("max_sentences", 100))
    
    # Create PyTorch dataset and dataloader
    hyperbolic_dataset = HyperbolicDataset(
        sentences=dataset['sentences'],
        tokenizer=tokenizer,
        max_length=config.get("max_length", 128)
    )
    
    dataloader = create_dataloader(
        dataset=hyperbolic_dataset,
        batch_size=config.get("batch_size", 8),
        shuffle=False
    )
    
    # Measure hyperbolicity across layers
    logger.info("Measuring hyperbolicity across BERT layers...")
    hyperbolicity_results = measure_hyperbolicity(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        device=args.device,
        batch_size=config.get("batch_size", 8),
        save_dir=output_dir
    )
    
    # Save hyperbolicity results
    logger.info(f"Saving hyperbolicity results to {output_dir}...")
    save_hyperbolicity_results(hyperbolicity_results, output_dir, model.config.model_type)
    
    # Plot overall hyperbolicity trends
    plot_hyperbolicity_trends(hyperbolicity_results, model.config.model_type, output_dir)
    
    # Report findings
    report_hyperbolicity_findings(hyperbolicity_results, model.config.model_type, output_dir)
    
    logger.info("Hyperbolicity analysis completed.")

def save_hyperbolicity_results(results, output_dir, model_name):
    """
    Save hyperbolicity results to disk.
    
    Args:
        results: Dictionary with hyperbolicity results
        output_dir: Output directory
        model_name: Name of the BERT model
    """
    # Convert to numpy arrays for saving
    np_results = {}
    for key, value in results['hyperbolicity'].items():
        np_results[key] = np.array(value)
    
    if results['correlation'] is not None:
        for key, value in results['correlation'].items():
            np_results[key] = np.array(value)
    
    # Save as numpy file
    np.savez_compressed(
        os.path.join(output_dir, f"hyperbolicity_results_{model_name}.npz"),
        **np_results
    )

def plot_hyperbolicity_trends(results, model_name, output_dir):
    """
    Plot overall trends in hyperbolicity across layers.
    
    Args:
        results: Dictionary with hyperbolicity results
        model_name: Name of the BERT model
        output_dir: Output directory
    """
    hyperbolicity_results = results['hyperbolicity']
    correlation_results = results['correlation']
    
    num_layers = len(next(iter(hyperbolicity_results.values())))
    layer_indices = list(range(num_layers))
    
    # Create trend visualization
    plt.figure(figsize=(15, 10))
    
    # Plot normalized hyperbolicity measures
    plt.subplot(2, 1, 1)
    for method, values in hyperbolicity_results.items():
        # Normalize to [0, 1] range for better comparison
        min_val = min(values)
        max_val = max(values)
        if max_val > min_val:
            normalized = [(v - min_val) / (max_val - min_val) for v in values]
        else:
            normalized = [0.5] * len(values)
        
        plt.plot(layer_indices, normalized, 'o-', label=method)
    
    plt.title(f'Normalized Hyperbolicity Measures Across Layers ({model_name})')
    plt.xlabel('Layer')
    plt.ylabel('Normalized Value')
    plt.xticks(layer_indices)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot correlation results if available
    if correlation_results is not None:
        plt.subplot(2, 1, 2)
        plt.plot(layer_indices, correlation_results['euclidean'], 'o-', label='Euclidean')
        plt.plot(layer_indices, correlation_results['hyperbolic'], 'o-', label='Hyperbolic')
        
        # Plot hyperbolic advantage
        advantage = [h - e for h, e in zip(correlation_results['hyperbolic'], correlation_results['euclidean'])]
        plt.plot(layer_indices, advantage, 'o-', color='green', label='Hyperbolic Advantage')
        
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.title(f'Correlation with Dependency Tree Distances ({model_name})')
        plt.xlabel('Layer')
        plt.ylabel('Spearman Correlation')
        plt.xticks(layer_indices)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"hyperbolicity_trends_{model_name}.png"), dpi=300)
    plt.close()

def report_hyperbolicity_findings(results, model_name, output_dir):
    """
    Generate a text report summarizing hyperbolicity findings.
    
    Args:
        results: Dictionary with hyperbolicity results
        model_name: Name of the BERT model
        output_dir: Output directory
    """
    hyperbolicity_results = results['hyperbolicity']
    correlation_results = results['correlation']
    
    num_layers = len(next(iter(hyperbolicity_results.values())))
    
    with open(os.path.join(output_dir, f"hyperbolicity_findings_{model_name}.txt"), 'w') as f:
        f.write(f"Hyperbolicity Analysis Findings for {model_name}\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall trends in hyperbolicity
        f.write("Overall Trends in Hyperbolicity:\n")
        f.write("-" * 30 + "\n")
        
        for method, values in hyperbolicity_results.items():
            f.write(f"{method}:\n")
            f.write(f"  Min: {min(values):.4f} (Layer {np.argmin(values)})\n")
            f.write(f"  Max: {max(values):.4f} (Layer {np.argmax(values)})\n")
            
            # Identify trend direction
            first_half = values[:num_layers//2]
            second_half = values[num_layers//2:]
            if np.mean(first_half) < np.mean(second_half):
                trend = "increasing"
            elif np.mean(first_half) > np.mean(second_half):
                trend = "decreasing"
            else:
                trend = "stable"
            
            f.write(f"  Trend: {trend} across layers\n\n")
        
        # Correlation with dependency trees
        if correlation_results is not None:
            f.write("Correlation with Dependency Trees:\n")
            f.write("-" * 30 + "\n")
            
            # Find layers with highest correlation
            max_euclidean_idx = np.argmax(correlation_results['euclidean'])
            max_hyperbolic_idx = np.argmax(correlation_results['hyperbolic'])
            
            f.write(f"Euclidean correlation peaks at Layer {max_euclidean_idx} " 
                   f"({correlation_results['euclidean'][max_euclidean_idx]:.4f})\n")
            f.write(f"Hyperbolic correlation peaks at Layer {max_hyperbolic_idx} "
                   f"({correlation_results['hyperbolic'][max_hyperbolic_idx]:.4f})\n\n")
            
            # Analyze hyperbolic advantage
            advantage = [h - e for h, e in zip(correlation_results['hyperbolic'], correlation_results['euclidean'])]
            max_advantage_idx = np.argmax(advantage)
            
            f.write(f"Hyperbolic geometry provides the greatest advantage at Layer {max_advantage_idx} "
                   f"(+{advantage[max_advantage_idx]:.4f})\n\n")
            
            # Overall assessment
            if np.mean(advantage) > 0:
                f.write("Overall, hyperbolic geometry better captures syntactic structure than Euclidean geometry.\n")
            else:
                f.write("Overall, Euclidean geometry captures syntactic structure as well as or better than hyperbolic geometry.\n")
        
        # Key findings and implications
        f.write("\nKey Findings and Implications:\n")
        f.write("-" * 30 + "\n")
        
        # Analyze delta hyperbolicity trend if available
        if 'delta' in hyperbolicity_results:
            delta_values = hyperbolicity_results['delta']
            if np.argmin(delta_values) < num_layers // 2:
                f.write("- Lower layers show more tree-like structure (lower δ-hyperbolicity).\n")
            else:
                f.write("- Higher layers show more tree-like structure (lower δ-hyperbolicity).\n")
        
        # Analyze estimated curvature trend if available
        if 'curvature' in hyperbolicity_results:
            curvature_values = hyperbolicity_results['curvature']
            if np.min(curvature_values) < -1.0:
                f.write("- Some layers exhibit high negative curvature, suggesting strong hyperbolic nature.\n")
            
            # Look for pattern of increasing negative curvature
            if curvature_values[0] > curvature_values[-1]:
                f.write("- Curvature becomes more negative in higher layers, indicating increasing hyperbolicity.\n")
        
        # Analyze tree-likeness trend if available
        if 'tree_likeness' in hyperbolicity_results:
            tree_likeness_values = hyperbolicity_results['tree_likeness']
            if np.argmax(tree_likeness_values) > num_layers // 2:
                f.write("- Tree-likeness increases in higher layers, suggesting syntax is better preserved there.\n")
            else:
                f.write("- Tree-likeness is stronger in lower layers, suggesting early capture of syntactic structure.\n")
        
        # Recommendations for hyperbolic fine-tuning
        f.write("\nRecommendations for Hyperbolic Fine-tuning:\n")
        f.write("-" * 30 + "\n")
        
        if correlation_results is not None and np.mean(advantage) > 0:
            best_layer = max_hyperbolic_idx
            f.write(f"- Consider using hyperbolic representations from Layer {best_layer} for parsing tasks.\n")
            f.write("- Hyperbolic fine-tuning may improve performance on syntactic tasks more than semantic tasks.\n")
        else:
            f.write("- The benefits of hyperbolic geometry may be task-dependent; evaluate empirically.\n")
        
        f.write("- Consider using different geometries for different layers or tasks.\n")
        
        f.write("\nNote: These findings are based on analysis of a subset of data and should be validated with broader experiments.\n")