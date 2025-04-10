"""
Hyperbolic Analysis of BERT's Syntax and Semantics

This project investigates how syntactic and semantic representations evolve across BERT layers
through the lens of hyperbolic geometry.

Project Structure:
- main.py: Entry point for running experiments
- data/
  - processors.py: Data loading and preprocessing utilities
  - datasets.py: Dataset classes for different tasks
- models/
  - probes.py: Euclidean and hyperbolic probing models
  - losses.py: Loss functions for Euclidean and hyperbolic space
  - fine_tuning.py: Fine-tuning utilities for BERT with hyperbolic loss
- analysis/
  - hyperbolicity.py: Measures of hyperbolicity across layers
  - syntax_semantics.py: Analysis of syntactic vs semantic relationships
  - visualization.py: Visualization utilities for hyperbolic space
- utils/
  - hyperbolic_utils.py: Utilities for hyperbolic geometry
  - embedding_utils.py: Utilities for extracting and manipulating embeddings
- config/
  - config.py: Configuration parameters
- experiments/
  - run_hyperbolicity_analysis.py: Script to analyze hyperbolicity across layers
  - run_syntax_semantics_analysis.py: Script to compare syntax and semantics
  - run_fine_tuning.py: Script for fine-tuning experiments
"""

import os
import argparse
import torch
import numpy as np
import random
import logging
from transformers import BertModel, BertTokenizer
from tqdm import tqdm

# Local imports
from config.config import get_config
from data.processors import UDTreebankProcessor, GlueProcessor
from models.probes import EuclideanProbe, HyperbolicProbe
from analysis.hyperbolicity import measure_hyperbolicity
from analysis.syntax_semantics import analyze_syntax_semantics
from utils.embedding_utils import extract_bert_embeddings
from utils.hyperbolic_utils import poincare_distance

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml", type=str,
                      help="Path to the config file")
    parser.add_argument("--task", choices=["hyperbolicity", "syntax_semantics", "fine_tuning"], 
                      default="hyperbolicity", help="Task to run")
    parser.add_argument("--model", default="bert-base-uncased", 
                      help="BERT model to use")
    parser.add_argument("--output_dir", default="outputs", 
                      help="Directory to save outputs")
    parser.add_argument("--data_dir", default="data",
                      help="Directory containing datasets")
    parser.add_argument("--seed", default=42, type=int,
                      help="Random seed")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device to use (cuda or cpu)")
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load config
    config = get_config(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load BERT model and tokenizer
    logger.info(f"Loading BERT model: {args.model}")
    tokenizer = BertTokenizer.from_pretrained(args.model)
    model = BertModel.from_pretrained(args.model, output_hidden_states=True)
    model.to(args.device)
    
    # Run requested task
    if args.task == "hyperbolicity":
        logger.info("Running hyperbolicity analysis...")
        from experiments.run_hyperbolicity_analysis import run_hyperbolicity_analysis
        run_hyperbolicity_analysis(model, tokenizer, config, args)
    elif args.task == "syntax_semantics":
        logger.info("Running syntax vs semantics analysis...")
        from experiments.run_syntax_semantics_analysis import run_syntax_semantics_analysis
        run_syntax_semantics_analysis(model, tokenizer, config, args)
    elif args.task == "fine_tuning":
        logger.info("Running fine-tuning experiments...")
        from experiments.run_fine_tuning import run_fine_tuning
        run_fine_tuning(model, tokenizer, config, args)
    
    logger.info("Done!")

if __name__ == "__main__":
    main()