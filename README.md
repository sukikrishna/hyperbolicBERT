# Hyperbolic Geometry Analysis of BERT's Representations

This repository investigates how syntactic and semantic information is encoded across BERT layers through the lens of hyperbolic geometry. The project explores whether BERT's representations become more hyperbolic across layers, and whether hyperbolic geometry is better suited for capturing hierarchical syntactic structures than Euclidean geometry.

## Project Overview

Recent research suggests that BERT's internal syntactic representations align well with hyperbolic geometry, preserving dependency tree structures more naturally than Euclidean space. This project builds on prior work to systematically analyze how hyperbolicity evolves across layers and how syntactic and semantic representations interact in hyperbolic space.

The project has two main components:
1. **Hyperbolicity Analysis**: Measuring how hyperbolic BERT's representations are at different layers using multiple hyperbolicity metrics.
2. **Syntax-Semantics Analysis**: Investigating how syntactic and semantic information is represented across layers, and how these representations differ in hyperbolic vs. Euclidean space.

## Setup and Installation

1. Clone this repository:
```bash
git clone https://github.com/sukikrishna/hyperbolicBERT.git
cd hyperbolicBERT
```

2. Create a conda environment and install dependencies:
```bash
conda create -n hyperbolic-bert python=3.9
conda activate hyperbolic-bert
pip install -r requirements.txt
```

3. Download the necessary datasets:
```bash
# Download UD Treebank
bash scripts/download_ud_treebank.sh

# Download GLUE data
python download_glue_data.py
```

## Project Structure

```
.
├── main.py                         # Entry point for running experiments
├── config/                         # Configuration files
│   └── config.py                   # Configuration parameters
├── data/                           # Data loading and preprocessing
│   ├── processors.py               # Data processors for different datasets
│   └── datasets.py                 # PyTorch dataset classes
├── models/                         # Model implementations
│   ├── probes.py                   # Euclidean and hyperbolic probes
│   ├── losses.py                   # Loss functions for different geometries
│   └── fine_tuning.py              # Fine-tuning utilities
├── analysis/                       # Analysis modules
│   ├── hyperbolicity.py            # Hyperbolicity measures and analysis
│   ├── syntax_semantics.py         # Syntax vs. semantics analysis
│   └── visualization.py            # Visualization utilities
├── utils/                          # Utility functions
│   ├── hyperbolic_utils.py         # Utilities for hyperbolic geometry
│   └── embedding_utils.py          # Utilities for embeddings
├── experiments/                    # Experiment scripts
│   ├── run_hyperbolicity_analysis.py    # Script for hyperbolicity analysis
│   └── run_syntax_semantics_analysis.py # Script for syntax-semantics analysis
└── outputs/                        # Results and visualizations
```

## Running Experiments

### Hyperbolicity Analysis

To analyze hyperbolicity across BERT layers:

```bash
python main.py --config config/hyperbolicity.yaml --task hyperbolicity --model bert-base-uncased
```

This will:
1. Measure various hyperbolicity metrics (δ-hyperbolicity, curvature, tree-likeness) across layers
2. Analyze correlation between embedding distances and dependency tree distances
3. Generate visualizations and a summary report of findings

### Syntax-Semantics Analysis

To analyze syntactic vs. semantic representations:

```bash
python main.py --config config/syntax_semantics.yaml --task syntax_semantics --model bert-base-uncased
```

This will:
1. Extract syntactic and semantic distances from BERT layers
2. Compute overlap between syntactic and semantic nearest neighbors
3. Analyze distance distributions and correlations
4. Generate visualizations in both Euclidean and hyperbolic spaces

### Fine-tuning Experiments

To run fine-tuning experiments with hyperbolic loss:

```bash
python main.py --config config/fine_tuning.yaml --task fine_tuning --model bert-base-uncased
```

This will:
1. Fine-tune BERT with a hyperbolic loss function
2. Compare performance with standard Euclidean fine-tuning
3. Analyze how hyperbolicity evolves during fine-tuning

## Key Visualizations

The code produces several visualizations to help understand the results:

1. **Hyperbolicity Metrics**: Plots showing δ-hyperbolicity, curvature, and tree-likeness across layers.
2. **Poincaré Disk Visualizations**: Embeddings projected into the Poincaré disk, showing hierarchical structure.
3. **Dependency Correlation**: Correlation between embedding distances and dependency tree distances.
4. **Syntax-Semantics Separation**: Visualizations showing separation between syntactic and semantic spaces.
5. **Layer Progression**: How representations evolve across layers in hyperbolic space.

## Customizing Experiments

You can customize experiments by modifying the configuration files in the `config/` directory:

- `hyperbolicity.yaml`: Parameters for hyperbolicity analysis
- `syntax_semantics.yaml`: Parameters for syntax-semantics analysis
- `fine_tuning.yaml`: Parameters for fine-tuning experiments

Key parameters include:
- `max_sentences`: Number of sentences to analyze
- `batch_size`: Batch size for processing
- `language`: Language code for UD treebank
- `glue_task`: GLUE task for semantic analysis
- `hyperbolicity_metrics`: Metrics to compute

## Interpreting Results

The hyperbolicity analysis provides insights into:
- How tree-like BERT's representations are at different layers
- Which layers have the strongest hyperbolic structure
- Whether syntax is better represented in hyperbolic space than Euclidean space

The syntax-semantics analysis reveals:
- How well-separated syntactic and semantic information is across layers
- Whether syntax is more hyperbolic than semantics
- Which layers are optimal for syntactic vs. semantic tasks

<!-- ## Citation

If you use this code in your research, please cite our paper: -->

<!-- ```
@article{author2025hyperbolic,
  title={Hyperbolic Geometry Analysis of BERT's Representations: Syntax, Semantics, and Layer Progression},
  author={Author, A.},
  journal={ArXiv},
  year={2025}
}
``` -->

## References

1. Chen, J., Li, Y., He, L., Deng, Y., Zhang, Y., & Xu, G. (2021). Hyperbolic Interaction Model for Hierarchical Multi-Label Classification. In Proceedings of AAAI Conference.
2. Hewitt, J., & Manning, C. D. (2019). A Structural Probe for Finding Syntax in Word Representations. In Proceedings of NAACL-HLT.
3. Nickel, M., & Kiela, D. (2017). Poincaré Embeddings for Learning Hierarchical Representations. In Advances in Neural Information Processing Systems.
4. Tifrea, A., Bécigneul, G., & Ganea, O. E. (2018). Poincaré Glove: Hyperbolic Word Embeddings. In International Conference on Learning Representations.

## License

This project is licensed under the MIT License - see the LICENSE file for details.