# Hyperbolic Geometry Analysis of BERT's Representations

This repository investigates how syntactic and semantic information is encoded across BERT layers through the lens of hyperbolic geometry. We explore whether BERT's representations exhibit increasing hyperbolicity across layers and whether hyperbolic space is better suited for capturing hierarchical syntactic structures than Euclidean space.

## Project Overview

Recent research suggests that BERT's internal syntactic representations align well with hyperbolic geometry, preserving dependency tree structures more naturally than Euclidean space. This project builds on prior work to systematically analyze how hyperbolicity evolves across layers and how syntactic and semantic representations interact in hyperbolic space.

The project has two main components:
1. **Hyperbolicity Analysis**: Measuring how hyperbolic BERT's representations are at different layers using multiple hyperbolicity metrics.
2. **Syntax-Semantics Analysis**: Studying the separation and interaction of syntactic vs. semantic signals in both Euclidean and hyperbolic embeddings.

This analysis is using the Universal Dependencies English Treebank ([UD_English-EWT](https://github.com/UniversalDependencies/UD_English-EWT)) as the primary dataset to analyze token-level embeddings generated from BERT across layers.

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

Running the `HyperbolicBERT_Updated.ipynb` notebook will give the hyperbolicity and Q-score analysis