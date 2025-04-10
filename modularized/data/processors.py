"""
Data processors for loading and preprocessing datasets.

This module includes processors for:
- Universal Dependencies (UD) Treebank data
- GLUE benchmark datasets
"""

import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import conllu

class UDTreebankProcessor:
    """
    Processor for Universal Dependencies (UD) Treebank data.
    
    This processor loads and preprocesses UD treebanks, which provide
    gold-standard dependency parse trees for sentences in various languages.
    """
    
    def __init__(self, base_path, language="en", split="train"):
        """
        Initialize the UD treebank processor.
        
        Args:
            base_path: Path to the UD treebank directory
            language: Language code (e.g., "en" for English)
            split: Data split ("train", "dev", or "test")
        """
        self.base_path = base_path
        self.language = language
        self.split = split
        
        # UD treebank paths typically follow this pattern
        self.treebank_path = self._find_treebank_path()
    
    def _find_treebank_path(self):
        """Find the path to the treebank file for the specified language and split."""
        # Look for common UD treebank directory structures
        language_pattern = f"*{self.language}*"
        for dirpath, _, filenames in os.walk(self.base_path):
            if self.language in dirpath.lower():
                for filename in filenames:
                    if self.split in filename and filename.endswith(".conllu"):
                        return os.path.join(dirpath, filename)
        
        raise FileNotFoundError(f"Could not find UD treebank for language {self.language}, split {self.split}")
    
    def load_data(self, max_sentences=None):
        """
        Load sentences and their dependency parse trees from the treebank.
        
        Args:
            max_sentences: Maximum number of sentences to load
            
        Returns:
            dict: Dictionary with 'sentences' and 'dependency_trees'
        """
        with open(self.treebank_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the CONLL-U formatted treebank
        parsed_data = conllu.parse(content)
        
        sentences = []
        dependency_trees = []
        
        for i, sentence in enumerate(parsed_data):
            if max_sentences is not None and i >= max_sentences:
                break
            
            # Extract tokenized sentence
            tokens = [token['form'] for token in sentence]
            sentences.append(' '.join(tokens))
            
            # Extract dependency tree (as adjacency list)
            # Node indices in CONLL-U start from 1, so we subtract 1
            tree = []
            for token in sentence:
                # Check if it's not a multiword token or empty node
                if isinstance(token['id'], int):
                    head = token['head']
                    # Skip root (head = 0)
                    if head > 0:
                        # Convert 1-based to 0-based indexing
                        child_idx = token['id'] - 1
                        head_idx = head - 1
                        tree.append((head_idx, child_idx))
            
            dependency_trees.append(tree)
        
        return {
            'sentences': sentences,
            'dependency_trees': dependency_trees
        }
    
    def create_adjacency_matrices(self, dependency_trees, directed=False):
        """
        Convert dependency trees to adjacency matrices.
        
        Args:
            dependency_trees: List of dependency trees (parent-child pairs)
            directed: Whether to create directed adjacency matrices
            
        Returns:
            list: List of adjacency matrices
        """
        adjacency_matrices = []
        
        for tree in dependency_trees:
            # Find the size of the tree
            max_idx = 0
            for parent, child in tree:
                max_idx = max(max_idx, parent, child)
            size = max_idx + 1
            
            # Create adjacency matrix
            adj_matrix = torch.zeros((size, size))
            for parent, child in tree:
                adj_matrix[parent, child] = 1
                if not directed:
                    adj_matrix[child, parent] = 1
            
            adjacency_matrices.append(adj_matrix)
        
        return adjacency_matrices
    
    def get_dataset(self, max_sentences=None):
        """
        Get a dataset with sentences and dependency trees.
        
        Args:
            max_sentences: Maximum number of sentences to include
            
        Returns:
            dict: Dictionary with 'sentences' and 'dependency_trees'
        """
        data = self.load_data(max_sentences)
        adj_matrices = self.create_adjacency_matrices(data['dependency_trees'])
        
        return {
            'sentences': data['sentences'],
            'dependency_trees': adj_matrices
        }

class GlueProcessor:
    """
    Processor for GLUE benchmark datasets.
    
    This processor loads and preprocesses datasets from the GLUE benchmark,
    which includes various NLP tasks like sentiment analysis, paraphrase detection, etc.
    """
    
    TASKS = {
        'cola': ('CoLA', ['sentence'], 'label'),
        'sst2': ('SST-2', ['sentence'], 'label'),
        'mrpc': ('MRPC', ['sentence1', 'sentence2'], 'label'),
        'qqp': ('QQP', ['question1', 'question2'], 'label'),
        'stsb': ('STS-B', ['sentence1', 'sentence2'], 'label'),
        'mnli': ('MNLI', ['premise', 'hypothesis'], 'label'),
        'qnli': ('QNLI', ['question', 'sentence'], 'label'),
        'rte': ('RTE', ['sentence1', 'sentence2'], 'label'),
        'wnli': ('WNLI', ['sentence1', 'sentence2'], 'label')
    }
    
    def __init__(self, base_path, task_name, split="train"):
        """
        Initialize the GLUE processor.
        
        Args:
            base_path: Path to the GLUE datasets directory
            task_name: Name of the GLUE task (e.g., 'sst2', 'cola')
            split: Data split ("train", "dev", or "test")
        """
        self.base_path = base_path
        
        # Validate task name
        task_name = task_name.lower()
        if task_name not in self.TASKS:
            raise ValueError(f"Unknown GLUE task: {task_name}. "
                            f"Available tasks: {list(self.TASKS.keys())}")
        
        self.task_name = task_name
        self.task_dir, self.text_fields, self.label_field = self.TASKS[task_name]
        self.split = split
        
        # Determine file path
        if split == "test":
            self.file_path = os.path.join(base_path, self.task_dir, f"{split}.tsv")
        else:
            self.file_path = os.path.join(base_path, self.task_dir, f"{split}.tsv")
    
    def load_data(self, max_examples=None):
        """
        Load examples from the GLUE dataset.
        
        Args:
            max_examples: Maximum number of examples to load
            
        Returns:
            dict: Dictionary with 'sentences' and 'labels'
        """
        # Load the dataset
        df = pd.read_csv(self.file_path, sep='\t')
        
        # Limit examples if needed
        if max_examples is not None:
            df = df.head(max_examples)
        
        # Extract text and labels
        if len(self.text_fields) == 1:
            # Single sentence tasks
            sentences = df[self.text_fields[0]].tolist()
            paired_inputs = None
        else:
            # Sentence pair tasks
            sentences = df[self.text_fields[0]].tolist()
            paired_inputs = df[self.text_fields[1]].tolist()
        
        # Extract labels if available
        if self.label_field in df.columns:
            labels = df[self.label_field].tolist()
        else:
            # Test set might not have labels
            labels = None
        
        return {
            'sentences': sentences,
            'paired_inputs': paired_inputs,
            'labels': labels
        }
    
    def get_dataset(self, max_examples=None):
        """
        Get a dataset with sentences and labels.
        
        Args:
            max_examples: Maximum number of examples to include
            
        Returns:
            dict: Dictionary with 'sentences', 'paired_inputs' (if applicable), and 'labels'
        """
        return self.load_data(max_examples)

class HyperbolicDataset(Dataset):
    """
    A PyTorch Dataset for hyperbolic representation analysis.
    
    This dataset stores sentences and their corresponding embeddings,
    facilitating batch processing for hyperbolic analysis.
    """
    
    def __init__(self, sentences, tokenizer, max_length=512):
        """
        Initialize the dataset.
        
        Args:
            sentences: List of sentences
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
        """
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Pre-tokenize all sentences
        self.encodings = tokenizer(
            sentences,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'token_type_ids': self.encodings.get('token_type_ids', None)[idx] if 'token_type_ids' in self.encodings else None,
            'sentence': self.sentences[idx]
        }

class SyntaxSemanticsDataset(Dataset):
    """
    A PyTorch Dataset for joint syntactic and semantic analysis.
    
    This dataset stores sentences, their dependency trees (for syntax),
    and semantic relatedness scores (for semantics).
    """
    
    def __init__(self, sentences, tokenizer, dependency_trees=None, semantic_scores=None, max_length=512):
        """
        Initialize the dataset.
        
        Args:
            sentences: List of sentences
            tokenizer: BERT tokenizer
            dependency_trees: List of dependency trees
            semantic_scores: Matrix of semantic relatedness scores
            max_length: Maximum sequence length
        """
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.dependency_trees = dependency_trees
        self.semantic_scores = semantic_scores
        self.max_length = max_length
        
        # Pre-tokenize all sentences
        self.encodings = tokenizer(
            sentences,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        item = {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'token_type_ids': self.encodings.get('token_type_ids', None)[idx] if 'token_type_ids' in self.encodings else None,
            'sentence': self.sentences[idx]
        }
        
        if self.dependency_trees is not None:
            item['dependency_tree'] = self.dependency_trees[idx]
        
        if self.semantic_scores is not None:
            item['semantic_scores'] = self.semantic_scores[idx]
        
        return item

def create_dataloader(dataset, batch_size=8, shuffle=False):
    """
    Create a DataLoader for the given dataset.
    
    Args:
        dataset: A PyTorch Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        
    Returns:
        torch.utils.data.DataLoader: DataLoader for the dataset
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'token_type_ids': torch.stack([item['token_type_ids'] for item in batch]) if batch[0]['token_type_ids'] is not None else None,
            'sentences': [item['sentence'] for item in batch],
            'dependency_trees': [item.get('dependency_tree') for item in batch] if 'dependency_tree' in batch[0] else None,
            'semantic_scores': [item.get('semantic_scores') for item in batch] if 'semantic_scores' in batch[0] else None
        }
    )

def load_ud_and_glue_data(ud_path, glue_path, ud_language="en", glue_task="sst2", max_sentences=100):
    """
    Load both UD treebank and GLUE data for joint syntax-semantics analysis.
    
    Args:
        ud_path: Path to UD treebank directory
        glue_path: Path to GLUE datasets directory
        ud_language: Language code for UD treebank
        glue_task: GLUE task name
        max_sentences: Maximum number of sentences to load
        
    Returns:
        dict: Dictionary with UD and GLUE datasets
    """
    # Load UD treebank data
    ud_processor = UDTreebankProcessor(ud_path, language=ud_language)
    ud_data = ud_processor.get_dataset(max_sentences=max_sentences)
    
    # Load GLUE data
    glue_processor = GlueProcessor(glue_path, task_name=glue_task)
    glue_data = glue_processor.get_dataset(max_examples=max_sentences)
    
    return {
        'ud': ud_data,
        'glue': glue_data
    }