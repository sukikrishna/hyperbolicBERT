"""
Dataset classes for hyperbolic analysis of BERT's representations.

This module implements PyTorch dataset classes for:
- HyperbolicDataset: Base dataset for hyperbolic analysis
- SyntaxDataset: Dataset with dependency parsing information
- SemanticsDataset: Dataset with semantic similarity information
- JointDataset: Dataset with both syntactic and semantic information
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import os
import json

class HyperbolicDataset(Dataset):
    """
    Base dataset for hyperbolic analysis of BERT embeddings.
    
    Stores sentences and their associated embeddings from BERT layers.
    """
    
    def __init__(self, sentences, tokenizer, max_length=128):
        """
        Initialize the dataset with sentences.
        
        Args:
            sentences: List of sentences
            tokenizer: BERT tokenizer
            max_length: Maximum token length for BERT
        """
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Tokenize all sentences
        self.encodings = tokenizer(
            sentences,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Placeholder for cached embeddings
        self.cached_embeddings = {}
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        """
        Get a single example from the dataset.
        
        Returns:
            Dictionary containing input_ids, attention_mask, token_type_ids, 
            and the original sentence
        """
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "token_type_ids": self.encodings.get("token_type_ids", None)[idx] if "token_type_ids" in self.encodings else None,
            "sentence": self.sentences[idx]
        }
    
    def cache_embeddings(self, model, device, layer_indices=None):
        """
        Compute and cache embeddings for all sentences.
        
        Args:
            model: BERT model
            device: Computation device
            layer_indices: List of layer indices to cache. If None, cache all layers.
        """
        model.eval()
        num_layers = model.config.num_hidden_layers + 1  # +1 for embeddings
        
        if layer_indices is None:
            layer_indices = list(range(num_layers))
        
        batch_size = 16
        num_batches = (len(self) + batch_size - 1) // batch_size
        
        all_embeddings = {layer_idx: [] for layer_idx in layer_indices}
        
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Caching embeddings"):
                # Get batch
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(self))
                
                batch_input_ids = self.encodings["input_ids"][start_idx:end_idx].to(device)
                batch_attention_mask = self.encodings["attention_mask"][start_idx:end_idx].to(device)
                batch_token_type_ids = self.encodings.get("token_type_ids", None)
                
                if batch_token_type_ids is not None:
                    batch_token_type_ids = batch_token_type_ids[start_idx:end_idx].to(device)
                
                # Get embeddings
                outputs = model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    token_type_ids=batch_token_type_ids,
                    output_hidden_states=True
                )
                
                # Cache embeddings for each requested layer
                hidden_states = outputs.hidden_states
                for layer_idx in layer_indices:
                    layer_embeddings = hidden_states[layer_idx].cpu()
                    all_embeddings[layer_idx].append(layer_embeddings)
        
        # Concatenate embeddings across batches
        for layer_idx in layer_indices:
            self.cached_embeddings[layer_idx] = torch.cat(all_embeddings[layer_idx], dim=0)


class SyntaxDataset(HyperbolicDataset):
    """
    Dataset for syntactic analysis with dependency parse information.
    
    Stores sentences, their embeddings, and dependency parse trees.
    """
    
    def __init__(self, sentences, dependency_trees, tokenizer, max_length=128):
        """
        Initialize the dataset with sentences and dependency trees.
        
        Args:
            sentences: List of sentences
            dependency_trees: List of dependency trees (e.g., adjacency matrices or edge lists)
            tokenizer: BERT tokenizer
            max_length: Maximum token length for BERT
        """
        super().__init__(sentences, tokenizer, max_length)
        self.dependency_trees = dependency_trees
    
    def __getitem__(self, idx):
        """
        Get a single example with dependency information.
        
        Returns:
            Dictionary containing input_ids, attention_mask, token_type_ids,
            sentence, and dependency_tree
        """
        item = super().__getitem__(idx)
        item["dependency_tree"] = self.dependency_trees[idx]
        return item


class SemanticsDataset(HyperbolicDataset):
    """
    Dataset for semantic analysis with similarity or relatedness information.
    
    Stores sentences, their embeddings, and semantic annotations.
    """
    
    def __init__(self, sentences, semantic_labels, tokenizer, max_length=128):
        """
        Initialize the dataset with sentences and semantic labels.
        
        Args:
            sentences: List of sentences
            semantic_labels: List of semantic labels (e.g., sentiment, similarity scores)
            tokenizer: BERT tokenizer
            max_length: Maximum token length for BERT
        """
        super().__init__(sentences, tokenizer, max_length)
        self.semantic_labels = semantic_labels
    
    def __getitem__(self, idx):
        """
        Get a single example with semantic information.
        
        Returns:
            Dictionary containing input_ids, attention_mask, token_type_ids,
            sentence, and semantic_label
        """
        item = super().__getitem__(idx)
        item["semantic_label"] = self.semantic_labels[idx]
        return item


class JointDataset(HyperbolicDataset):
    """
    Dataset for joint syntactic and semantic analysis.
    
    Stores sentences, their embeddings, dependency trees, and semantic annotations.
    """
    
    def __init__(self, sentences, dependency_trees, semantic_labels, tokenizer, max_length=128):
        """
        Initialize the dataset with sentences, dependency trees, and semantic labels.
        
        Args:
            sentences: List of sentences
            dependency_trees: List of dependency trees
            semantic_labels: List of semantic labels
            tokenizer: BERT tokenizer
            max_length: Maximum token length for BERT
        """
        super().__init__(sentences, tokenizer, max_length)
        self.dependency_trees = dependency_trees
        self.semantic_labels = semantic_labels
    
    def __getitem__(self, idx):
        """
        Get a single example with both dependency and semantic information.
        
        Returns:
            Dictionary containing input_ids, attention_mask, token_type_ids,
            sentence, dependency_tree, and semantic_label
        """
        item = super().__getitem__(idx)
        item["dependency_tree"] = self.dependency_trees[idx]
        item["semantic_label"] = self.semantic_labels[idx]
        return item


class PairDataset(Dataset):
    """
    Dataset for sentence pair tasks (e.g., NLI, semantic similarity).
    
    Stores sentence pairs and their labels.
    """
    
    def __init__(self, sentence_pairs, labels, tokenizer, max_length=128):
        """
        Initialize the dataset with sentence pairs and labels.
        
        Args:
            sentence_pairs: List of (sentence1, sentence2) tuples
            labels: List of labels for each pair
            tokenizer: BERT tokenizer
            max_length: Maximum token length for BERT
        """
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Extract sentence pairs
        self.sentence1 = [pair[0] for pair in sentence_pairs]
        self.sentence2 = [pair[1] for pair in sentence_pairs]
        
        # Tokenize sentence pairs
        self.encodings = tokenizer(
            self.sentence1,
            self.sentence2,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    def __len__(self):
        return len(self.sentence_pairs)
    
    def __getitem__(self, idx):
        """
        Get a single example with sentence pair information.
        
        Returns:
            Dictionary containing input_ids, attention_mask, token_type_ids,
            sentence1, sentence2, and label
        """
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "token_type_ids": self.encodings.get("token_type_ids", None)[idx] if "token_type_ids" in self.encodings else None,
            "sentence1": self.sentence1[idx],
            "sentence2": self.sentence2[idx],
            "label": self.labels[idx]
        }


def create_data_loaders(dataset, batch_size=16, shuffle=True):
    """
    Create DataLoader for a dataset.
    
    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the data
        
    Returns:
        torch.utils.data.DataLoader: DataLoader for the dataset
    """
    from torch.utils.data import DataLoader
    
    def collate_fn(batch):
        """Custom collate function for batching examples."""
        batch_dict = {}
        
        # Extract all keys from the first item
        keys = batch[0].keys()
        
        for key in keys:
            if key in ["input_ids", "attention_mask", "token_type_ids"]:
                # Stack tensor values
                values = [item[key] for item in batch if item[key] is not None]
                if values:
                    batch_dict[key] = torch.stack(values)
                else:
                    batch_dict[key] = None
            elif key in ["dependency_tree", "semantic_label", "label"]:
                # Collect non-tensor values
                batch_dict[key] = [item[key] for item in batch]
            else:
                # Collect string values
                batch_dict[key] = [item[key] for item in batch]
        
        return batch_dict
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )