"""
Utilities for extracting and manipulating BERT embeddings.

This module contains functions for:
- Extracting hidden states from BERT layers
- Processing embeddings for tokens and sentences
- Analyzing embedding properties in Euclidean and hyperbolic spaces
"""

import torch
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import defaultdict

def extract_bert_embeddings(model, tokenizer, sentences, device="cuda", batch_size=16, layer_indices=None):
    """
    Extract embeddings from BERT layers for a list of sentences.
    
    Args:
        model: BERT model
        tokenizer: BERT tokenizer
        sentences: List of sentences
        device: Computation device
        batch_size: Batch size for processing
        layer_indices: List of layer indices to extract. If None, extract all layers.
        
    Returns:
        dict: Dictionary mapping layer indices to tensors of shape 
             (num_sentences, max_seq_len, hidden_dim)
    """
    model.eval()
    num_layers = model.config.num_hidden_layers + 1  # +1 for embeddings
    
    if layer_indices is None:
        layer_indices = list(range(num_layers))
    
    # Tokenize sentences
    encodings = tokenizer(
        sentences,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    # Prepare result dictionary
    all_embeddings = {layer_idx: [] for layer_idx in layer_indices}
    
    # Process in batches
    num_batches = (len(sentences) + batch_size - 1) // batch_size
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Extracting embeddings"):
            # Get batch
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(sentences))
            
            batch_input_ids = encodings["input_ids"][start_idx:end_idx].to(device)
            batch_attention_mask = encodings["attention_mask"][start_idx:end_idx].to(device)
            batch_token_type_ids = encodings.get("token_type_ids", None)
            
            if batch_token_type_ids is not None:
                batch_token_type_ids = batch_token_type_ids[start_idx:end_idx].to(device)
            
            # Get embeddings
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                token_type_ids=batch_token_type_ids,
                output_hidden_states=True
            )
            
            # Extract embeddings for each layer
            hidden_states = outputs.hidden_states
            for layer_idx in layer_indices:
                layer_embeddings = hidden_states[layer_idx].cpu()
                all_embeddings[layer_idx].append(layer_embeddings)
    
    # Concatenate embeddings across batches
    for layer_idx in layer_indices:
        all_embeddings[layer_idx] = torch.cat(all_embeddings[layer_idx], dim=0)
    
    return all_embeddings

def filter_special_tokens(embeddings, attention_mask):
    """
    Filter out embeddings of special tokens ([CLS], [SEP], [PAD], etc.).
    
    Args:
        embeddings: Tensor of shape (batch_size, seq_len, hidden_dim)
        attention_mask: Tensor of shape (batch_size, seq_len)
        
    Returns:
        list: List of tensors, each containing embeddings of real tokens for one sentence
    """
    batch_size = embeddings.shape[0]
    filtered_embeddings = []
    
    for i in range(batch_size):
        # Get indices of real tokens (excluding [CLS] and [SEP])
        real_tokens_mask = attention_mask[i] == 1
        # The first token is [CLS], so exclude it
        real_tokens_mask[0] = False
        # Find the position of [SEP] and exclude it too
        sep_pos = attention_mask[i].sum().item() - 1
        if sep_pos > 0:
            real_tokens_mask[sep_pos] = False
        
        # Extract embeddings of real tokens
        filtered_embeddings.append(embeddings[i, real_tokens_mask])
    
    return filtered_embeddings

def get_sentence_embeddings(token_embeddings, pooling="mean"):
    """
    Convert token embeddings to sentence embeddings.
    
    Args:
        token_embeddings: Tensor of shape (batch_size, seq_len, hidden_dim)
        pooling: Pooling method ("mean", "max", "cls")
        
    Returns:
        torch.Tensor: Tensor of shape (batch_size, hidden_dim)
    """
    if pooling == "mean":
        # Mean pooling - take attention mask into account for correct averaging
        return torch.mean(token_embeddings, dim=1)
    elif pooling == "max":
        # Max pooling - take the max over each feature dimension
        return torch.max(token_embeddings, dim=1)[0]
    elif pooling == "cls":
        # [CLS] token pooling - use the first token's embedding
        return token_embeddings[:, 0]
    else:
        raise ValueError(f"Unknown pooling method: {pooling}")

def visualize_embeddings_2d(embeddings, labels=None, method="pca", title=None, save_path=None):
    """
    Visualize embeddings in 2D using PCA or t-SNE.
    
    Args:
        embeddings: Tensor or numpy array of shape (n_samples, hidden_dim)
        labels: Optional labels for coloring points
        method: Dimensionality reduction method ("pca" or "tsne")
        title: Plot title
        save_path: Path to save the visualization
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Convert to numpy if tensor
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    
    # Apply dimensionality reduction
    if method == "pca":
        reducer = PCA(n_components=2)
        reduced_embeddings = reducer.fit_transform(embeddings)
        method_name = "PCA"
    elif method == "tsne":
        reducer = TSNE(n_components=2, perplexity=min(30, max(5, len(embeddings) // 10)))
        reduced_embeddings = reducer.fit_transform(embeddings)
        method_name = "t-SNE"
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if labels is not None:
        # Color by labels
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                reduced_embeddings[mask, 0],
                reduced_embeddings[mask, 1],
                label=f"Class {label}",
                alpha=0.7
            )
        ax.legend()
    else:
        # Single color
        ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.7)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"{method_name} visualization of embeddings")
    
    ax.grid(True, linestyle="--", alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig

def compute_distance_matrices(embeddings, distance_fn, batch_size=32):
    """
    Compute pairwise distance matrices for a batch of embeddings.
    
    Args:
        embeddings: Tensor of shape (batch_size, seq_len, hidden_dim)
        distance_fn: Function that computes distance between two vectors
        batch_size: Batch size for processing to avoid OOM
        
    Returns:
        torch.Tensor: Tensor of shape (batch_size, seq_len, seq_len)
    """
    n_samples = embeddings.shape[0]
    seq_len = embeddings.shape[1]
    distance_matrices = torch.zeros((n_samples, seq_len, seq_len), device=embeddings.device)
    
    # Process in batches
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch_embeddings = embeddings[i:end_idx]
        
        for j in range(batch_embeddings.shape[0]):
            for k in range(seq_len):
                for l in range(seq_len):
                    distance_matrices[i+j, k, l] = distance_fn(
                        batch_embeddings[j, k].unsqueeze(0),
                        batch_embeddings[j, l].unsqueeze(0)
                    )
    
    return distance_matrices

def analyze_embedding_statistics(embeddings_by_layer):
    """
    Compute statistics of embeddings across layers.
    
    Args:
        embeddings_by_layer: Dictionary mapping layer indices to embeddings
        
    Returns:
        dict: Dictionary with statistics for each layer
    """
    stats = {}
    
    for layer_idx, embeddings in embeddings_by_layer.items():
        layer_stats = {}
        
        # Convert to numpy for statistics
        if isinstance(embeddings, torch.Tensor):
            embeddings_np = embeddings.cpu().numpy()
        else:
            embeddings_np = embeddings
        
        # Compute statistics
        layer_stats["mean"] = np.mean(embeddings_np)
        layer_stats["std"] = np.std(embeddings_np)
        layer_stats["min"] = np.min(embeddings_np)
        layer_stats["max"] = np.max(embeddings_np)
        
        # Compute norm statistics
        norms = np.linalg.norm(embeddings_np, axis=-1)
        layer_stats["norm_mean"] = np.mean(norms)
        layer_stats["norm_std"] = np.std(norms)
        
        stats[layer_idx] = layer_stats
    
    return stats

def compute_token_similarity_matrices(embeddings, similarity_fn=None, normalize=True):
    """
    Compute token similarity matrices for sentences.
    
    Args:
        embeddings: List of tensors, each of shape (seq_len, hidden_dim)
        similarity_fn: Function to compute similarity. If None, use cosine similarity.
        normalize: Whether to normalize embeddings before computing similarity
        
    Returns:
        list: List of similarity matrices, each of shape (seq_len, seq_len)
    """
    similarity_matrices = []
    
    if similarity_fn is None:
        # Default to cosine similarity
        similarity_fn = lambda x, y: torch.nn.functional.cosine_similarity(x, y, dim=-1)
    
    for sent_embeddings in embeddings:
        seq_len = sent_embeddings.shape[0]
        sim_matrix = torch.zeros((seq_len, seq_len))
        
        # Normalize if requested
        if normalize:
            sent_embeddings = torch.nn.functional.normalize(sent_embeddings, p=2, dim=-1)
        
        # Compute pairwise similarities
        for i in range(seq_len):
            for j in range(seq_len):
                sim_matrix[i, j] = similarity_fn(
                    sent_embeddings[i].unsqueeze(0),
                    sent_embeddings[j].unsqueeze(0)
                )
        
        similarity_matrices.append(sim_matrix)
    
    return similarity_matrices

def extract_syntactic_features(distance_matrices, dependency_trees):
    """
    Extract features related to syntactic structure from distance matrices.
    
    Args:
        distance_matrices: List of distance matrices, each of shape (seq_len, seq_len)
        dependency_trees: List of dependency trees (adjacency matrices or edge lists)
        
    Returns:
        dict: Dictionary with syntactic features
    """
    features = defaultdict(list)
    
    for distance_matrix, tree in zip(distance_matrices, dependency_trees):
        # Convert tree to adjacency matrix if it's an edge list
        if isinstance(tree, list):
            seq_len = distance_matrix.shape[0]
            adj_matrix = torch.zeros((seq_len, seq_len))
            for parent, child in tree:
                adj_matrix[parent, child] = 1
                adj_matrix[child, parent] = 1  # Undirected
        else:
            adj_matrix = tree
        
        # Extract features
        # 1. Mean distance between connected nodes
        connected_distances = distance_matrix[adj_matrix == 1]
        if len(connected_distances) > 0:
            features["connected_mean_dist"].append(connected_distances.mean().item())
        
        # 2. Mean distance between disconnected nodes
        disconnected_distances = distance_matrix[(adj_matrix == 0) & (torch.ones_like(adj_matrix) - torch.eye(adj_matrix.shape[0])) == 1]
        if len(disconnected_distances) > 0:
            features["disconnected_mean_dist"].append(disconnected_distances.mean().item())
        
        # 3. Ratio of connected to disconnected distances
        if len(disconnected_distances) > 0 and len(connected_distances) > 0:
            ratio = connected_distances.mean().item() / disconnected_distances.mean().item()
            features["connected_disconnected_ratio"].append(ratio)
    
    # Compute averages
    result = {}
    for key, values in features.items():
        result[key] = sum(values) / len(values) if values else float("nan")
    
    return result

def compute_sentence_similarities(sentence_embeddings, similarity_fn=None):
    """
    Compute pairwise similarities between sentence embeddings.
    
    Args:
        sentence_embeddings: Tensor of shape (num_sentences, hidden_dim)
        similarity_fn: Function to compute similarity. If None, use cosine similarity.
        
    Returns:
        torch.Tensor: Similarity matrix of shape (num_sentences, num_sentences)
    """
    num_sentences = sentence_embeddings.shape[0]
    similarity_matrix = torch.zeros((num_sentences, num_sentences))
    
    if similarity_fn is None:
        # Default to cosine similarity
        similarity_fn = lambda x, y: torch.nn.functional.cosine_similarity(x, y, dim=-1)
    
    # Normalize embeddings for cosine similarity
    sentence_embeddings_norm = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=-1)
    
    # Compute pairwise similarities
    for i in range(num_sentences):
        for j in range(num_sentences):
            similarity_matrix[i, j] = similarity_fn(
                sentence_embeddings_norm[i].unsqueeze(0),
                sentence_embeddings_norm[j].unsqueeze(0)
            )
    
    return similarity_matrix