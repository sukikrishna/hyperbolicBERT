"""
BERT fine-tuning utilities for hyperbolic embeddings.

This module implements:
- Hyperbolic fine-tuning for BERT
- Evaluation of fine-tuned models
- Tracking of hyperbolicity during fine-tuning
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time
from transformers import (
    BertForSequenceClassification,
    BertConfig,
    get_linear_schedule_with_warmup,
    AdamW
)
import geoopt as gt
from geoopt.optim import RiemannianAdam, RiemannianSGD

from utils.hyperbolic_utils import (
    init_poincare_ball,
    euclidean_to_poincare,
    poincare_distance,
    mobius_addition,
    mobius_matrix_multiplication
)
from losses import HyperbolicDistanceLoss, MobiusMLPLoss, SyntaxSemanticsLoss

class HyperbolicBertClassifier(nn.Module):
    """
    BERT classifier with a hyperbolic output layer.
    
    Projects BERT embeddings to the Poincaré ball and performs 
    classification in hyperbolic space.
    """
    
    def __init__(self, bert_model, num_labels, curvature=-1.0, hidden_size=768, 
                 dropout_prob=0.1, device="cuda"):
        """
        Initialize the hyperbolic BERT classifier.
        
        Args:
            bert_model: Pre-trained BERT model
            num_labels: Number of output classes
            curvature: Curvature of the hyperbolic space
            hidden_size: Size of BERT hidden representations
            dropout_prob: Dropout probability
            device: Computation device
        """
        super().__init__()
        self.bert = bert_model
        self.num_labels = num_labels
        self.ball = gt.Stereographic(c=curvature)
        self.device = device
        
        # Hyperbolic projection layer
        self.projection = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size // 2)
        )
        
        # Class representatives in the Poincaré ball
        # Initialize class representatives near the origin
        bound = 0.1
        poincare_weight = torch.zeros(num_labels, hidden_size // 2).uniform_(-bound, bound)
        poincare_weight = self.ball.expmap0(poincare_weight)
        self.poincare_weight = gt.ManifoldParameter(poincare_weight, manifold=self.ball)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_length)
            attention_mask: Attention mask of shape (batch_size, seq_length)
            token_type_ids: Token type IDs of shape (batch_size, seq_length)
            labels: Optional labels of shape (batch_size,)
            
        Returns:
            dict: Dictionary with model outputs
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True
        )
        
        # Get [CLS] token embedding
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Project to tangent space at origin
        projected = self.projection(cls_output)
        
        # Map to Poincaré ball
        poincare_embeddings = self.ball.expmap0(projected)
        
        # Compute hyperbolic distances to class representatives
        batch_size = poincare_embeddings.shape[0]
        logits = torch.zeros(batch_size, self.num_labels, device=self.device)
        
        for i in range(self.num_labels):
            class_rep = self.poincare_weight[i].unsqueeze(0).expand(batch_size, -1)
            distances = self.ball.dist(poincare_embeddings, class_rep)
            # Use negative distance as logits (closer = higher score)
            logits[:, i] = -distances.squeeze()
        
        output_dict = {"logits": logits, "embeddings": poincare_embeddings}
        
        # Compute loss if labels are provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            output_dict["loss"] = loss
        
        return output_dict


class HyperbolicProbeFineTuner:
    """
    Fine-tuner for hyperbolic structural probes on BERT.
    
    Trains a hyperbolic probe to extract syntactic information from BERT.
    """
    
    def __init__(self, model, probe, optimizer, scheduler, device, curvature=-1.0):
        """
        Initialize the fine-tuner.
        
        Args:
            model: BERT model
            probe: Hyperbolic probe model
            optimizer: Optimizer for training
            scheduler: Learning rate scheduler
            device: Computation device
            curvature: Curvature of the hyperbolic space
        """
        self.model = model
        self.probe = probe
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.ball = gt.Stereographic(c=curvature)
        
        # Set up loss function
        self.loss_fn = HyperbolicDistanceLoss(curvature=curvature)
        
        # Tracking metrics
        self.train_losses = []
        self.val_losses = []
        self.hyperbolicity_measures = []
    
    def train_epoch(self, train_dataloader):
        """
        Train the probe for one epoch.
        
        Args:
            train_dataloader: DataLoader for training data
            
        Returns:
            float: Average training loss
        """
        self.model.eval()  # Freeze BERT model
        self.probe.train()
        
        total_loss = 0
        num_batches = len(train_dataloader)
        
        for batch in tqdm(train_dataloader, desc="Training"):
            # Unpack batch
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)
            
            dependency_trees = batch["dependency_tree"]
            
            # Convert dependency trees to distance matrices
            target_distances = []
            for tree in dependency_trees:
                n = len(tree)
                distances = torch.zeros((n, n), device=self.device)
                
                # Build adjacency matrix
                adjacency = torch.zeros((n, n), device=self.device)
                for parent, child in tree:
                    adjacency[parent, child] = 1
                    adjacency[child, parent] = 1  # Undirected
                
                # Compute shortest paths using Floyd-Warshall algorithm
                dist = adjacency.clone()
                dist[dist == 0] = float('inf')
                dist.fill_diagonal_(0)
                
                for k in range(n):
                    for i in range(n):
                        for j in range(n):
                            if dist[i, k] + dist[k, j] < dist[i, j]:
                                dist[i, j] = dist[i, k] + dist[k, j]
                
                target_distances.append(dist)
            
            # Get BERT embeddings
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    output_hidden_states=True
                )
                hidden_states = outputs.hidden_states[-1]  # Use last layer
            
            # Forward pass through probe
            predicted_distances = self.probe(hidden_states)
            
            # Compute loss
            loss = 0
            for i, (pred, target) in enumerate(zip(predicted_distances, target_distances)):
                seq_len = attention_mask[i].sum().item()
                pred = pred[:seq_len, :seq_len]
                target = target[:seq_len, :seq_len].to(self.device)
                loss += self.loss_fn(pred.unsqueeze(0), target.unsqueeze(0))
            
            loss = loss / len(dependency_trees)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # Update learning rate
        if self.scheduler is not None:
            self.scheduler.step()
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def evaluate(self, val_dataloader):
        """
        Evaluate the probe on validation data.
        
        Args:
            val_dataloader: DataLoader for validation data
            
        Returns:
            float: Average validation loss
        """
        self.model.eval()
        self.probe.eval()
        
        total_loss = 0
        num_batches = len(val_dataloader)
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Evaluating"):
                # Unpack batch
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch.get("token_type_ids", None)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)
                
                dependency_trees = batch["dependency_tree"]
                
                # Convert dependency trees to distance matrices
                target_distances = []
                for tree in dependency_trees:
                    n = len(tree)
                    distances = torch.zeros((n, n), device=self.device)
                    
                    # Build adjacency matrix
                    adjacency = torch.zeros((n, n), device=self.device)
                    for parent, child in tree:
                        adjacency[parent, child] = 1
                        adjacency[child, parent] = 1  # Undirected
                    
                    # Compute shortest paths using Floyd-Warshall algorithm
                    dist = adjacency.clone()
                    dist[dist == 0] = float('inf')
                    dist.fill_diagonal_(0)
                    
                    for k in range(n):
                        for i in range(n):
                            for j in range(n):
                                if dist[i, k] + dist[k, j] < dist[i, j]:
                                    dist[i, j] = dist[i, k] + dist[k, j]
                    
                    target_distances.append(dist)
                
                # Get BERT embeddings
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    output_hidden_states=True
                )
                hidden_states = outputs.hidden_states[-1]  # Use last layer
                
                # Forward pass through probe
                predicted_distances = self.probe(hidden_states)
                
                # Compute loss
                loss = 0
                for i, (pred, target) in enumerate(zip(predicted_distances, target_distances)):
                    seq_len = attention_mask[i].sum().item()
                    pred = pred[:seq_len, :seq_len]
                    target = target[:seq_len, :seq_len].to(self.device)
                    loss += self.loss_fn(pred.unsqueeze(0), target.unsqueeze(0))
                
                loss = loss / len(dependency_trees)
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def train(self, train_dataloader, val_dataloader, num_epochs=5):
        """
        Train the probe for multiple epochs.
        
        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            num_epochs: Number of training epochs
            
        Returns:
            dict: Dictionary with training history
        """
        best_val_loss = float("inf")
        best_model = None
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            # Train one epoch
            train_loss = self.train_epoch(train_dataloader)
            
            # Evaluate
            val_loss = self.evaluate(val_dataloader)
            
            # Measure hyperbolicity
            hyperbolicity = self.measure_hyperbolicity()
            self.hyperbolicity_measures.append(hyperbolicity)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Hyperbolicity: {hyperbolicity:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = {
                    "probe_state_dict": self.probe.state_dict(),
                    "epoch": epoch + 1,
                    "val_loss": val_loss
                }
        
        # Load best model
        if best_model is not None:
            self.probe.load_state_dict(best_model["probe_state_dict"])
        
        # Return training history
        history = {
            "train_loss": self.train_losses,
            "val_loss": self.val_losses,
            "hyperbolicity": self.hyperbolicity_measures,
            "best_epoch": best_model["epoch"] if best_model else num_epochs,
            "best_val_loss": best_model["val_loss"] if best_model else self.val_losses[-1]
        }
        
        return history
    
    def measure_hyperbolicity(self):
        """
        Measure the hyperbolicity of the probe.
        
        Returns:
            float: Hyperbolicity measure (e.g., delta-hyperbolicity)
        """
        # Simple implementation for demonstration
        # In a real implementation, compute proper hyperbolicity measures
        
        # Get weights
        weights = None
        for p in self.probe.parameters():
            if weights is None:
                weights = p.data.flatten()
            else:
                weights = torch.cat([weights, p.data.flatten()])
        
        # Project to Poincaré ball
        poincare_weights = self.ball.expmap0(weights.reshape(-1, 10))
        
        # Compute sample of pairwise distances
        n = min(100, poincare_weights.shape[0])
        indices = torch.randperm(poincare_weights.shape[0])[:n]
        sample = poincare_weights[indices]
        
        distances = torch.zeros((n, n), device=self.device)
        for i in range(n):
            for j in range(i+1, n):
                distances[i, j] = self.ball.dist(sample[i].unsqueeze(0), sample[j].unsqueeze(0))
                distances[j, i] = distances[i, j]
        
        # Simple estimate of delta-hyperbolicity
        hyperbolicity = 0.0
        num_samples = min(n, 20)  # Limit number of samples for efficiency
        
        for i in range(num_samples):
            for j in range(i+1, num_samples):
                for k in range(j+1, num_samples):
                    for l in range(k+1, num_samples):
                        # Compute Gromov products
                        gij_kl = (distances[i, j] + distances[k, l]) / 2
                        gik_jl = (distances[i, k] + distances[j, l]) / 2
                        gil_jk = (distances[i, l] + distances[j, k]) / 2
                        
                        # Sort the three terms
                        products = sorted([gij_kl, gik_jl, gil_jk])
                        
                        # Delta is half the difference between the two largest products
                        delta = (products[2] - products[1]) / 2
                        hyperbolicity = max(hyperbolicity, delta)
        
        return hyperbolicity.item() if isinstance(hyperbolicity, torch.Tensor) else hyperbolicity
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history.
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        fig, axes = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot loss
        epochs = range(1, len(self.train_losses) + 1)
        axes[0].plot(epochs, self.train_losses, 'b-', label='Training Loss')
        axes[0].plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot hyperbolicity
        axes[1].plot(epochs, self.hyperbolicity_measures, 'g-')
        axes[1].set_title('Hyperbolicity during Training')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('δ-Hyperbolicity')
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class BertHyperbolicFineTuner:
    """
    Fine-tuner for BERT with hyperbolic classification layer.
    
    Trains BERT with a hyperbolic classification head for NLP tasks.
    """
    
    def __init__(self, model, optimizer, scheduler, device, num_labels, curvature=-1.0):
        """
        Initialize the fine-tuner.
        
        Args:
            model: HyperbolicBertClassifier model
            optimizer: Optimizer for training
            scheduler: Learning rate scheduler
            device: Computation device
            num_labels: Number of output classes
            curvature: Curvature of the hyperbolic space
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_labels = num_labels
        self.ball = gt.Stereographic(c=curvature)
        
        # Tracking metrics
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.hyperbolicity_measures = []
    
    def train_epoch(self, train_dataloader):
        """
        Train the model for one epoch.
        
        Args:
            train_dataloader: DataLoader for training data
            
        Returns:
            tuple: (average loss, accuracy)
        """
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        num_batches = len(train_dataloader)
        
        for batch in tqdm(train_dataloader, desc="Training"):
            # Unpack batch
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)
            
            labels = batch["label"]
            if isinstance(labels, list):
                labels = torch.tensor(labels, device=self.device)
            else:
                labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
            
            loss = outputs["loss"]
            logits = outputs["logits"]
            
            # Compute accuracy
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # Update learning rate
        if self.scheduler is not None:
            self.scheduler.step()
        
        avg_loss = total_loss / num_batches
        accuracy = correct / total
        
        self.train_losses.append(avg_loss)
        self.train_accs.append(accuracy)
        
        return avg_loss, accuracy
    
    def evaluate(self, val_dataloader):
        """
        Evaluate the model on validation data.
        
        Args:
            val_dataloader: DataLoader for validation data
            
        Returns:
            tuple: (average loss, accuracy)
        """
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        num_batches = len(val_dataloader)
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Evaluating"):
                # Unpack batch
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch.get("token_type_ids", None)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)
                
                labels = batch["label"]
                if isinstance(labels, list):
                    labels = torch.tensor(labels, device=self.device)
                else:
                    labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )
                
                loss = outputs["loss"]
                logits = outputs["logits"]
                
                # Compute accuracy
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        accuracy = correct / total
        
        self.val_losses.append(avg_loss)
        self.val_accs.append(accuracy)
        
        return avg_loss, accuracy
    
    def train(self, train_dataloader, val_dataloader, num_epochs=3):
        """
        Train the model for multiple epochs.
        
        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            num_epochs: Number of training epochs
            
        Returns:
            dict: Dictionary with training history
        """
        best_val_loss = float("inf")
        best_model = None
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            # Train one epoch
            train_loss, train_acc = self.train_epoch(train_dataloader)
            
            # Evaluate
            val_loss, val_acc = self.evaluate(val_dataloader)
            
            # Measure hyperbolicity
            hyperbolicity = self.measure_hyperbolicity()
            self.hyperbolicity_measures.append(hyperbolicity)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Hyperbolicity: {hyperbolicity:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = {
                    "model_state_dict": self.model.state_dict(),
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                    "val_acc": val_acc
                }
        
        # Load best model
        if best_model is not None:
            self.model.load_state_dict(best_model["model_state_dict"])
        
        # Return training history
        history = {
            "train_loss": self.train_losses,
            "train_acc": self.train_accs,
            "val_loss": self.val_losses,
            "val_acc": self.val_accs,
            "hyperbolicity": self.hyperbolicity_measures,
            "best_epoch": best_model["epoch"] if best_model else num_epochs,
            "best_val_loss": best_model["val_loss"] if best_model else self.val_losses[-1],
            "best_val_acc": best_model["val_acc"] if best_model else self.val_accs[-1]
        }
        
        return history
    
    def measure_hyperbolicity(self):
        """
        Measure the hyperbolicity of the model's hyperbolic layer.
        
        Returns:
            float: Hyperbolicity measure
        """
        # Extract class representatives in the Poincaré ball
        poincare_weight = self.model.poincare_weight.data
        
        # Compute pairwise distances
        n = self.num_labels
        distances = torch.zeros((n, n), device=self.device)
        
        for i in range(n):
            for j in range(i+1, n):
                distances[i, j] = self.ball.dist(
                    poincare_weight[i].unsqueeze(0),
                    poincare_weight[j].unsqueeze(0)
                )
                distances[j, i] = distances[i, j]
        
        # Simple estimate of delta-hyperbolicity
        hyperbolicity = 0.0
        
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        # Compute Gromov products
                        gij_kl = (distances[i, j] + distances[k, l]) / 2
                        gik_jl = (distances[i, k] + distances[j, l]) / 2
                        gil_jk = (distances[i, l] + distances[j, k]) / 2
                        
                        # Sort the three terms
                        products = sorted([gij_kl, gik_jl, gil_jk])
                        
                        # Delta is half the difference between the two largest products
                        delta = (products[2] - products[1]) / 2
                        hyperbolicity = max(hyperbolicity, delta)
        
        return hyperbolicity.item() if isinstance(hyperbolicity, torch.Tensor) else hyperbolicity
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history.
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
        
        # Plot loss
        epochs = range(1, len(self.train_losses) + 1)
        axes[0].plot(epochs, self.train_losses, 'b-', label='Training Loss')
        axes[0].plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot accuracy
        axes[1].plot(epochs, self.train_accs, 'b-', label='Training Accuracy')
        axes[1].plot(epochs, self.val_accs, 'r-', label='Validation Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_ylim([0, 1])
        axes[1].legend()
        axes[1].grid(True)
        
        # Plot hyperbolicity
        axes[2].plot(epochs, self.hyperbolicity_measures, 'g-')
        axes[2].set_title('Hyperbolicity during Training')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('δ-Hyperbolicity')
        axes[2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def create_hyperbolic_optimizer(model, learning_rate=2e-5, adam_epsilon=1e-8,
                                weight_decay=0.01, use_riemannian=True):
    """
    Create an optimizer for hyperbolic models.
    
    Args:
        model: Model with hyperbolic parameters
        learning_rate: Learning rate
        adam_epsilon: Epsilon for Adam optimizer
        weight_decay: Weight decay factor
        use_riemannian: Whether to use Riemannian optimizer
        
    Returns:
        torch.optim.Optimizer: Optimizer for the model
    """
    # Separate hyperbolic and Euclidean parameters
    hyperbolic_params = []
    euclidean_params = []
    
    for name, param in model.named_parameters():
        if hasattr(param, "manifold") and param.manifold is not None:
            hyperbolic_params.append(param)
        else:
            euclidean_params.append(param)
    
    if use_riemannian and hyperbolic_params:
        # Create parameter groups
        param_groups = [
            {"params": euclidean_params},
            {"params": hyperbolic_params, "riemannian": True}
        ]
        
        # Create Riemannian Adam optimizer
        optimizer = RiemannianAdam(
            param_groups,
            lr=learning_rate,
            eps=adam_epsilon,
            weight_decay=weight_decay
        )
    else:
        # Create standard Adam optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            eps=adam_epsilon,
            weight_decay=weight_decay
        )
    
    return optimizer


def create_scheduler(optimizer, num_training_steps, warmup_steps=0):
    """
    Create a learning rate scheduler.
    
    Args:
        optimizer: Optimizer
        num_training_steps: Total number of training steps
        warmup_steps: Number of warmup steps
        
    Returns:
        torch.optim.lr_scheduler._LRScheduler: Learning rate scheduler
    """
    return get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )


def save_model(model, tokenizer, output_dir, hyperbolicity_history=None):
    """
    Save model, tokenizer, and training history.
    
    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        output_dir: Output directory
        hyperbolicity_history: Optional hyperbolicity history
    """
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    # Save hyperbolicity history
    if hyperbolicity_history is not None:
        with open(os.path.join(output_dir, "hyperbolicity_history.json"), "w") as f:
            json.dump(hyperbolicity_history, f)
    
    print(f"Model saved to {output_dir}")


def load_hyperbolic_model(model_path, num_labels, device, curvature=-1.0):
    """
    Load a hyperbolic BERT model from a directory.
    
    Args:
        model_path: Path to the saved model
        num_labels: Number of output classes
        device: Computation device
        curvature: Curvature of the hyperbolic space
        
    Returns:
        HyperbolicBertClassifier: Loaded model
    """
    # Load BERT configuration
    config = BertConfig.from_pretrained(model_path)
    
    # Update config with num_labels
    config.num_labels = num_labels
    
    # Load BERT model
    bert_model = BertForSequenceClassification.from_pretrained(
        model_path,
        config=config
    )
    
    # Create hyperbolic model
    hyperbolic_model = HyperbolicBertClassifier(
        bert_model=bert_model,
        num_labels=num_labels,
        curvature=curvature,
        hidden_size=config.hidden_size,
        device=device
    )
    
    # Load hyperbolic model weights
    hyperbolic_model.load_state_dict(torch.load(
        os.path.join(model_path, "pytorch_model.bin"),
        map_location=device
    ))
    
    return hyperbolic_model.to(device)