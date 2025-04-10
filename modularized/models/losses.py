"""
Loss functions for Euclidean and hyperbolic spaces.

This module implements custom loss functions for:
- Euclidean distance loss
- Hyperbolic distance loss
- Joint syntax-semantics loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt as gt
from geoopt.manifolds.stereographic import StereographicExact

class EuclideanDistanceLoss(nn.Module):
    """
    Loss function based on Euclidean distances.
    
    For structural probing, we want model distances to match dependency tree distances.
    Based on: Hewitt, J., & Manning, C. D. (2019). A Structural Probe for Finding Syntax in Word Representations.
    """
    
    def __init__(self, normalize=True, squared=False):
        """
        Initialize the Euclidean distance loss.
        
        Args:
            normalize: Whether to normalize distances by sentence length
            squared: Whether to use squared distance
        """
        super().__init__()
        self.normalize = normalize
        self.squared = squared
    
    def forward(self, predicted_distances, target_distances, attention_mask=None):
        """
        Compute the Euclidean distance loss.
        
        Args:
            predicted_distances: Predicted distances of shape (batch_size, seq_len, seq_len)
            target_distances: Target distances of shape (batch_size, seq_len, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            
        Returns:
            torch.Tensor: Loss value
        """
        # Create a mask for valid token pairs
        if attention_mask is not None:
            batch_size, seq_len = attention_mask.size()
            # Create mask for valid token pairs
            token_pair_mask = attention_mask.unsqueeze(2) * attention_mask.unsqueeze(1)
            # Exclude masked positions from loss computation
            predicted_distances = predicted_distances * token_pair_mask
            target_distances = target_distances * token_pair_mask
        else:
            batch_size, seq_len, _ = predicted_distances.size()
            token_pair_mask = torch.ones_like(predicted_distances)
        
        # Compute absolute differences
        if self.squared:
            diffs = (predicted_distances - target_distances) ** 2
        else:
            diffs = torch.abs(predicted_distances - target_distances)
        
        # Sum differences for each sentence
        sentence_losses = torch.sum(diffs, dim=(1, 2))
        
        if self.normalize:
            # Count valid token pairs for normalization
            valid_pairs = torch.sum(token_pair_mask, dim=(1, 2))
            # Avoid division by zero
            valid_pairs = torch.max(valid_pairs, torch.ones_like(valid_pairs))
            # Normalize by number of valid pairs
            sentence_losses = sentence_losses / valid_pairs
        
        # Average loss across the batch
        loss = torch.mean(sentence_losses)
        
        return loss


class HyperbolicDistanceLoss(nn.Module):
    """
    Loss function based on hyperbolic distances in the Poincaré ball.
    
    For structural probing in hyperbolic space, we want hyperbolic distances 
    to match dependency tree distances.
    """
    
    def __init__(self, curvature=-1.0, normalize=True, squared=False):
        """
        Initialize the hyperbolic distance loss.
        
        Args:
            curvature: Curvature of the hyperbolic space
            normalize: Whether to normalize distances by sentence length
            squared: Whether to use squared distance
        """
        super().__init__()
        self.ball = StereographicExact(c=curvature)
        self.normalize = normalize
        self.squared = squared
    
    def forward(self, predicted_distances, target_distances, attention_mask=None):
        """
        Compute the hyperbolic distance loss.
        
        Args:
            predicted_distances: Predicted distances of shape (batch_size, seq_len, seq_len)
            target_distances: Target distances of shape (batch_size, seq_len, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            
        Returns:
            torch.Tensor: Loss value
        """
        # Create a mask for valid token pairs
        if attention_mask is not None:
            batch_size, seq_len = attention_mask.size()
            # Create mask for valid token pairs
            token_pair_mask = attention_mask.unsqueeze(2) * attention_mask.unsqueeze(1)
            # Exclude masked positions from loss computation
            predicted_distances = predicted_distances * token_pair_mask
            target_distances = target_distances * token_pair_mask
        else:
            batch_size, seq_len, _ = predicted_distances.size()
            token_pair_mask = torch.ones_like(predicted_distances)
        
        # Compute absolute differences
        if self.squared:
            diffs = (predicted_distances - target_distances) ** 2
        else:
            diffs = torch.abs(predicted_distances - target_distances)
        
        # Sum differences for each sentence
        sentence_losses = torch.sum(diffs, dim=(1, 2))
        
        if self.normalize:
            # Count valid token pairs for normalization
            valid_pairs = torch.sum(token_pair_mask, dim=(1, 2))
            # Avoid division by zero
            valid_pairs = torch.max(valid_pairs, torch.ones_like(valid_pairs))
            # Normalize by number of valid pairs
            sentence_losses = sentence_losses / valid_pairs
        
        # Average loss across the batch
        loss = torch.mean(sentence_losses)
        
        return loss


class MobiusMLPLoss(nn.Module):
    """
    Loss function for classification in the Poincaré ball.
    
    Uses Möbius operations for classification in hyperbolic space.
    """
    
    def __init__(self, curvature=-1.0, reduction='mean'):
        """
        Initialize the Möbius MLP loss.
        
        Args:
            curvature: Curvature of the hyperbolic space
            reduction: Reduction method for the loss ('mean', 'sum', 'none')
        """
        super().__init__()
        self.ball = StereographicExact(c=curvature)
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        Compute the classification loss in hyperbolic space.
        
        Args:
            logits: Predicted logits in Poincaré ball of shape (batch_size, num_classes)
            targets: Target class indices of shape (batch_size,)
            
        Returns:
            torch.Tensor: Loss value
        """
        # Map hyperbolic logits to the tangent space at the origin
        euclidean_logits = self.ball.logmap0(logits)
        
        # Apply softmax and compute cross-entropy loss
        loss = F.cross_entropy(
            euclidean_logits, targets, 
            reduction=self.reduction
        )
        
        return loss


class HyperbolicTripletLoss(nn.Module):
    """
    Triplet loss in hyperbolic space.
    
    For learning embeddings where similar items are closer in hyperbolic space.
    """
    
    def __init__(self, curvature=-1.0, margin=1.0, reduction='mean'):
        """
        Initialize the hyperbolic triplet loss.
        
        Args:
            curvature: Curvature of the hyperbolic space
            margin: Margin for the triplet loss
            reduction: Reduction method for the loss ('mean', 'sum', 'none')
        """
        super().__init__()
        self.ball = StereographicExact(c=curvature)
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, anchors, positives, negatives):
        """
        Compute the triplet loss in hyperbolic space.
        
        Args:
            anchors: Anchor points in Poincaré ball of shape (batch_size, dim)
            positives: Positive points in Poincaré ball of shape (batch_size, dim)
            negatives: Negative points in Poincaré ball of shape (batch_size, dim)
            
        Returns:
            torch.Tensor: Loss value
        """
        # Compute hyperbolic distances
        pos_distances = self.ball.dist(anchors, positives)
        neg_distances = self.ball.dist(anchors, negatives)
        
        # Compute triplet loss
        losses = F.relu(pos_distances - neg_distances + self.margin)
        
        # Apply reduction
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:  # 'none'
            return losses


class SyntaxSemanticsLoss(nn.Module):
    """
    Joint loss for syntax and semantics tasks.
    
    Combines a hyperbolic loss for syntax with a Euclidean loss for semantics.
    """
    
    def __init__(self, syntax_weight=0.5, curvature=-1.0, normalize=True):
        """
        Initialize the joint syntax-semantics loss.
        
        Args:
            syntax_weight: Weight for the syntax loss (1 - syntax_weight for semantics)
            curvature: Curvature of the hyperbolic space
            normalize: Whether to normalize distances
        """
        super().__init__()
        self.syntax_weight = syntax_weight
        self.hyperbolic_loss = HyperbolicDistanceLoss(curvature, normalize)
        self.euclidean_loss = EuclideanDistanceLoss(normalize)
    
    def forward(self, syntax_pred, syntax_target, semantic_pred, semantic_target, attention_mask=None):
        """
        Compute the joint syntax-semantics loss.
        
        Args:
            syntax_pred: Predicted syntactic distances (batch_size, seq_len, seq_len)
            syntax_target: Target syntactic distances (batch_size, seq_len, seq_len)
            semantic_pred: Predicted semantic distances or logits
            semantic_target: Target semantic distances or labels
            attention_mask: Attention mask of shape (batch_size, seq_len)
            
        Returns:
            torch.Tensor: Combined loss value
        """
        # Compute syntax loss (hyperbolic)
        syntax_loss = self.hyperbolic_loss(syntax_pred, syntax_target, attention_mask)
        
        # Compute semantics loss (Euclidean)
        # If semantic targets are class labels
        if semantic_target.dim() == 1:
            semantic_loss = F.cross_entropy(semantic_pred, semantic_target)
        # If semantic targets are distances
        else:
            semantic_loss = self.euclidean_loss(semantic_pred, semantic_target, attention_mask)
        
        # Combine losses
        combined_loss = self.syntax_weight * syntax_loss + (1 - self.syntax_weight) * semantic_loss
        
        return combined_loss, syntax_loss, semantic_loss