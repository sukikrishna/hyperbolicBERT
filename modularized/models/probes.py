"""
Probe models for investigating syntactic and semantic information in BERT.

This module implements both Euclidean and hyperbolic probes that can be
trained to extract specific linguistic properties from BERT embeddings.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt as gt
from geoopt.manifolds.stereographic import StereographicExact

from utils.hyperbolic_utils import (
    init_poincare_ball,
    euclidean_to_poincare,
    poincare_distance,
    mobius_addition,
    mobius_matrix_multiplication,
    MobiusLinear
)

class EuclideanDistanceProbe(nn.Module):
    """
    A probe that extracts syntactic tree distances using Euclidean distance.
    
    This probe projects BERT embeddings to a space where Euclidean distances
    correspond to syntactic distances in the dependency parse tree.
    
    Based on:
    Hewitt, J., & Manning, C. D. (2019). A Structural Probe for Finding Syntax in Word Representations.
    """
    def __init__(self, input_dim, probe_rank, device="cuda"):
        super().__init__()
        self.input_dim = input_dim
        self.probe_rank = probe_rank
        self.device = device
        
        # Projection matrix for the structural probe
        self.proj = nn.Parameter(torch.FloatTensor(probe_rank, input_dim))
        nn.init.uniform_(self.proj, -0.05, 0.05)
    
    def forward(self, embeddings):
        """
        Projects token embeddings and computes the squared distances.
        
        Args:
            embeddings: A tensor of token embeddings with shape (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Pairwise distances of shape (batch_size, seq_len, seq_len)
        """
        batch_size, seq_len, _ = embeddings.shape
        
        # Project embeddings to probe space: (batch_size, seq_len, probe_rank)
        projected = torch.matmul(embeddings, self.proj.transpose(0, 1))
        
        # Compute pairwise squared distances
        # (b, i, r) -> (b, i, j, r)
        projected_i = projected.unsqueeze(2).expand(-1, -1, seq_len, -1)
        # (b, j, r) -> (b, i, j, r)
        projected_j = projected.unsqueeze(1).expand(-1, seq_len, -1, -1)
        
        # Squared distance: ||B(h_i - h_j)||^2
        distances = ((projected_i - projected_j) ** 2).sum(dim=-1)
        
        return distances

class HyperbolicDistanceProbe(nn.Module):
    """
    A probe that extracts syntactic tree distances using hyperbolic distance.
    
    This probe projects BERT embeddings to the Poincaré ball where hyperbolic
    distances correspond to syntactic distances in the dependency parse tree.
    """
    def __init__(self, input_dim, probe_rank, curvature=-1.0, device="cuda"):
        super().__init__()
        self.input_dim = input_dim
        self.probe_rank = probe_rank
        self.device = device
        self.ball = StereographicExact(c=curvature)
        
        # Projection from input space to tangent space at origin
        self.proj = nn.Parameter(torch.FloatTensor(probe_rank, input_dim))
        nn.init.uniform_(self.proj, -0.05, 0.05)
        
        # Optional Möbius linear transformation in hyperbolic space
        self.mobius_transform = nn.Parameter(torch.FloatTensor(probe_rank, probe_rank))
        nn.init.uniform_(self.mobius_transform, -0.05, 0.05)
    
    def forward(self, embeddings):
        """
        Projects token embeddings to Poincaré ball and computes hyperbolic distances.
        
        Args:
            embeddings: A tensor of token embeddings with shape (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Pairwise distances of shape (batch_size, seq_len, seq_len)
        """
        batch_size, seq_len, _ = embeddings.shape
        
        # Project embeddings to probe space: (batch_size, seq_len, probe_rank)
        projected = torch.matmul(embeddings, self.proj.transpose(0, 1))
        
        # Map to Poincaré ball
        poincare_embeddings = self.ball.expmap0(projected)
        
        # Apply Möbius transformation
        transformed = self.ball.mobius_matvec(self.mobius_transform, poincare_embeddings)
        
        # Compute pairwise hyperbolic distances
        distances = torch.zeros(batch_size, seq_len, seq_len, device=self.device)
        
        for b in range(batch_size):
            for i in range(seq_len):
                for j in range(i, seq_len):
                    dist = self.ball.dist(
                        transformed[b, i].unsqueeze(0),
                        transformed[b, j].unsqueeze(0)
                    )
                    distances[b, i, j] = dist
                    distances[b, j, i] = dist
        
        return distances

class EuclideanDepthProbe(nn.Module):
    """
    A probe that extracts parse tree depths using Euclidean norm.
    
    Based on:
    Hewitt, J., & Manning, C. D. (2019). A Structural Probe for Finding Syntax in Word Representations.
    """
    def __init__(self, input_dim, probe_rank, device="cuda"):
        super().__init__()
        self.input_dim = input_dim
        self.probe_rank = probe_rank
        self.device = device
        
        # Projection matrix for the depth probe
        self.proj = nn.Parameter(torch.FloatTensor(probe_rank, input_dim))
        nn.init.uniform_(self.proj, -0.05, 0.05)
    
    def forward(self, embeddings):
        """
        Projects token embeddings and computes the squared norm.
        
        Args:
            embeddings: A tensor of token embeddings with shape (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Token depths of shape (batch_size, seq_len)
        """
        # Project embeddings to probe space: (batch_size, seq_len, probe_rank)
        projected = torch.matmul(embeddings, self.proj.transpose(0, 1))
        
        # Compute squared norm: ||Bh_i||^2
        depths = (projected ** 2).sum(dim=-1)
        
        return depths

class HyperbolicDepthProbe(nn.Module):
    """
    A probe that extracts parse tree depths using hyperbolic distance from the origin.
    
    Uses the Poincaré ball model where tree depth corresponds to the distance from the origin.
    """
    def __init__(self, input_dim, probe_rank, curvature=-1.0, device="cuda"):
        super().__init__()
        self.input_dim = input_dim
        self.probe_rank = probe_rank
        self.device = device
        self.ball = StereographicExact(c=curvature)
        
        # Projection from input space to tangent space at origin
        self.proj = nn.Parameter(torch.FloatTensor(probe_rank, input_dim))
        nn.init.uniform_(self.proj, -0.05, 0.05)
        
        # Optional Möbius linear transformation in hyperbolic space
        self.mobius_transform = nn.Parameter(torch.FloatTensor(probe_rank, probe_rank))
        nn.init.uniform_(self.mobius_transform, -0.05, 0.05)
    
    def forward(self, embeddings):
        """
        Projects token embeddings to Poincaré ball and computes distance from origin.
        
        Args:
            embeddings: A tensor of token embeddings with shape (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Token depths of shape (batch_size, seq_len)
        """
        # Project embeddings to probe space: (batch_size, seq_len, probe_rank)
        projected = torch.matmul(embeddings, self.proj.transpose(0, 1))
        
        # Map to Poincaré ball
        poincare_embeddings = self.ball.expmap0(projected)
        
        # Apply Möbius transformation
        transformed = self.ball.mobius_matvec(self.mobius_transform, poincare_embeddings)
        
        # Compute distance from origin
        origin = torch.zeros_like(transformed[:, 0]).unsqueeze(1)
        depths = self.ball.dist(origin, transformed)
        
        return depths.squeeze(1)

class JointEuclideanProbe(nn.Module):
    """
    A joint probe for both syntactic distances and depths in Euclidean space.
    """
    def __init__(self, input_dim, probe_rank, device="cuda"):
        super().__init__()
        self.input_dim = input_dim
        self.probe_rank = probe_rank
        self.device = device
        
        self.distance_probe = EuclideanDistanceProbe(input_dim, probe_rank, device)
        self.depth_probe = EuclideanDepthProbe(input_dim, probe_rank, device)
    
    def forward(self, embeddings, task="both"):
        """
        Forward pass for the joint probe.
        
        Args:
            embeddings: A tensor of token embeddings with shape (batch_size, seq_len, input_dim)
            task: One of ["both", "distance", "depth"]
            
        Returns:
            Distances and/or depths depending on the task
        """
        if task == "both":
            distances = self.distance_probe(embeddings)
            depths = self.depth_probe(embeddings)
            return distances, depths
        elif task == "distance":
            return self.distance_probe(embeddings)
        elif task == "depth":
            return self.depth_probe(embeddings)
        else:
            raise ValueError(f"Unknown task: {task}")

class JointHyperbolicProbe(nn.Module):
    """
    A joint probe for both syntactic distances and depths in hyperbolic space.
    """
    def __init__(self, input_dim, probe_rank, curvature=-1.0, device="cuda"):
        super().__init__()
        self.input_dim = input_dim
        self.probe_rank = probe_rank
        self.device = device
        
        self.distance_probe = HyperbolicDistanceProbe(input_dim, probe_rank, curvature, device)
        self.depth_probe = HyperbolicDepthProbe(input_dim, probe_rank, curvature, device)
    
    def forward(self, embeddings, task="both"):
        """
        Forward pass for the joint probe.
        
        Args:
            embeddings: A tensor of token embeddings with shape (batch_size, seq_len, input_dim)
            task: One of ["both", "distance", "depth"]
            
        Returns:
            Distances and/or depths depending on the task
        """
        if task == "both":
            distances = self.distance_probe(embeddings)
            depths = self.depth_probe(embeddings)
            return distances, depths
        elif task == "distance":
            return self.distance_probe(embeddings)
        elif task == "depth":
            return self.depth_probe(embeddings)
        else:
            raise ValueError(f"Unknown task: {task}")

class SyntaxSemanticsProbe(nn.Module):
    """
    A probe that separately captures syntactic and semantic aspects of representations.
    
    Uses both Euclidean and hyperbolic spaces to capture different types of structure.
    """
    def __init__(self, input_dim, syntax_rank, semantic_rank, curvature=-1.0, device="cuda"):
        super().__init__()
        self.input_dim = input_dim
        self.syntax_rank = syntax_rank
        self.semantic_rank = semantic_rank
        self.device = device
        self.ball = StereographicExact(c=curvature)
        
        # Syntax projection (hyperbolic)
        self.syntax_proj = nn.Parameter(torch.FloatTensor(syntax_rank, input_dim))
        nn.init.uniform_(self.syntax_proj, -0.05, 0.05)
        
        # Semantic projection (Euclidean)
        self.semantic_proj = nn.Parameter(torch.FloatTensor(semantic_rank, input_dim))
        nn.init.uniform_(self.semantic_proj, -0.05, 0.05)
        
        # Transformation in hyperbolic space
        self.mobius_transform = nn.Parameter(torch.FloatTensor(syntax_rank, syntax_rank))
        nn.init.uniform_(self.mobius_transform, -0.05, 0.05)
    
    def project_syntax(self, embeddings):
        """
        Project embeddings to the hyperbolic space for syntactic analysis.
        
        Args:
            embeddings: A tensor of token embeddings with shape (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Syntax representations in Poincaré ball
        """
        projected = torch.matmul(embeddings, self.syntax_proj.transpose(0, 1))
        poincare_embeddings = self.ball.expmap0(projected)
        transformed = self.ball.mobius_matvec(self.mobius_transform, poincare_embeddings)
        return transformed
    
    def project_semantics(self, embeddings):
        """
        Project embeddings to the Euclidean space for semantic analysis.
        
        Args:
            embeddings: A tensor of token embeddings with shape (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Semantic representations in Euclidean space
        """
        return torch.matmul(embeddings, self.semantic_proj.transpose(0, 1))
    
    def syntax_distances(self, embeddings):
        """
        Compute syntax distances in hyperbolic space.
        
        Args:
            embeddings: A tensor of token embeddings with shape (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Pairwise syntactic distances
        """
        batch_size, seq_len, _ = embeddings.shape
        syntax_emb = self.project_syntax(embeddings)
        
        distances = torch.zeros(batch_size, seq_len, seq_len, device=self.device)
        
        for b in range(batch_size):
            for i in range(seq_len):
                for j in range(i, seq_len):
                    dist = self.ball.dist(
                        syntax_emb[b, i].unsqueeze(0),
                        syntax_emb[b, j].unsqueeze(0)
                    )
                    distances[b, i, j] = dist
                    distances[b, j, i] = dist
        
        return distances
    
    def semantic_similarities(self, embeddings):
        """
        Compute semantic similarities in Euclidean space.
        
        Args:
            embeddings: A tensor of token embeddings with shape (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Pairwise semantic similarities
        """
        semantic_emb = self.project_semantics(embeddings)
        
        # Normalize embeddings for cosine similarity
        semantic_emb_norm = F.normalize(semantic_emb, p=2, dim=-1)
        
        # Compute pairwise cosine similarities
        similarities = torch.bmm(semantic_emb_norm, semantic_emb_norm.transpose(1, 2))
        
        return similarities
    
    def forward(self, embeddings, task="both"):
        """
        Forward pass for the syntax-semantics probe.
        
        Args:
            embeddings: A tensor of token embeddings with shape (batch_size, seq_len, input_dim)
            task: One of ["both", "syntax", "semantics"]
            
        Returns:
            Syntactic distances and/or semantic similarities depending on the task
        """
        if task == "both":
            syntax = self.syntax_distances(embeddings)
            semantics = self.semantic_similarities(embeddings)
            return syntax, semantics
        elif task == "syntax":
            return self.syntax_distances(embeddings)
        elif task == "semantics":
            return self.semantic_similarities(embeddings)
        else:
            raise ValueError(f"Unknown task: {task}")