"""
Utilities for hyperbolic geometry operations.

This module contains implementations of operations in the Poincaré ball model
of hyperbolic geometry, including distance computations, projections, and
various hyperbolicity measures.
"""

import torch
import numpy as np
from torch import nn
import geoopt as gt  # Geometric optimization library for PyTorch
from geoopt.manifolds.stereographic import StereographicExact

def init_poincare_ball(curvature=-1.0):
    """
    Initialize a Poincaré ball model with given curvature.
    
    Args:
        curvature (float): Curvature parameter of the hyperbolic space. Default is -1.0.
    
    Returns:
        A Poincaré ball manifold object
    """
    return StereographicExact(c=curvature)

def euclidean_to_poincare(x, ball=None):
    """
    Map Euclidean vectors to the Poincaré ball using exponential map at origin.
    
    Args:
        x (torch.Tensor): Euclidean vectors of shape (..., dim)
        ball (StereographicExact, optional): Poincaré ball manifold. If None, creates one with curvature -1.
    
    Returns:
        torch.Tensor: Vectors mapped to the Poincaré ball
    """
    if ball is None:
        ball = init_poincare_ball()
    return ball.expmap0(x)

def poincare_to_euclidean(x, ball=None):
    """
    Map points from the Poincaré ball to the tangent space at the origin.
    
    Args:
        x (torch.Tensor): Vectors in the Poincaré ball of shape (..., dim)
        ball (StereographicExact, optional): Poincaré ball manifold. If None, creates one with curvature -1.
    
    Returns:
        torch.Tensor: Vectors in the tangent space (Euclidean space)
    """
    if ball is None:
        ball = init_poincare_ball()
    return ball.logmap0(x)

def poincare_distance(x, y, ball=None):
    """
    Compute hyperbolic distance between points in the Poincaré ball.
    
    Args:
        x (torch.Tensor): Points in Poincaré ball of shape (..., dim)
        y (torch.Tensor): Points in Poincaré ball of shape (..., dim)
        ball (StereographicExact, optional): Poincaré ball manifold. If None, creates one with curvature -1.
    
    Returns:
        torch.Tensor: Hyperbolic distances between x and y
    """
    if ball is None:
        ball = init_poincare_ball()
    return ball.dist(x, y)

def mobius_addition(x, y, ball=None):
    """
    Compute the Möbius addition of points in the Poincaré ball.
    
    Args:
        x (torch.Tensor): Points in Poincaré ball of shape (..., dim)
        y (torch.Tensor): Points in Poincaré ball of shape (..., dim)
        ball (StereographicExact, optional): Poincaré ball manifold. If None, creates one with curvature -1.
    
    Returns:
        torch.Tensor: Result of Möbius addition x ⊕ y
    """
    if ball is None:
        ball = init_poincare_ball()
    return ball.mobius_add(x, y)

def mobius_matrix_multiplication(M, x, ball=None):
    """
    Apply matrix multiplication in hyperbolic space (Möbius version).
    
    Args:
        M (torch.Tensor): Matrix of shape (..., m, n)
        x (torch.Tensor): Points in Poincaré ball of shape (..., n)
        ball (StereographicExact, optional): Poincaré ball manifold. If None, creates one with curvature -1.
    
    Returns:
        torch.Tensor: Result of Möbius matrix multiplication
    """
    if ball is None:
        ball = init_poincare_ball()
    return ball.mobius_matvec(M, x)

def compute_delta_hyperbolicity(distances, method='four_point'):
    """
    Compute δ-hyperbolicity of a metric space.
    
    The δ-hyperbolicity measures how close a metric space is to a tree.
    Lower values indicate more tree-like structure.
    
    Args:
        distances (torch.Tensor): Pairwise distance matrix of shape (n, n)
        method (str): Method to compute hyperbolicity. Options: 'four_point', 'thin_triangles'
    
    Returns:
        float: δ-hyperbolicity value
    """
    if method == 'four_point':
        # Implement Gromov's four-point condition
        n = distances.shape[0]
        max_delta = 0.0
        
        # Sample points if the space is too large
        if n > 50:
            indices = np.random.choice(n, 50, replace=False)
            distances = distances[indices][:, indices]
            n = 50
        
        # Iterate through all 4-tuples of points
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        d_ij = distances[i, j]
                        d_kl = distances[k, l]
                        d_ik = distances[i, k]
                        d_jl = distances[j, l]
                        d_il = distances[i, l]
                        d_jk = distances[j, k]
                        
                        # Sort the three sums
                        sums = sorted([d_ij + d_kl, d_ik + d_jl, d_il + d_jk])
                        # Delta is half the difference between the largest two sums
                        delta = (sums[2] - sums[1]) / 2
                        max_delta = max(max_delta, delta)
        
        return max_delta
    
    elif method == 'thin_triangles':
        # Implement thin triangles definition
        n = distances.shape[0]
        max_delta = 0.0
        
        # Sample points if the space is too large
        if n > 50:
            indices = np.random.choice(n, 50, replace=False)
            distances = distances[indices][:, indices]
            n = 50
        
        # Iterate through all triangles
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    sides = [distances[i, j], distances[j, k], distances[i, k]]
                    sides.sort()
                    # Compute insize (distance from a vertex to the opposite side)
                    # For simplicity, we approximate using the Euclidean formula
                    # Real implementation would use hyperbolic geometry formulas
                    s = sum(sides) / 2
                    area = np.sqrt(s * (s - sides[0]) * (s - sides[1]) * (s - sides[2]))
                    insize = 2 * area / sides[2]
                    max_delta = max(max_delta, insize)
        
        return max_delta
    
    else:
        raise ValueError(f"Unknown method: {method}")

def compute_tree_likeness(embeddings, ball=None):
    """
    Compute a measure of how tree-like a set of embeddings is in hyperbolic space.
    
    Args:
        embeddings (torch.Tensor): Embeddings in the Poincaré ball of shape (n, dim)
        ball (StereographicExact, optional): Poincaré ball manifold. If None, creates one with curvature -1.
    
    Returns:
        float: Tree-likeness score (higher means more tree-like)
    """
    if ball is None:
        ball = init_poincare_ball()
    
    # Compute pairwise distances
    n = embeddings.shape[0]
    distances = torch.zeros((n, n), device=embeddings.device)
    for i in range(n):
        for j in range(i+1, n):
            distances[i, j] = ball.dist(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0))
            distances[j, i] = distances[i, j]
    
    # Compute δ-hyperbolicity
    delta = compute_delta_hyperbolicity(distances)
    
    # Smaller delta means more tree-like, so return inverse
    return 1.0 / (delta + 1e-5)

def estimate_curvature(embeddings, ball=None):
    """
    Estimate the curvature of the space that best fits the embeddings.
    
    Args:
        embeddings (torch.Tensor): Embeddings of shape (n, dim)
        ball (StereographicExact, optional): Initial ball manifold for optimization
    
    Returns:
        float: Estimated curvature value
    """
    if ball is None:
        ball = init_poincare_ball()
    
    # Simple method: try different curvatures and find the one that gives 
    # the best fit (lowest distortion)
    curvatures = torch.tensor([-0.1, -0.5, -1.0, -2.0, -5.0])
    distortions = []
    
    n = embeddings.shape[0]
    
    # Compute Euclidean distances
    euclidean_distances = torch.zeros((n, n), device=embeddings.device)
    for i in range(n):
        for j in range(i+1, n):
            euclidean_distances[i, j] = torch.norm(embeddings[i] - embeddings[j])
            euclidean_distances[j, i] = euclidean_distances[i, j]
    
    # Try different curvatures
    for c in curvatures:
        test_ball = StereographicExact(c=c)
        poincare_emb = euclidean_to_poincare(embeddings, ball=test_ball)
        
        # Compute hyperbolic distances
        hyperbolic_distances = torch.zeros((n, n), device=embeddings.device)
        for i in range(n):
            for j in range(i+1, n):
                hyperbolic_distances[i, j] = test_ball.dist(poincare_emb[i].unsqueeze(0), poincare_emb[j].unsqueeze(0))
                hyperbolic_distances[j, i] = hyperbolic_distances[i, j]
        
        # Compute distortion (normalized difference between distributions)
        euclidean_distances_flat = euclidean_distances[torch.triu_indices(n, n, 1)]
        hyperbolic_distances_flat = hyperbolic_distances[torch.triu_indices(n, n, 1)]
        
        # Normalize distances
        euclidean_distances_flat = euclidean_distances_flat / euclidean_distances_flat.max()
        hyperbolic_distances_flat = hyperbolic_distances_flat / hyperbolic_distances_flat.max()
        
        # Compute distortion as L2 norm of differences
        distortion = torch.norm(euclidean_distances_flat - hyperbolic_distances_flat)
        distortions.append(distortion.item())
    
    # Return the curvature with minimum distortion
    return curvatures[np.argmin(distortions)].item()

class MobiusLinear(nn.Module):
    """
    Linear layer in the Poincaré ball model of hyperbolic space.
    
    Performs Möbius matrix multiplication followed by Möbius addition.
    """
    def __init__(self, in_features, out_features, ball=None, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ball = ball if ball is not None else init_poincare_ball()
        
        # Weight in the tangent space at the origin
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            # Bias in the Poincaré ball
            self.bias = gt.ManifoldParameter(
                torch.Tensor(out_features), manifold=self.ball
            )
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        import math
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            # Initialize bias close to the origin
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        """
        Apply Möbius linear transformation.
        
        Args:
            x (torch.Tensor): Points in the Poincaré ball of shape (..., in_features)
            
        Returns:
            torch.Tensor: Transformed points in the Poincaré ball of shape (..., out_features)
        """
        res = mobius_matrix_multiplication(self.weight, x, ball=self.ball)
        if self.bias is not None:
            res = mobius_addition(res, self.bias, ball=self.ball)
        return res