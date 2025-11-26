"""
Metrics for analyzing horizon trees.

This module contains functions to calculate various metrics
that describe properties of an expanded horizon tree:

- Entropy: Measure of uncertainty/variation in predictions
- Branching factor: Average number of children per node
- Path diversity: Diversity in different paths through the tree
- Probability concentration: How concentrated the probability is
"""

from typing import List
from .models import Node, HorizonResult
import numpy as np


def calculate_entropy(probabilities: List[float]) -> float:
    """
    Calculates Shannon entropy for a list of probabilities.
    
    Formula: H = -Σ p_i * log2(p_i) for all p_i > 0
    
    Args:
        probabilities: List of probabilities (should sum to ~1.0)
        
    Returns:
        Entropy value (bits)
    """
    probs = np.array(probabilities)
    # Filter out zeros (log(0) is undefined)
    probs = probs[probs > 0]
    
    if len(probs) == 0:
        return 0.0
    
    # Calculate entropy: H = -Σ p * log2(p)
    entropy = -np.sum(probs * np.log2(probs))
    return float(entropy)


def calculate_entropy_per_depth(probabilities: List[float]) -> float:
    """
    Calculates Shannon entropy for a depth level.
    
    Same as calculate_entropy but with clearer name for depth-specific usage.
    
    Args:
        probabilities: List of probabilities for nodes at a depth level
        
    Returns:
        Entropy value (bits)
    """
    return calculate_entropy(probabilities)


def calculate_branching_factor_per_depth(probabilities: List[float]) -> float:
    """
    Calculates branching factor for a depth level.
    
    Branching factor = exp(entropy) gives the effective number of branches
    based on the entropy in the probability distribution.
    
    Args:
        probabilities: List of probabilities for nodes at a depth level
        
    Returns:
        Branching factor (effective number of branches)
    """
    entropy = calculate_entropy_per_depth(probabilities)
    # BF = exp(H) where H is entropy
    branching_factor = np.exp(entropy)
    return float(branching_factor)


def calculate_branching_factor(result: HorizonResult) -> float:
    """
    Calculates average branching factor (number of children per node).
    
    Args:
        result: HorizonResult with expanded tree
        
    Returns:
        Average number of children per node
    """
    if len(result.nodes) <= 1:  # Only root
        return 0.0
    
    # Count number of edges (each edge is a parent->child link)
    num_edges = len(result.edges)
    
    # Number of nodes that have children (all except leaf nodes)
    parent_nodes = set(edge.source_id for edge in result.edges)
    num_parents = len(parent_nodes)
    
    if num_parents == 0:
        return 0.0
    
    # Average number of children per parent
    avg_children = num_edges / num_parents
    return float(avg_children)


def calculate_path_diversity(result: HorizonResult) -> float:
    """
    Calculates diversity in different paths through the tree.
    
    Args:
        result: HorizonResult with expanded tree
        
    Returns:
        Measure of path diversity (higher = more diversity)
    """
    # Simple implementation: number of unique paths to leaf nodes
    # A path is unique if it has a unique sequence of cumulative_probs
    
    # Find leaf nodes (nodes without children)
    all_node_ids = {node.id for node in result.nodes}
    parent_ids = {edge.source_id for edge in result.edges}
    leaf_ids = all_node_ids - parent_ids
    
    # Count number of leaf nodes (each leaf represents a unique path)
    num_paths = len(leaf_ids)
    
    return float(num_paths)

