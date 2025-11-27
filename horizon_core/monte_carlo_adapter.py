"""
Adapter functions to connect ModelAdapter with Monte Carlo sampling.

This module provides adapter functions that wrap ModelAdapter methods
to work with the Monte Carlo sampler's function signatures.
"""

from typing import Callable, List
import numpy as np
from .adapters import ModelAdapter

# Import from src/sampling (at project root level)
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.sampling.monte_carlo_sampler import Prefix


def create_get_logits_fn(adapter: ModelAdapter) -> Callable[[Prefix], np.ndarray]:
    """
    Creates a get_logits_fn that works with ModelAdapter.
    
    Args:
        adapter: ModelAdapter instance
        
    Returns:
        Function that takes Prefix (tuple of token IDs) and returns logits for last token
    """
    def get_logits_fn(prefix: Prefix) -> np.ndarray:
        """
        Get logits for the last token in a prefix.
        
        Args:
            prefix: Tuple of token IDs
            
        Returns:
            Logits array for the last token [vocab_size]
        """
        token_ids = list(prefix)
        if len(token_ids) == 0:
            raise ValueError("Prefix cannot be empty")
        
        logits = adapter.get_logits(token_ids)
        
        # Convert to numpy if needed
        if hasattr(logits, 'cpu'):
            logits = logits.cpu().numpy()
        elif not isinstance(logits, np.ndarray):
            logits = np.array(logits)
        
        # Return last token's logits
        # get_logits returns [seq_len, vocab_size], we want [-1, :]
        if len(logits.shape) == 2:
            return logits[-1]
        elif len(logits.shape) == 1:
            # Already just vocab_size, return as is
            return logits
        else:
            raise ValueError(f"Unexpected logits shape: {logits.shape}")
    
    return get_logits_fn


def create_embed_fn(adapter: ModelAdapter) -> Callable[[Prefix], np.ndarray]:
    """
    Creates an embed_fn that works with ModelAdapter.
    
    Uses the sequence embedding method, which averages token embeddings.
    For better results, you might want to use hidden states from the model.
    
    Args:
        adapter: ModelAdapter instance
        
    Returns:
        Function that takes Prefix and returns embedding vector
    """
    def embed_fn(prefix: Prefix) -> np.ndarray:
        """
        Get embedding for a prefix sequence.
        
        Args:
            prefix: Tuple of token IDs
            
        Returns:
            Embedding vector [embed_dim]
        """
        token_ids = list(prefix)
        if len(token_ids) == 0:
            raise ValueError("Prefix cannot be empty")
        
        embedding = adapter.get_sequence_embedding(token_ids)
        
        # Convert to numpy if needed
        if hasattr(embedding, 'cpu'):
            embedding = embedding.cpu().numpy()
        elif not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
        
        return embedding
    
    return embed_fn


def create_cluster_fn(num_clusters: int, random_state: int = 42) -> Callable[[np.ndarray, int], np.ndarray]:
    """
    Creates a cluster_fn using sklearn KMeans.
    
    Args:
        num_clusters: Default number of clusters (can be overridden)
        random_state: Random state for reproducibility
        
    Returns:
        Function that takes embeddings and num_clusters, returns cluster labels
    """
    from sklearn.cluster import KMeans
    
    def cluster_fn(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        Cluster embeddings using KMeans.
        
        Args:
            embeddings: Array of shape [n_samples, embed_dim]
            n_clusters: Number of clusters (uses num_clusters if not provided)
            
        Returns:
            Cluster labels array [n_samples]
        """
        if n_clusters is None:
            n_clusters = num_clusters
        
        if len(embeddings) < n_clusters:
            # Not enough samples, assign all to cluster 0
            return np.zeros(len(embeddings), dtype=np.int32)
        
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
        labels = kmeans.fit_predict(embeddings)
        return labels.astype(np.int32)
    
    return cluster_fn

