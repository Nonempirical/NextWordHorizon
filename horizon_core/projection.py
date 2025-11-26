"""
Dimensionality reduction and projection for visualization.
"""

import numpy as np
from typing import List, Tuple
from sklearn.decomposition import PCA
from .config import PCA_COMPONENTS, UMAP_N_NEIGHBORS, UMAP_MIN_DIST
from .models import HorizonResult, Node

# Try to import UMAP, but handle if it's not installed
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    UMAP = None


def project_horizon_to_3d(
    horizon: HorizonResult,
    pca_components: int = PCA_COMPONENTS,
    n_neighbors: int = UMAP_N_NEIGHBORS,
    min_dist: float = UMAP_MIN_DIST
) -> HorizonResult:
    """Projects horizon tree embeddings to 3D coordinates for visualization."""
    embeddings_list = []
    node_indices = []
    
    for i, node in enumerate(horizon.nodes):
        if node.depth > 0 and node.embedding is not None:
            embeddings_list.append(node.embedding)
            node_indices.append(i)
    
    if len(embeddings_list) == 0:
        return horizon
    
    if len(embeddings_list) < 2:
        for idx in node_indices:
            horizon.nodes[idx].proj = (0.0, 0.0, 0.0)
        return horizon
    
    X = np.array(embeddings_list)
    n_samples, n_features = X.shape
    actual_pca_components = min(pca_components, n_samples - 1, n_features)
    
    if actual_pca_components < n_features:
        pca = PCA(n_components=actual_pca_components)
        X_pca = pca.fit_transform(X)
    else:
        X_pca = X
    
    if not UMAP_AVAILABLE:
        raise ImportError(
            "umap-learn is not installed. Please install it with: pip install umap-learn\n"
            "Note: umap-learn requires numba, which may not support Python 3.14 yet.\n"
            "Consider using Python 3.11 or 3.13 instead."
        )
    
    actual_n_neighbors = min(n_neighbors, len(X_pca) - 1)
    
    umap = UMAP(
        n_components=3,
        n_neighbors=actual_n_neighbors,
        min_dist=min_dist,
        random_state=42
    )
    X_3d = umap.fit_transform(X_pca)
    
    for idx, coords in zip(node_indices, X_3d):
        horizon.nodes[idx].proj = (float(coords[0]), float(coords[1]), float(coords[2]))
    
    return horizon


