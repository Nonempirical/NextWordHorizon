"""
Data classes to represent the horizon tree structure.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import numpy as np


@dataclass
class Node:
    """A node in the horizon tree."""
    id: str
    parent_id: Optional[str]
    depth: int
    token_id: int
    token_text: str
    local_prob: float
    cumulative_prob: float
    embedding: Optional[np.ndarray] = None
    proj: Optional[Tuple[float, float, float]] = None


@dataclass
class Edge:
    """An edge between two nodes in the horizon tree."""
    source_id: str
    target_id: str


@dataclass
class DepthStats:
    """Statistics for a specific depth level in the horizon tree."""
    depth: int
    entropy: float
    branching_factor: float
    total_prob_mass: float
    num_nodes: int


@dataclass
class HorizonResult:
    """Complete result from a horizon expansion."""
    nodes: List[Node]
    edges: List[Edge]
    root_node_id: str
    max_depth: int
    depth_stats: Dict[int, DepthStats]

