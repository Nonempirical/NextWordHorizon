"""
Probabilistic Horizon Expansion - Main logic for tree expansion.

This module implements the core algorithm for building a probabilistic
horizon tree. The algorithm expands recursively from a start node by
using a model adapter to generate next-word candidates and build
a tree-structured network of possible continuations.

The tree represents different paths through probability space and can
be used to visualize and analyze the model's predictions.
"""

from typing import List, Optional, Dict
from .models import Node, Edge, HorizonResult, DepthStats
from .adapters import ModelAdapter
from .config import K, DEPTH
from .metrics import calculate_entropy_per_depth, calculate_branching_factor_per_depth
import numpy as np


def _build_sequence_from_node(node: Node, nodes_dict: Dict[str, Node], prompt_ids: List[int]) -> List[int]:
    """Builds a sequence of token IDs from root to given node."""
    sequence = prompt_ids.copy()
    current = node
    
    path_tokens = []
    while current.parent_id is not None:
        if current.parent_id not in nodes_dict:
            break
        path_tokens.append(current.token_id)
        current = nodes_dict[current.parent_id]
    
    path_tokens.reverse()
    sequence.extend(path_tokens)
    
    return sequence


def expand_horizon(
    adapter: ModelAdapter,
    prompt: str,
    top_k: int = K,
    max_depth: int = DEPTH,
) -> HorizonResult:
    """Expands a probabilistic horizon tree from a given prompt."""
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if max_depth < 1:
        raise ValueError("max_depth must be at least 1")
    
    prompt_ids = adapter.encode(prompt)
    if len(prompt_ids) == 0:
        raise ValueError("Encoded prompt is empty")
    
    # Create root node (virtual, depth 0, no token)
    root_node = Node(
        id="root",
        parent_id=None,
        depth=0,
        token_id=-1,  # No token for root
        token_text="<ROOT>",
        local_prob=1.0,
        cumulative_prob=1.0
    )
    
    nodes: List[Node] = [root_node]
    edges: List[Edge] = []
    nodes_dict: Dict[str, Node] = {root_node.id: root_node}
    
    current_frontier: List[Node] = [root_node]
    node_counter = 0  # For generating unique node IDs
    
    for depth in range(1, max_depth + 1):
        if len(current_frontier) == 0:
            break
        
        next_frontier: List[Node] = []
        
        for node in current_frontier:
            # Build sequence_ids = prompt + all tokens along path
            sequence_ids = _build_sequence_from_node(node, nodes_dict, prompt_ids)
            
            if len(sequence_ids) == 0:
                continue
            
            try:
                logits = adapter.get_logits(sequence_ids)
            except Exception as e:
                continue
            
            if isinstance(logits, np.ndarray):
                if logits.shape[0] == 0:
                    continue
                last_logits = logits[-1]
            else:
                import torch
                if isinstance(logits, torch.Tensor):
                    if logits.shape[0] == 0:
                        continue
                    last_logits = logits[-1].cpu().numpy()
                else:
                    if len(logits) == 0:
                        continue
                    last_logits = np.array(logits[-1])
            
            exp_logits = np.exp(last_logits - np.max(last_logits))
            exp_sum = exp_logits.sum()
            if exp_sum == 0:
                continue
            probs = exp_logits / exp_sum
            
            top_k_indices = np.argsort(probs)[-top_k:][::-1]
            top_k_probs = probs[top_k_indices]
            
            top_k_sum = top_k_probs.sum()
            if top_k_sum == 0:
                continue
            top_k_probs_norm = top_k_probs / top_k_sum
            
            # For each top-k token
            for i, (token_id, local_prob) in enumerate(zip(top_k_indices, top_k_probs_norm)):
                # Calculate cumulative_prob
                cumulative_prob = node.cumulative_prob * local_prob
                
                # Generate unique node ID
                node_counter += 1
                node_id = f"d{depth}_t{node_counter}"
                
                # Decode token
                token_text = adapter.decode([int(token_id)])
                
                # Create new Node
                new_node = Node(
                    id=node_id,
                    parent_id=node.id,
                    depth=depth,
                    token_id=int(token_id),
                    token_text=token_text,
                    local_prob=float(local_prob),
                    cumulative_prob=float(cumulative_prob)
                )
                
                # Add to nodes + next_frontier + create Edge
                nodes.append(new_node)
                nodes_dict[new_node.id] = new_node
                next_frontier.append(new_node)
                
                # Create Edge from parent to child
                edge = Edge(
                    source_id=node.id,
                    target_id=new_node.id
                )
                edges.append(edge)
        
        # Update current_frontier for next iteration
        current_frontier = next_frontier
    
    # Calculate per-depth entropy and branching factor
    depth_stats = {}
    for d in range(1, max_depth + 1):
        # Get all nodes at this depth
        depth_nodes = [n for n in nodes if n.depth == d]
        
        if len(depth_nodes) == 0:
            continue
        
        local_probs = np.array([n.local_prob for n in depth_nodes])
        prob_sum = local_probs.sum()
        if prob_sum == 0:
            continue
        local_probs_norm = local_probs / prob_sum
        
        # Calculate entropy and branching factor
        entropy = calculate_entropy_per_depth(local_probs_norm.tolist())
        bf = calculate_branching_factor_per_depth(local_probs_norm.tolist())
        
        # Total probability mass
        total_prob_mass = float(local_probs.sum())
        
        depth_stats[d] = DepthStats(
            depth=d,
            entropy=entropy,
            branching_factor=bf,
            total_prob_mass=total_prob_mass,
            num_nodes=len(depth_nodes)
        )
    
    # Get embeddings for each node (except root)
    for node in nodes:
        if node.depth > 0:  # Skip root
            try:
                embedding = adapter.get_token_embedding(node.token_id)
                # Convert to numpy if it's a torch tensor
                if hasattr(embedding, 'cpu'):
                    embedding = embedding.cpu().numpy()
                elif not isinstance(embedding, np.ndarray):
                    embedding = np.array(embedding)
                node.embedding = embedding
            except Exception as e:
                # If embedding retrieval fails, leave it as None
                # (can happen if adapter doesn't support get_token_embedding)
                pass
    
    return HorizonResult(
        nodes=nodes,
        edges=edges,
        root_node_id=root_node.id,
        max_depth=max_depth,
        depth_stats=depth_stats
    )

