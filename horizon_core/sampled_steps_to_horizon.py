"""
Convert List[SampledStep] to HorizonResult.

This module converts the output from Monte Carlo sampling into
the existing HorizonResult format for visualization.
"""

from typing import List, Dict, Set, Tuple
from collections import defaultdict
import numpy as np
import math

from .models import Node, Edge, HorizonResult, DepthStats

# Import from src/sampling (at project root level)
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.sampling.monte_carlo_sampler import SampledStep, Prefix


def sampled_steps_to_horizon(
    steps: List[SampledStep],
    prompt_ids: Prefix,
    adapter,  # ModelAdapter
    root_node_id: str = "root"
) -> HorizonResult:
    """
    Convert a list of SampledStep records into a HorizonResult.
    
    Each unique prefix becomes a node.
    Each (prefix -> next_token) becomes an edge.
    We aggregate probabilities and frequencies from multiple rollouts.
    
    Args:
        steps: List of SampledStep records from Monte Carlo sampling
        prompt_ids: The original prompt token IDs (for root node)
        adapter: ModelAdapter for decoding tokens
        root_node_id: ID for the root node
        
    Returns:
        HorizonResult with nodes and edges built from sampled steps
    """
    # Build mapping: prefix -> node_id
    prefix_to_node_id: Dict[Prefix, str] = {}
    node_id_counter = 0
    
    # Track edges: (source_prefix, target_prefix) -> (count, total_prob, avg_prob)
    edge_data: Dict[Tuple[Prefix, Prefix], Tuple[int, float, float]] = {}
    
    # Track node data: prefix -> (depth, token_id, token_text, probs, cumulative_probs)
    node_data: Dict[Prefix, Dict] = {}
    
    # Add root node
    root_prefix = prompt_ids
    prefix_to_node_id[root_prefix] = root_node_id
    node_data[root_prefix] = {
        'depth': 0,
        'token_id': -1,
        'token_text': '<ROOT>',
        'probs': [],
        'cumulative_probs': [1.0]
    }
    
    # Process all steps
    for step in steps:
        source_prefix = step.prefix
        target_prefix = step.prefix + (step.next_token,)
        
        # Ensure source node exists
        if source_prefix not in prefix_to_node_id:
            node_id_counter += 1
            node_id = f"d{step.depth}_t{node_id_counter}"
            prefix_to_node_id[source_prefix] = node_id
            
            # Decode token for source (last token in prefix)
            if len(source_prefix) > len(prompt_ids):
                last_token_id = source_prefix[-1]
                token_text = adapter.decode([last_token_id])
            else:
                token_text = '<ROOT>'
            
            node_data[source_prefix] = {
                'depth': step.depth,
                'token_id': source_prefix[-1] if len(source_prefix) > len(prompt_ids) else -1,
                'token_text': token_text,
                'probs': [],
                'cumulative_probs': []
            }
        
        # Ensure target node exists
        if target_prefix not in prefix_to_node_id:
            node_id_counter += 1
            node_id = f"d{step.depth + 1}_t{node_id_counter}"
            prefix_to_node_id[target_prefix] = node_id
            
            token_text = adapter.decode([step.next_token])
            node_data[target_prefix] = {
                'depth': step.depth + 1,
                'token_id': step.next_token,
                'token_text': token_text,
                'probs': [],
                'cumulative_probs': []
            }
        
        # Record edge
        edge_key = (source_prefix, target_prefix)
        if edge_key not in edge_data:
            edge_data[edge_key] = (0, 0.0, 0.0)
        
        count, total_prob, _ = edge_data[edge_key]
        edge_data[edge_key] = (count + 1, total_prob + step.prob, 0.0)
        
        # Record node probabilities
        node_data[target_prefix]['probs'].append(step.prob)
        node_data[target_prefix]['cumulative_probs'].append(math.exp(-step.cum_logprob) if step.cum_logprob > 0 else step.prob)
    
    # Calculate average probabilities for edges
    for edge_key in edge_data:
        count, total_prob, _ = edge_data[edge_key]
        edge_data[edge_key] = (count, total_prob, total_prob / count if count > 0 else 0.0)
    
    # Build nodes
    nodes: List[Node] = []
    for prefix, node_id in prefix_to_node_id.items():
        data = node_data[prefix]
        
        # Calculate average local prob
        probs = data['probs']
        local_prob = np.mean(probs) if probs else 0.0
        
        # Calculate average cumulative prob
        cum_probs = data['cumulative_probs']
        cumulative_prob = np.mean(cum_probs) if cum_probs else 0.0
        
        # Find parent
        parent_id = None
        if len(prefix) > len(prompt_ids):
            parent_prefix = prefix[:-1]
            parent_id = prefix_to_node_id.get(parent_prefix)
        
        node = Node(
            id=node_id,
            parent_id=parent_id,
            depth=data['depth'],
            token_id=data['token_id'],
            token_text=data['token_text'],
            local_prob=float(local_prob),
            cumulative_prob=float(cumulative_prob)
        )
        nodes.append(node)
    
    # Build edges
    edges: List[Edge] = []
    for (source_prefix, target_prefix), (count, total_prob, avg_prob) in edge_data.items():
        source_id = prefix_to_node_id[source_prefix]
        target_id = prefix_to_node_id[target_prefix]
        
        edge = Edge(
            source_id=source_id,
            target_id=target_id
        )
        edges.append(edge)
    
    # Calculate depth stats
    depth_stats: Dict[int, DepthStats] = {}
    nodes_by_depth: Dict[int, List[Node]] = defaultdict(list)
    for node in nodes:
        nodes_by_depth[node.depth].append(node)
    
    max_depth = max([node.depth for node in nodes]) if nodes else 0
    
    for depth in range(max_depth + 1):
        depth_nodes = nodes_by_depth.get(depth, [])
        if not depth_nodes:
            continue
        
        # Calculate entropy from local probabilities
        probs = [n.local_prob for n in depth_nodes if n.local_prob > 0]
        if probs:
            # Normalize
            prob_sum = sum(probs)
            if prob_sum > 0:
                probs = [p / prob_sum for p in probs]
                entropy = -sum(p * np.log(p + 1e-12) for p in probs)
            else:
                entropy = 0.0
        else:
            entropy = 0.0
        
        # Branching factor: average number of children per node
        children_count = defaultdict(int)
        for edge in edges:
            source_node = next((n for n in nodes if n.id == edge.source_id), None)
            if source_node and source_node.depth == depth:
                children_count[edge.source_id] += 1
        
        if depth_nodes:
            branching_factor = sum(children_count.values()) / len(depth_nodes)
        else:
            branching_factor = 0.0
        
        # Total probability mass
        total_prob_mass = sum(n.cumulative_prob for n in depth_nodes)
        
        depth_stats[depth] = DepthStats(
            depth=depth,
            entropy=float(entropy),
            branching_factor=float(branching_factor),
            total_prob_mass=float(total_prob_mass),
            num_nodes=len(depth_nodes)
        )
    
    return HorizonResult(
        nodes=nodes,
        edges=edges,
        root_node_id=root_node_id,
        max_depth=max_depth,
        depth_stats=depth_stats,
        truncated=False  # Monte Carlo sampling respects max_nodes in config
    )

