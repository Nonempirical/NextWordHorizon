"""
Probabilistic Horizon Expansion - Main logic for tree expansion.

This module implements the core algorithm for building a probabilistic
horizon tree. The algorithm expands recursively from a start node by
using a model adapter to generate next-word candidates and build
a tree-structured network of possible continuations.

The tree represents different paths through probability space and can
be used to visualize and analyze the model's predictions.
"""

from typing import List, Optional, Dict, Any
from .models import Node, Edge, HorizonResult, DepthStats
from .adapters import ModelAdapter
from .config import K, DEPTH
from .metrics import calculate_entropy_per_depth, calculate_branching_factor_per_depth
import numpy as np

# Horizon limiting parameters
MAX_NODES = 150_000             # Global hård gräns för antal noder i ett HorizonResult
MASS_CUTOFF = 0.95              # Hur mycket lokal sannolikhetsmassa vi vill täcka per nod
MAX_CHILDREN_PER_NODE = 12      # Max antal barn per nod (efter diversitet)
MIN_DIVERSITY_COSINE = 0.15     # Minsta cosinus-avstånd mellan barn (1 - cos_sim)


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


def select_children_for_node(
    token_ids: List[int],
    probs: List[float],
    embeddings: List[np.ndarray],
    mass_cutoff: float = MASS_CUTOFF,
    max_children: int = MAX_CHILDREN_PER_NODE,
    min_diversity_cosine: float = MIN_DIVERSITY_COSINE,
) -> List[int]:
    """
    Väljer en representativ, sannolikhetsviktad och diversifierad uppsättning barn för en nod.
    
    Antaganden:
    - token_ids, probs, embeddings har samma längd
    - probs är redan normaliserade (softmax)
    - tokens är redan sorterade efter probs (störst först)
    
    Args:
        token_ids: Lista med token IDs (sorterade efter sannolikhet, högst först)
        probs: Lista med sannolikheter (redan normaliserade softmax, sorterade)
        embeddings: Lista med embeddings för varje token (samma ordning)
        mass_cutoff: Minsta sannolikhetsmassa att täcka (0-1)
        max_children: Max antal barn att välja
        min_diversity_cosine: Minsta cosinus-avstånd (1 - cos_sim) mellan valda barn
        
    Returns:
        En lista med index i token_ids/probs/embeddings för de barn som ska skapas.
    """
    if len(token_ids) == 0:
        return []
    
    # Säkerställ att de är sorterade efter sannolikhet (högst först)
    # Om de redan är sorterade, argsort ger [0, 1, 2, ...]
    sorted_indices = np.argsort(probs)[::-1]
    
    selected_indices = []
    cumulative_mass = 0.0
    selected_embeddings = []
    
    for idx in sorted_indices:
        # Kolla om vi redan har tillräckligt med massa
        if cumulative_mass >= mass_cutoff:
            break
        
        # Kolla om vi redan har max antal barn
        if len(selected_indices) >= max_children:
            break
        
        # Hämta embedding för detta token
        emb = embeddings[idx]
        
        # Konvertera till numpy om det behövs
        if hasattr(emb, 'cpu'):
            emb = emb.cpu().numpy()
        elif not isinstance(emb, np.ndarray):
            emb = np.array(emb)
        
        # Normalisera embedding för cosinus-avstånd
        emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
        
        # Kolla diversitet mot alla redan valda
        is_diverse = True
        for selected_emb in selected_embeddings:
            # Beräkna cosinus-likhet
            cos_sim = np.dot(emb_norm, selected_emb)
            # Konvertera till avstånd: dist = 1 - cos_sim
            distance = 1.0 - cos_sim
            
            if distance < min_diversity_cosine:
                is_diverse = False
                break
        
        # Om den är tillräckligt olik, lägg till
        if is_diverse:
            selected_indices.append(int(idx))
            selected_embeddings.append(emb_norm)
            cumulative_mass += probs[idx]
    
    return selected_indices


def expand_horizon(
    adapter: ModelAdapter,
    prompt: str,
    top_k: int = K,
    max_depth: int = DEPTH,
    cancel_event: Optional[Any] = None,  # threading.Event or None
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
    truncated = False  # Track if we hit MAX_NODES limit
    node_count = 0  # Antal icke-root-noder som faktiskt skapas
    stop_expansion = False
    
    for depth in range(1, max_depth + 1):
        if stop_expansion:
            break
        
        if len(current_frontier) == 0:
            break
        
        next_frontier: List[Node] = []
        
        for node in current_frontier:
            if stop_expansion:
                break
            
            # Check if cancelled
            if cancel_event is not None and cancel_event.is_set():
                stop_expansion = True
                truncated = True
                break
            
            # BEFORE calling the model: kolla budget
            if node_count >= MAX_NODES:
                stop_expansion = True
                truncated = True
                break
            
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
            
            # Hämta top-k tokens (mer än MAX_CHILDREN_PER_NODE för att ha utrymme för diversitet)
            top_k_indices = np.argsort(probs)[-top_k:][::-1]
            top_k_probs = probs[top_k_indices]
            
            top_k_sum = top_k_probs.sum()
            if top_k_sum == 0:
                continue
            top_k_probs_norm = top_k_probs / top_k_sum
            
            # Hämta embeddings för alla top-k tokens
            # Vi behöver matcha token_ids, probs och embeddings
            top_k_embeddings = []
            top_k_token_ids = []
            top_k_probs_filtered = []
            
            for i, idx in enumerate(top_k_indices):
                try:
                    emb = adapter.get_token_embedding(int(idx))
                    # Konvertera till numpy
                    if hasattr(emb, 'cpu'):
                        emb = emb.cpu().numpy()
                    elif not isinstance(emb, np.ndarray):
                        emb = np.array(emb)
                    top_k_embeddings.append(emb)
                    top_k_token_ids.append(int(idx))
                    top_k_probs_filtered.append(top_k_probs_norm[i])
                except Exception:
                    # Om embedding misslyckas, hoppa över denna token
                    continue
            
            if len(top_k_token_ids) == 0:
                continue
            
            # Normalisera probs igen efter att vi filtrerat bort tokens utan embeddings
            if len(top_k_probs_filtered) > 0:
                prob_sum = sum(top_k_probs_filtered)
                if prob_sum > 0:
                    top_k_probs_filtered = [p / prob_sum for p in top_k_probs_filtered]
            
            # Välj barn baserat på sannolikhet, massa och diversitet
            child_indices = select_children_for_node(
                token_ids=top_k_token_ids,
                probs=top_k_probs_filtered,
                embeddings=top_k_embeddings,
                mass_cutoff=MASS_CUTOFF,
                max_children=MAX_CHILDREN_PER_NODE,
                min_diversity_cosine=MIN_DIVERSITY_COSINE,
            )
            
            # Skapa barnnoder endast för de valda indexen
            for child_idx in child_indices:
                # Check if cancelled
                if cancel_event is not None and cancel_event.is_set():
                    stop_expansion = True
                    truncated = True
                    break
                
                # Kolla budget innan vi skapar noden
                if node_count >= MAX_NODES:
                    stop_expansion = True
                    truncated = True
                    break
                
                token_id = top_k_token_ids[child_idx]
                local_prob = top_k_probs_filtered[child_idx]
                
                # Calculate cumulative_prob
                cumulative_prob = node.cumulative_prob * float(local_prob)
                
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
                node_count += 1  # Öka räknaren (räknar icke-root-noder)
                
                # Create Edge from parent to child
                edge = Edge(
                    source_id=node.id,
                    target_id=new_node.id
                )
                edges.append(edge)
                
                # Kolla igen efter att ha lagt till noden
                if node_count >= MAX_NODES:
                    stop_expansion = True
                    truncated = True
                    break
            
            if stop_expansion:
                break
        
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
        depth_stats=depth_stats,
        truncated=truncated
    )

