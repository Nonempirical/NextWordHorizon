"""
FastAPI server for Horizon Expansion API.

This module defines REST API endpoints for:
- Expanding horizon trees from a given context
- Getting projections and visualizations
- Calculating metrics for expanded trees

The API can run locally or be deployed as a separate service.
It is neutral enough to run both locally and in Colab.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Literal
import uvicorn
import os
from pathlib import Path

from horizon_core.horizon import expand_horizon
from horizon_core.projection import project_horizon_to_3d
from horizon_core.adapters import LocalHFAdapter, RemoteHTTPAdapter, ModelAdapter
from horizon_core.models import HorizonResult, Node, Edge, DepthStats


app = FastAPI(
    title="Next Word Horizon API",
    description="API for Probabilistic Horizon Expansion",
    version="0.1.0"
)

# Cache for adapters (to avoid reloading models on every request)
_adapter_cache: Dict[str, ModelAdapter] = {}


class ExpandRequest(BaseModel):
    """Request model for horizon expansion."""
    prompt: str
    top_k: int = 20
    max_depth: int = 3
    model_backend: Literal["local", "remote"] = "local"
    model_name: Optional[str] = None  # If backend = local
    remote_base_url: Optional[str] = None  # If backend = remote
    temperature: Optional[float] = 1.0  # Optional, for future use
    seed: Optional[int] = 42  # Optional, for reproducibility


class NodeResponse(BaseModel):
    """Response model for a node."""
    id: str
    parent_id: Optional[str]
    depth: int
    token_id: Optional[int]
    token_text: Optional[str]
    local_prob: float
    cumulative_prob: float
    proj: Optional[List[float]]  # [x, y, z] or None


class DepthStatsResponse(BaseModel):
    """Response model for depth statistics."""
    entropy: float
    branching_factor: float
    total_prob_mass: float
    num_nodes: int


class ExpandResponse(BaseModel):
    """Response model for horizon expansion."""
    nodes: List[NodeResponse]
    edges: List[List[str]]  # [[source_id, target_id], ...]
    depth_stats: Dict[str, DepthStatsResponse]
    max_depth: int
    root_node_id: str
    truncated: bool = False  # True om expansionen stoppades p.g.a. MAX_NODES


def _create_adapter(
    model_backend: str,
    model_name: Optional[str] = None,
    remote_base_url: Optional[str] = None
) -> ModelAdapter:
    """
    Creates a ModelAdapter based on model_backend.
    
    Args:
        model_backend: "local" or "remote"
        model_name: Model name for local backend
        remote_base_url: Base URL for remote backend
        
    Returns:
        ModelAdapter instance
        
    Raises:
        HTTPException: If parameters are missing or adapter cannot be created
    """
    if model_backend == "local":
        if not model_name:
            raise HTTPException(
                status_code=400,
                detail="model_name is required when model_backend is 'local'"
            )
        
        # Use cache if possible
        cache_key = f"local:{model_name}"
        if cache_key in _adapter_cache:
            return _adapter_cache[cache_key]
        
        try:
            # Resolve relative paths to absolute paths (relative to project root)
            # This ensures models can be found regardless of current working directory
            if model_name.startswith('./') or model_name.startswith('.\\'):
                # Get the project root (where this file is located)
                project_root = Path(__file__).parent.parent.resolve()
                model_path = (project_root / model_name.lstrip('./\\')).resolve()
                if model_path.exists():
                    model_name = str(model_path)
                else:
                    # If resolved path doesn't exist, try original path
                    # (might be a HuggingFace model ID)
                    pass
            
            # Determine device (try to use CUDA if available)
            device = None
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
            except ImportError:
                pass
            
            adapter = LocalHFAdapter(model_name=model_name, device=device)
            _adapter_cache[cache_key] = adapter
            return adapter
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create LocalHFAdapter: {str(e)}"
            )
    
    elif model_backend == "remote":
        if not remote_base_url:
            raise HTTPException(
                status_code=400,
                detail="remote_base_url is required when model_backend is 'remote'"
            )
        
        # Use cache if possible
        cache_key = f"remote:{remote_base_url}"
        if cache_key in _adapter_cache:
            return _adapter_cache[cache_key]
        
        try:
            adapter = RemoteHTTPAdapter(base_url=remote_base_url)
            _adapter_cache[cache_key] = adapter
            return adapter
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create RemoteHTTPAdapter: {str(e)}"
            )
    
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model_backend: {model_backend}. Must be 'local' or 'remote'"
        )


def _serialize_horizon_result(result: HorizonResult) -> Dict[str, Any]:
    """
    Serializes HorizonResult to a JSON-serializable dict.
    
    Removes embeddings (not needed client-side) and keeps
    only proj, probabilities, text etc.
    
    Args:
        result: HorizonResult to serialize
        
    Returns:
        Dict that can be serialized to JSON
    """
    # Serialize nodes (without embeddings)
    nodes = []
    for node in result.nodes:
        node_dict = {
            "id": node.id,
            "parent_id": node.parent_id,
            "depth": node.depth,
            "token_id": node.token_id if node.token_id != -1 else None,
            "token_text": node.token_text if node.token_text != "<ROOT>" else None,
            "local_prob": node.local_prob,
            "cumulative_prob": node.cumulative_prob,
            "proj": list(node.proj) if node.proj is not None else None
        }
        nodes.append(node_dict)
    
    # Serialize edges
    edges = [[edge.source_id, edge.target_id] for edge in result.edges]
    
    # Serialize depth_stats
    depth_stats = {}
    for depth, stats in result.depth_stats.items():
        depth_stats[str(depth)] = {
            "entropy": stats.entropy,
            "branching_factor": stats.branching_factor,
            "total_prob_mass": stats.total_prob_mass,
            "num_nodes": stats.num_nodes
        }
    
    return {
        "nodes": nodes,
        "edges": edges,
        "depth_stats": depth_stats,
        "max_depth": result.max_depth,
        "root_node_id": result.root_node_id,
        "truncated": result.truncated
    }


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Next Word Horizon API", "status": "running"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/expand_horizon", response_model=ExpandResponse)
async def expand_horizon_endpoint(request: ExpandRequest):
    """
    Expands a horizon tree from given prompt and returns it with 3D projection.
    
    Flow:
    1. Creates ModelAdapter based on model_backend
    2. Calls expand_horizon() to build the tree
    3. Calls project_horizon_to_3d() to project to 3D
    4. Serializes result to JSON (without embeddings)
    
    Args:
        request: ExpandRequest with prompt, parameters and model configuration
        
    Returns:
        ExpandResponse with expanded tree including 3D projection
    """
    try:
        # Step 1: Create ModelAdapter
        adapter = _create_adapter(
            model_backend=request.model_backend,
            model_name=request.model_name,
            remote_base_url=request.remote_base_url
        )
        
        # Step 2: Expand horizon tree
        horizon_result = expand_horizon(
            adapter=adapter,
            prompt=request.prompt,
            top_k=request.top_k,
            max_depth=request.max_depth
        )
        
        # Step 3: Project to 3D
        horizon_result = project_horizon_to_3d(horizon_result)
        
        # Step 4: Serialize to JSON format
        serialized = _serialize_horizon_result(horizon_result)
        
        return ExpandResponse(**serialized)
        
    except HTTPException:
        # Let HTTPException pass through
        raise
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

