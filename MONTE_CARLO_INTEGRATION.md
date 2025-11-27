# Monte Carlo Sampling Integration Guide

This document explains how to integrate the new Monte Carlo sampling approach into Next Word Horizon.

## Overview

The new Monte Carlo sampling replaces the old tree expansion with a two-phase approach:
1. **Phase 1**: Shallow rollouts to discover semantic modes (clusters)
2. **Phase 2**: Guided deep sampling with head/mid/tail stratification

## What Has Been Created

### 1. Monte Carlo Sampler Module (`src/sampling/monte_carlo_sampler.py`)
- `SamplerConfig`: Configuration for sampling parameters
- `SampledStep`: Data class for each sampled step
- Phase 1 functions: `run_phase1_mode_discovery`, clustering, budget allocation
- Phase 2 functions: `run_phase2_guided_sampling` with head/mid/tail
- Top-level entrypoint: `build_semantic_dataset_for_prompt`

### 2. Adapter Functions (`horizon_core/monte_carlo_adapter.py`)
- `create_get_logits_fn(adapter)`: Wraps `ModelAdapter.get_logits()` for Monte Carlo sampler
- `create_embed_fn(adapter)`: Wraps `ModelAdapter.get_sequence_embedding()` for embeddings
- `create_cluster_fn(num_clusters)`: Creates KMeans clustering function

### 3. Converter (`horizon_core/sampled_steps_to_horizon.py`)
- `sampled_steps_to_horizon()`: Converts `List[SampledStep]` → `HorizonResult`
- Builds nodes from unique prefixes
- Builds edges from (prefix → next_token) transitions
- Aggregates probabilities and frequencies
- Calculates depth statistics

## Integration Steps

### Option 1: Use in API Server (Recommended)

Update `api/server.py` to add a new endpoint or modify existing one:

```python
from horizon_core.monte_carlo_adapter import (
    create_get_logits_fn,
    create_embed_fn,
    create_cluster_fn
)
from horizon_core.sampled_steps_to_horizon import sampled_steps_to_horizon
from src.sampling.monte_carlo_sampler import (
    SamplerConfig,
    build_semantic_dataset_for_prompt
)

@app.post("/expand_horizon_mc", response_model=ExpandResponse)
async def expand_horizon_mc_endpoint(request: ExpandRequest):
    """New endpoint using Monte Carlo sampling."""
    try:
        # Create adapter
        adapter = _create_adapter(
            model_backend=request.model_backend,
            model_name=request.model_name,
            remote_base_url=request.remote_base_url
        )
        
        # Create adapter functions
        get_logits_fn = create_get_logits_fn(adapter)
        embed_fn = create_embed_fn(adapter)
        cluster_fn = create_cluster_fn(num_clusters=40)
        
        # Configure sampler
        cfg = SamplerConfig(
            max_nodes=request.max_nodes if hasattr(request, 'max_nodes') else 30_000,
            phase1_rollouts=800,
            phase2_total_rollouts=600,
        )
        
        # Run Monte Carlo sampling
        steps = build_semantic_dataset_for_prompt(
            prompt=request.prompt,
            tokenizer=adapter.tokenizer,  # Need to expose tokenizer
            model=None,  # Not used directly
            get_logits_fn=get_logits_fn,
            embed_fn=embed_fn,
            cluster_fn=cluster_fn,
            cfg=cfg,
        )
        
        # Convert to HorizonResult
        prompt_ids = tuple(adapter.encode(request.prompt))
        result = sampled_steps_to_horizon(
            steps=steps,
            prompt_ids=prompt_ids,
            adapter=adapter
        )
        
        # Project to 3D
        result = project_horizon_to_3d(result)
        
        # Serialize
        serialized = _serialize_horizon_result(result)
        return ExpandResponse(**serialized)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Option 2: Use in Colab Notebook

For Colab, you can use the functions directly:

```python
from src.sampling.monte_carlo_sampler import (
    SamplerConfig,
    build_semantic_dataset_for_prompt,
)

# Create adapter functions
def get_logits_fn(prefix):
    input_ids = torch.tensor(prefix, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(input_ids)
        logits = out.logits[0, -1, :].detach().cpu().numpy()
    return logits

def embed_fn(prefix):
    input_ids = torch.tensor(prefix, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True)
        hidden = out.hidden_states[-1][0, -1, :].detach().cpu().numpy()
    return hidden

from sklearn.cluster import KMeans
def cluster_fn(embeddings, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    return labels

# Configure and run
cfg = SamplerConfig(
    max_nodes=30_000,
    phase1_rollouts=800,
    phase2_total_rollouts=600,
)

steps = build_semantic_dataset_for_prompt(
    prompt=prompt_text,
    tokenizer=tokenizer,
    model=model,
    get_logits_fn=get_logits_fn,
    embed_fn=embed_fn,
    cluster_fn=cluster_fn,
    cfg=cfg,
)

# Convert to HorizonResult (if using existing visualization)
from horizon_core.sampled_steps_to_horizon import sampled_steps_to_horizon
result = sampled_steps_to_horizon(
    steps=steps,
    prompt_ids=tuple(tokenizer.encode(prompt_text)),
    adapter=adapter
)
```

## Key Differences from Old Approach

### Old Approach (`expand_horizon`)
- Breadth-first tree expansion
- Fixed branching at each level
- No semantic clustering
- Can explode exponentially

### New Approach (`build_semantic_dataset_for_prompt`)
- Monte Carlo sampling with two phases
- Semantic clustering for mode discovery
- Head/mid/tail stratification
- Hard node cap enforcement
- Better coverage of probability space

## Configuration Tuning

Key parameters in `SamplerConfig`:

- `max_nodes`: Global hard cap (default: 30,000)
- `phase1_rollouts`: Number of shallow rollouts (default: 1000)
- `phase1_depth`: Depth of Phase 1 rollouts (default: 8)
- `phase2_total_rollouts`: Total deep rollouts (default: 800)
- `phase2_depth`: Depth of Phase 2 rollouts (default: 20)
- `num_clusters`: Number of semantic clusters (default: 40)
- `top_k_phase1`, `top_k_phase2`: Token selection limits
- `head_k`, `mid_k`: Head/mid/tail split points

## Benefits

1. **Controlled Growth**: Hard cap on nodes prevents explosion
2. **Semantic Diversity**: Clustering ensures coverage of different modes
3. **Efficient Sampling**: Head/mid/tail stratification balances exploration
4. **Backward Compatible**: Output can be converted to existing `HorizonResult` format

## Next Steps

1. Update `ModelAdapter` to expose `tokenizer` attribute (for Colab compatibility)
2. Add Monte Carlo endpoint to API (or make it the default)
3. Update UI to optionally use Monte Carlo sampling
4. Test with various prompts and configurations

