"""
Monte Carlo sampling module for horizon expansion.

This module provides Monte Carlo sampling strategies for exploring
the probabilistic horizon tree.
"""

from .monte_carlo_sampler import (
    SamplerConfig,
    SampledStep,
    TokenId,
    Prefix,
    logits_to_probs,
    sample_one,
    run_phase1_mode_discovery,
    compute_rollout_endpoints,
    cluster_phase1_endpoints,
    allocate_phase2_budgets,
    split_head_mid_tail,
    sample_from_buckets,
    run_phase2_guided_sampling,
    build_semantic_dataset_for_prompt,
)

__all__ = [
    "SamplerConfig",
    "SampledStep",
    "TokenId",
    "Prefix",
    "logits_to_probs",
    "sample_one",
    "run_phase1_mode_discovery",
    "compute_rollout_endpoints",
    "cluster_phase1_endpoints",
    "allocate_phase2_budgets",
    "split_head_mid_tail",
    "sample_from_buckets",
    "run_phase2_guided_sampling",
    "build_semantic_dataset_for_prompt",
]

