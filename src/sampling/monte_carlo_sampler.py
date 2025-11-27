from __future__ import annotations

from dataclasses import dataclass

from typing import List, Tuple, Dict, Any, Callable, Optional

import math

import random

import numpy as np


TokenId = int

Prefix = Tuple[TokenId, ...]  # immutable so we can use it as dict key





@dataclass

class SamplerConfig:

    # Phase 1: shallow exploration

    phase1_rollouts: int = 1000

    phase1_depth: int = 8



    # Phase 2: guided deep sampling

    phase2_total_rollouts: int = 800

    phase2_depth: int = 20

    phase2_min_rollouts_per_cluster: int = 5

    phase2_alpha: float = 0.7  # how strongly we follow cluster mass when allocating budget



    # Token selection

    top_k_phase1: int = 50

    top_k_phase2: int = 100

    top_p: float = 0.95  # can be None if you don't want top-p



    # Head/mid/tail split for phase 2

    head_k: int = 10

    mid_k: int = 100

    tail_min_prob: float = 1e-5  # ignore tokens with prob below this for "tail"



    # Global graph budget

    max_nodes: int = 30_000



    # Clusters

    num_clusters: int = 40

    cluster_max_nodes_factor: float = 0.7  # fraction of max_nodes spread across clusters



    # Cumulative probability / surprisal stopping

    min_branch_prob: float = 1e-8  # stop if branch_prob < this

    max_cum_surprisal: float = 80.0  # stop if -log P > this





@dataclass

class SampledStep:

    prefix: Prefix             # tokens BEFORE this step

    next_token: TokenId        # token generated at this step

    prob: float                # model prob of next_token at this step

    depth: int                 # step index in this rollout

    phase: int                 # 1 or 2

    rollout_id: int

    cluster_id: Optional[int] = None

    cum_logprob: float = 0.0


def logits_to_probs(logits: np.ndarray, top_k: Optional[int] = None, top_p: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:

    """

    Convert logits -> probs and optionally apply top-k / top-p.

    Returns (token_ids, probs) for the filtered subset.

    """

    # logits: shape [vocab_size]

    # For now assume logits is a 1D numpy array



    # numerical stability

    logits = logits - np.max(logits)

    probs = np.exp(logits)

    probs = probs / probs.sum()



    vocab_size = probs.shape[0]

    token_ids = np.arange(vocab_size)



    # sort by prob descending

    sort_idx = np.argsort(-probs)

    token_ids = token_ids[sort_idx]

    probs = probs[sort_idx]



    # top-k

    if top_k is not None and top_k < len(token_ids):

        token_ids = token_ids[:top_k]

        probs = probs[:top_k]



    # top-p (nucleus)

    if top_p is not None and 0.0 < top_p < 1.0:

        cum_probs = np.cumsum(probs)

        cutoff_idx = np.searchsorted(cum_probs, top_p) + 1

        cutoff_idx = min(cutoff_idx, len(token_ids))

        token_ids = token_ids[:cutoff_idx]

        probs = probs[:cutoff_idx]

        probs = probs / probs.sum()



    return token_ids, probs


def sample_one(token_ids: np.ndarray, probs: np.ndarray) -> Tuple[int, float]:

    """

    Sample a single token id from (token_ids, probs).

    Returns (token_id, prob_of_that_token).

    """

    idx = np.random.choice(len(token_ids), p=probs)

    return int(token_ids[idx]), float(probs[idx])


def run_phase1_mode_discovery(

    prompt_ids: Prefix,

    model: Any,

    get_logits_fn: Callable[[Prefix], np.ndarray],

    cfg: SamplerConfig,

) -> List[SampledStep]:

    """

    Run many short rollouts from the prompt.

    Returns a flat list of SampledStep (phase=1).

    """

    all_steps: List[SampledStep] = []

    rollout_id = 0



    for r in range(cfg.phase1_rollouts):

        prefix = list(prompt_ids)

        cum_logprob = 0.0



        for depth in range(cfg.phase1_depth):

            logits = get_logits_fn(tuple(prefix))  # shape [vocab_size]

            token_ids, probs = logits_to_probs(

                logits,

                top_k=cfg.top_k_phase1,

                top_p=cfg.top_p,

            )



            next_tok, p = sample_one(token_ids, probs)

            cum_logprob += -math.log(max(p, 1e-12))



            step = SampledStep(

                prefix=tuple(prefix),

                next_token=next_tok,

                prob=p,

                depth=depth,

                phase=1,

                rollout_id=rollout_id,

                cluster_id=None,

                cum_logprob=cum_logprob,

            )

            all_steps.append(step)



            prefix.append(next_tok)



        rollout_id += 1



    return all_steps


def compute_rollout_endpoints(

    steps: List[SampledStep],

    max_depth: int,

) -> Dict[int, Prefix]:

    """

    From phase-1 steps, recover for each rollout_id the final prefix.

    """

    endpoints: Dict[int, Prefix] = {}

    for step in steps:

        if step.phase != 1:

            continue

        if step.depth == max_depth - 1:

            # this is the last step of that rollout

            full_prefix = list(step.prefix) + [step.next_token]

            endpoints[step.rollout_id] = tuple(full_prefix)

    return endpoints


def cluster_phase1_endpoints(

    endpoints: Dict[int, Prefix],

    embed_fn: Callable[[Prefix], np.ndarray],

    cluster_fn: Callable[[np.ndarray, int], np.ndarray],

    cfg: SamplerConfig,

) -> Dict[int, int]:

    """

    Returns mapping rollout_id -> cluster_id.

    """

    rollout_ids = sorted(endpoints.keys())

    embeddings = []

    for rid in rollout_ids:

        emb = embed_fn(endpoints[rid])

        embeddings.append(emb)

    embeddings = np.stack(embeddings, axis=0)  # [R1, dim]



    cluster_labels = cluster_fn(embeddings, cfg.num_clusters)  # [R1]

    rollout_to_cluster: Dict[int, int] = {

        rid: int(cluster_labels[i]) for i, rid in enumerate(rollout_ids)

    }

    return rollout_to_cluster


def allocate_phase2_budgets(

    rollout_to_cluster: Dict[int, int],

    cfg: SamplerConfig,

) -> Dict[int, int]:

    """

    Compute how many deep rollouts to run per cluster.

    """

    # count how many rollouts per cluster in phase 1

    cluster_counts: Dict[int, int] = {}

    for cl in rollout_to_cluster.values():

        cluster_counts[cl] = cluster_counts.get(cl, 0) + 1



    total = sum(cluster_counts.values())

    budgets: Dict[int, int] = {}

    remaining = cfg.phase2_total_rollouts



    # first pass: fractional allocation

    for cl, n in cluster_counts.items():

        w = n / total if total > 0 else 0.0

        # soften with alpha

        effective = (w ** cfg.phase2_alpha) if w > 0 else 0.0

        r2 = int(round(cfg.phase2_total_rollouts * effective))

        r2 = max(r2, cfg.phase2_min_rollouts_per_cluster)

        budgets[cl] = r2

        remaining -= r2



    # if sum != total_rollouts, adjust a bit (not super crucial)

    # we can just add/remove 1 from random clusters to fix the mismatch

    if remaining > 0:

        # add leftover rollouts arbitrarily

        for cl in list(budgets.keys()):

            if remaining <= 0:

                break

            budgets[cl] += 1

            remaining -= 1

    elif remaining < 0:

        # remove extra rollouts

        remaining = -remaining

        for cl in list(budgets.keys()):

            if remaining <= 0:

                break

            if budgets[cl] > cfg.phase2_min_rollouts_per_cluster:

                budgets[cl] -= 1

                remaining -= 1



    return budgets


def split_head_mid_tail(

    token_ids: np.ndarray,

    probs: np.ndarray,

    cfg: SamplerConfig,

) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]], List[Tuple[int, float]]]:

    """

    Split (token_ids, probs) into head/mid/tail buckets.

    Returns lists of (token_id, prob).

    """

    pairs = list(zip(token_ids.tolist(), probs.tolist()))



    head = pairs[:cfg.head_k]

    mid = pairs[cfg.head_k:cfg.mid_k]

    tail = [(tid, p) for (tid, p) in pairs[cfg.mid_k:] if p >= cfg.tail_min_prob]



    return head, mid, tail


def sample_from_buckets(

    head: List[Tuple[int, float]],

    mid: List[Tuple[int, float]],

    tail: List[Tuple[int, float]],

) -> Tuple[int, float]:

    """

    Simple strategy: pick which bucket, then sample uniformly within it,

    weighting head > mid > tail.

    """

    # weights between buckets

    bucket_weights = []

    buckets = []



    if head:

        buckets.append(head)

        bucket_weights.append(0.6)

    if mid:

        buckets.append(mid)

        bucket_weights.append(0.3)

    if tail:

        buckets.append(tail)

        bucket_weights.append(0.1)



    # normalize

    s = sum(bucket_weights)

    bucket_weights = [w / s for w in bucket_weights]



    bucket_idx = np.random.choice(len(buckets), p=bucket_weights)

    bucket = buckets[bucket_idx]



    # within bucket: sample proportional to the token prob

    token_ids = np.array([t for (t, _) in bucket], dtype=np.int32)

    probs = np.array([p for (_, p) in bucket], dtype=np.float64)

    probs = probs / probs.sum()



    idx = np.random.choice(len(token_ids), p=probs)

    return int(token_ids[idx]), float(probs[idx])  # note: bucket-local prob


def run_phase2_guided_sampling(

    prompt_ids: Prefix,

    model: Any,

    get_logits_fn: Callable[[Prefix], np.ndarray],

    embed_fn: Callable[[Prefix], np.ndarray],

    endpoints: Dict[int, Prefix],

    rollout_to_cluster: Dict[int, int],

    budgets: Dict[int, int],

    cfg: SamplerConfig,

) -> List[SampledStep]:

    """

    Phase 2: from phase-1 endpoints, run deeper rollouts with stratified sampling,

    respecting global node and cluster caps.

    """

    all_steps: List[SampledStep] = []

    # cluster node caps

    cluster_max_nodes = int(cfg.max_nodes * cfg.cluster_max_nodes_factor / max(1, cfg.num_clusters))

    cluster_node_counts: Dict[int, int] = {}

    # global node bookkeeping (unique prefixes)

    seen_prefixes: Dict[Prefix, int] = {}

    num_nodes = 0



    # seed id for phase2 rollouts

    rollout_id_base = max(rollout_to_cluster.keys()) + 1 if rollout_to_cluster else 0

    next_rollout_id = rollout_id_base



    # for each cluster, choose seeds and run rollouts

    cluster_to_rollout_ids: Dict[int, List[int]] = {}

    for rid, cl in rollout_to_cluster.items():

        cluster_to_rollout_ids.setdefault(cl, []).append(rid)



    for cl, seeds in cluster_to_rollout_ids.items():

        budget = budgets.get(cl, 0)

        if budget <= 0:

            continue



        # If cluster is already saturated (unlikely initially)

        if cluster_node_counts.get(cl, 0) >= cluster_max_nodes:

            continue



        for _ in range(budget):

            if not seeds:

                break



            # pick a random seed endpoint in this cluster

            seed_rid = random.choice(seeds)

            seed_prefix = endpoints[seed_rid]

            prefix = list(seed_prefix)

            cum_logprob = 0.0

            rollout_id = next_rollout_id

            next_rollout_id += 1



            # ensure seed prefix is counted as a node

            pref_t = tuple(prefix)

            if pref_t not in seen_prefixes:

                if num_nodes >= cfg.max_nodes:

                    return all_steps

                seen_prefixes[pref_t] = 1

                num_nodes += 1

                cluster_node_counts[cl] = cluster_node_counts.get(cl, 0) + 1



            # run deep rollout

            for depth in range(cfg.phase2_depth):

                # stop if cluster is saturated

                if cluster_node_counts.get(cl, 0) >= cluster_max_nodes:

                    break

                # stop if global graph saturated

                if num_nodes >= cfg.max_nodes:

                    return all_steps



                logits = get_logits_fn(tuple(prefix))

                token_ids, probs = logits_to_probs(

                    logits,

                    top_k=cfg.top_k_phase2,

                    top_p=cfg.top_p,

                )



                head, mid, tail = split_head_mid_tail(token_ids, probs, cfg)

                if not head and not mid and not tail:

                    break  # nothing usable



                next_tok, _local_p = sample_from_buckets(head, mid, tail)



                # find this token's true prob in the original probs array

                # (if you want exact prob; otherwise _local_p is fine)

                # map back to original probs

                # here: token_ids, probs are filtered lists from logits_to_probs

                mask = (token_ids == next_tok)

                if mask.any():

                    p_true = float(probs[mask][0])

                else:

                    p_true = 1e-12



                cum_logprob += -math.log(max(p_true, 1e-12))



                # check branch stopping criteria

                if math.exp(-cum_logprob) < cfg.min_branch_prob:

                    break

                if cum_logprob > cfg.max_cum_surprisal:

                    break



                step = SampledStep(

                    prefix=tuple(prefix),

                    next_token=next_tok,

                    prob=p_true,

                    depth=depth,

                    phase=2,

                    rollout_id=rollout_id,

                    cluster_id=cl,

                    cum_logprob=cum_logprob,

                )

                all_steps.append(step)



                prefix.append(next_tok)



                # register new prefix as node

                pref_t = tuple(prefix)

                if pref_t not in seen_prefixes:

                    if num_nodes >= cfg.max_nodes:

                        return all_steps

                    seen_prefixes[pref_t] = 1

                    num_nodes += 1

                    cluster_node_counts[cl] = cluster_node_counts.get(cl, 0) + 1



    return all_steps


def build_semantic_dataset_for_prompt(

    prompt: str,

    tokenizer: Any,

    model: Any,

    get_logits_fn: Callable[[Prefix], np.ndarray],

    embed_fn: Callable[[Prefix], np.ndarray],

    cluster_fn: Callable[[np.ndarray, int], np.ndarray],

    cfg: Optional[SamplerConfig] = None,

) -> List[SampledStep]:

    """

    High-level sampler for Next Word Horizon:

    - Phase 1: shallow rollouts + mode discovery

    - Phase 2: guided deep sampling with head/mid/tail and node caps



    Returns a flat list of SampledStep records. The existing graph-building

    code can consume this list to construct nodes/edges, weights, etc.

    """

    if cfg is None:

        cfg = SamplerConfig()



    # tokenize prompt

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    prompt_ids = tuple(prompt_ids)



    # Phase 1

    phase1_steps = run_phase1_mode_discovery(

        prompt_ids=prompt_ids,

        model=model,

        get_logits_fn=get_logits_fn,

        cfg=cfg,

    )



    endpoints = compute_rollout_endpoints(phase1_steps, max_depth=cfg.phase1_depth)

    rollout_to_cluster = cluster_phase1_endpoints(

        endpoints=endpoints,

        embed_fn=embed_fn,

        cluster_fn=cluster_fn,

        cfg=cfg,

    )

    budgets = allocate_phase2_budgets(rollout_to_cluster, cfg)



    # Phase 2

    phase2_steps = run_phase2_guided_sampling(

        prompt_ids=prompt_ids,

        model=model,

        get_logits_fn=get_logits_fn,

        embed_fn=embed_fn,

        endpoints=endpoints,

        rollout_to_cluster=rollout_to_cluster,

        budgets=budgets,

        cfg=cfg,

    )



    return phase1_steps + phase2_steps

