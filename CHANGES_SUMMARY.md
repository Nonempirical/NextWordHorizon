# Comprehensive Summary of Changes to Next Word Horizon

This document summarizes all logical and structural changes made to the software.

## Table of Contents
1. [Core Algorithm Changes](#core-algorithm-changes)
2. [Tree Expansion Improvements](#tree-expansion-improvements)
3. [UI/UX Changes](#uiux-changes)
4. [New Monte Carlo Sampling System](#new-monte-carlo-sampling-system)
5. [API and Infrastructure](#api-and-infrastructure)
6. [File Structure Changes](#file-structure-changes)

---

## Core Algorithm Changes

### 1. Probabilistic and Geometric Cutoff Mechanisms

**Location**: `horizon_core/horizon.py`

**What Changed**:
- Added intelligent child selection instead of simple top-k
- Implemented three cutoff mechanisms to control tree growth

**New Constants** (lines 20-24):
```python
MAX_NODES = 150_000              # Global hard limit for nodes
MASS_CUTOFF = 0.95               # Probability mass coverage per node
MAX_CHILDREN_PER_NODE = 12      # Max children per node (after diversity)
MIN_DIVERSITY_COSINE = 0.15     # Minimum cosine distance between children
```

**New Function**: `select_children_for_node()` (lines 45-122)
- **Purpose**: Selects representative, probability-weighted, and diverse children
- **Logic**:
  1. Sorts tokens by probability (descending)
  2. Iterates through candidates
  3. Checks diversity using cosine distance (1 - cos_sim)
  4. Stops when:
     - Cumulative probability mass reaches `MASS_CUTOFF` (95%), OR
     - Maximum children limit (`MAX_CHILDREN_PER_NODE`) is reached
  5. Only adds tokens that are sufficiently different from already selected ones

**Impact**:
- Prevents exponential explosion by limiting children per node
- Ensures semantic diversity through embedding-based filtering
- Maintains probability-weighted selection (covers 95% of mass)
- Remains deterministic (no randomness in selection)

### 2. Global Node Limit Enforcement

**Location**: `horizon_core/horizon.py`

**What Changed**:
- Changed from checking `len(nodes)` to using a `node_count` counter
- Added early stopping before model forward passes
- Prevents unnecessary computation when budget is exhausted

**Key Changes**:
- `node_count` starts at 0 (counts non-root nodes only)
- Checks `node_count >= MAX_NODES` **before** calling model
- Stops expansion immediately when limit is reached
- No more nodes, embeddings, or edges created after limit

**Before**: Could create nodes beyond limit, then filter
**After**: Stops creating nodes exactly at limit

**MAX_NODES Value**:
- Initially: 30,000
- Updated to: 150,000 (as requested)

### 3. Cancel Functionality

**Location**: `api/server.py`, `horizon_core/horizon.py`

**What Changed**:
- Added ability to cancel running expansions when new request arrives
- Uses `threading.Event` for thread-safe cancellation

**Implementation**:
- `api/server.py`: Global `_cancel_event` and `_cancel_lock`
- When new request arrives, sets cancel event to stop previous expansion
- `expand_horizon()` accepts optional `cancel_event` parameter
- Checks cancel event before model calls and before creating nodes

**Impact**:
- Users can start new expansion without waiting for old one
- Prevents wasted computation on cancelled requests
- Thread-safe implementation

---

## Tree Expansion Improvements

### Modified `expand_horizon()` Function

**Location**: `horizon_core/horizon.py` (lines 125-376)

**Key Changes**:

1. **Child Selection** (lines 252-270):
   - **Before**: Used raw top-k indices
   - **After**: Uses `select_children_for_node()` with:
     - Probability mass cutoff
     - Maximum children limit
     - Diversity filtering

2. **Embedding Retrieval** (lines 221-243):
   - Retrieves embeddings for top-k tokens
   - Filters out tokens where embedding retrieval fails
   - Normalizes probabilities after filtering

3. **Node Counting** (lines 162, 300):
   - Tracks `node_count` separately from `len(nodes)`
   - Increments after each node creation
   - Checks limit at multiple points

4. **Early Stopping** (lines 179-189, 275-281, 310-313):
   - Checks before model forward pass
   - Checks before creating each child
   - Checks after creating each child
   - Uses `stop_expansion` flag to break nested loops

### New `HorizonResult` Field

**Location**: `horizon_core/models.py` (line 49)

**Added**:
```python
truncated: bool = False  # True if expansion stopped due to MAX_NODES
```

**Purpose**: Indicates whether tree was truncated due to node limit

---

## UI/UX Changes

### Location: `ui/app.py`

### 1. Maximum Depth Slider (lines 335-341)
- **Before**: Maximum = 5
- **After**: Maximum = 20
- **Impact**: Users can explore deeper trees

### 2. Top-K Candidates Slider (lines 328-334)
- **Before**: Minimum = 5
- **After**: Minimum = 1
- **Impact**: Users can explore with fewer candidates

### 3. API Response Updates
- Added `truncated` field to API response
- UI can display warning when tree is truncated

---

## New Monte Carlo Sampling System

### Overview
Complete new sampling approach that replaces breadth-first tree expansion with a two-phase Monte Carlo strategy.

### New Files Created

#### 1. `src/sampling/monte_carlo_sampler.py` (817 lines)

**Purpose**: Core Monte Carlo sampling implementation

**Key Components**:

**Configuration** (`SamplerConfig`):
- Phase 1 parameters: rollouts, depth
- Phase 2 parameters: total rollouts, depth, cluster budgets
- Token selection: top-k, top-p for each phase
- Head/mid/tail split parameters
- Global node cap: `max_nodes = 30,000`
- Clustering: number of clusters, node distribution
- Stopping criteria: min branch prob, max cumulative surprisal

**Data Structure** (`SampledStep`):
- Records each sampled step with:
  - Prefix (tokens before step)
  - Next token
  - Probability
  - Depth, phase, rollout_id
  - Cluster ID (for phase 2)
  - Cumulative log probability

**Helper Functions**:
- `logits_to_probs()`: Converts logits → probabilities with top-k/top-p
- `sample_one()`: Samples single token from distribution

**Phase 1 Functions**:
- `run_phase1_mode_discovery()`: Runs many shallow rollouts
- `compute_rollout_endpoints()`: Extracts final prefixes
- `cluster_phase1_endpoints()`: Clusters endpoints using embeddings
- `allocate_phase2_budgets()`: Allocates deep rollout budget per cluster

**Phase 2 Functions**:
- `split_head_mid_tail()`: Splits tokens into head/mid/tail buckets
- `sample_from_buckets()`: Samples with weighted strategy (head > mid > tail)
- `run_phase2_guided_sampling()`: Runs deep rollouts with:
  - Global node cap enforcement
  - Per-cluster node caps
  - Stratified sampling (head/mid/tail)
  - Stopping criteria (min prob, max surprisal)

**Top-Level Entrypoint**:
- `build_semantic_dataset_for_prompt()`: Orchestrates entire process
  - Tokenizes prompt
  - Runs Phase 1
  - Clusters and allocates budgets
  - Runs Phase 2
  - Returns combined list of steps

#### 2. `src/sampling/__init__.py`

**Purpose**: Module exports
- Exports all functions and classes for easy import

#### 3. `horizon_core/monte_carlo_adapter.py` (136 lines)

**Purpose**: Adapter functions to connect `ModelAdapter` with Monte Carlo sampler

**Functions**:
- `create_get_logits_fn(adapter)`: Wraps `ModelAdapter.get_logits()`
  - Takes `Prefix` (tuple of token IDs)
  - Returns logits for last token as numpy array
- `create_embed_fn(adapter)`: Wraps `ModelAdapter.get_sequence_embedding()`
  - Takes `Prefix`
  - Returns embedding vector as numpy array
- `create_cluster_fn(num_clusters)`: Creates KMeans clustering function
  - Uses sklearn KMeans
  - Handles edge cases (not enough samples)

#### 4. `horizon_core/sampled_steps_to_horizon.py` (217 lines)

**Purpose**: Converts Monte Carlo output to existing `HorizonResult` format

**Function**: `sampled_steps_to_horizon()`

**Logic**:
1. Builds mapping: `prefix → node_id`
2. Processes all `SampledStep` records
3. Creates nodes from unique prefixes
4. Creates edges from (prefix → next_token) transitions
5. Aggregates probabilities:
   - Average local probability per node
   - Average cumulative probability per node
   - Edge frequencies and average probabilities
6. Calculates depth statistics:
   - Entropy per depth
   - Branching factor per depth
   - Total probability mass per depth
   - Node count per depth

**Impact**: Enables Monte Carlo sampling to work with existing visualization

#### 5. `MONTE_CARLO_INTEGRATION.md`

**Purpose**: Integration guide
- Explains how to use new sampling system
- Provides code examples for API and Colab
- Compares old vs new approach
- Configuration tuning guide

---

## API and Infrastructure

### Location: `api/server.py`

### Changes Made:

1. **Cancel Mechanism** (lines 33-34):
   ```python
   _cancel_event = threading.Event()
   _cancel_lock = threading.Lock()
   ```

2. **Updated `expand_horizon_endpoint()`** (lines 255-264):
   - Cancels previous expansion before starting new one
   - Passes `cancel_event` to `expand_horizon()`

3. **Updated `ExpandResponse`** (line 75):
   - Added `truncated: bool = False` field

4. **Updated `_serialize_horizon_result()`** (line 217):
   - Includes `truncated` field in serialization

---

## File Structure Changes

### New Directories:
```
src/
  sampling/
    __init__.py
    monte_carlo_sampler.py (817 lines)
```

### New Files in `horizon_core/`:
- `monte_carlo_adapter.py` (136 lines)
- `sampled_steps_to_horizon.py` (217 lines)

### Modified Files:
- `horizon_core/horizon.py` (389 lines, +150 lines of new logic)
- `horizon_core/models.py` (+1 field to `HorizonResult`)
- `api/server.py` (+cancel mechanism, +truncated field)
- `ui/app.py` (slider limits updated)

### Documentation Files:
- `MONTE_CARLO_INTEGRATION.md` (new)
- `CHANGES_SUMMARY.md` (this file)

---

## Logical Flow Changes

### Old Flow:
```
Prompt → expand_horizon() → 
  For each depth:
    For each node:
      Get logits → Top-k → Create all children
  → HorizonResult
```

**Problems**:
- Exponential growth
- No diversity control
- No hard node limit enforcement
- Could create millions of nodes

### New Flow (Traditional Expansion):
```
Prompt → expand_horizon() → 
  For each depth:
    For each node:
      Check node_count < MAX_NODES
      Get logits → Top-k → Get embeddings
      select_children_for_node() → 
        (mass cutoff + diversity + max children)
      Create selected children only
      Increment node_count
  → HorizonResult (with truncated flag)
```

**Improvements**:
- Hard node limit enforced
- Diversity filtering
- Probability mass coverage
- Early stopping

### New Flow (Monte Carlo Sampling):
```
Prompt → build_semantic_dataset_for_prompt() →
  Phase 1:
    Run 1000 shallow rollouts (depth 8)
    Cluster endpoints
    Allocate Phase 2 budgets per cluster
  Phase 2:
    For each cluster:
      Run deep rollouts (depth 20)
      Use head/mid/tail stratified sampling
      Enforce global and per-cluster node caps
  → List[SampledStep]
  → sampled_steps_to_horizon() →
  → HorizonResult
```

**Advantages**:
- Semantic mode discovery
- Better probability space coverage
- Controlled growth with hard caps
- Stratified sampling (head/mid/tail)

---

## Performance Improvements

### Before:
- Could create 100,000+ nodes easily
- No early stopping
- All top-k children created
- No diversity control

### After:
1. **Traditional Expansion**:
   - Hard cap at 150,000 nodes
   - Stops before model calls when limit reached
   - Max 12 children per node (after diversity)
   - ~95% probability mass coverage

2. **Monte Carlo Sampling**:
   - Hard cap at 30,000 nodes (configurable)
   - Per-cluster caps
   - Stratified sampling for efficiency
   - Better semantic coverage

---

## Backward Compatibility

### Maintained:
- `HorizonResult` structure (with added `truncated` field)
- `ModelAdapter` interface (unchanged)
- API response format (with added `truncated` field)
- Visualization code (works with both approaches)

### New Options:
- Can use traditional expansion (with improvements)
- Can use Monte Carlo sampling (new approach)
- Both produce `HorizonResult` format

---

## Configuration Summary

### Traditional Expansion (`horizon_core/horizon.py`):
```python
MAX_NODES = 150_000
MASS_CUTOFF = 0.95
MAX_CHILDREN_PER_NODE = 12
MIN_DIVERSITY_COSINE = 0.15
```

### Monte Carlo Sampling (`src/sampling/monte_carlo_sampler.py`):
```python
SamplerConfig(
    max_nodes=30_000,
    phase1_rollouts=1000,
    phase1_depth=8,
    phase2_total_rollouts=800,
    phase2_depth=20,
    num_clusters=40,
    # ... many more parameters
)
```

---

## Testing Considerations

### What to Test:
1. Node limit enforcement (should stop at exactly MAX_NODES)
2. Diversity filtering (children should be semantically different)
3. Cancel functionality (new request should stop old one)
4. Monte Carlo sampling (should produce valid HorizonResult)
5. UI sliders (min/max values work correctly)
6. Truncated flag (correctly set when limit reached)

---

## Migration Path

### For Existing Users:
- No breaking changes
- Existing code continues to work
- New features are opt-in (Monte Carlo sampling)

### For New Development:
- Can use improved traditional expansion (automatic)
- Can adopt Monte Carlo sampling (new module)
- Both approaches available

---

## Summary Statistics

### Code Added:
- ~1,200 lines (Monte Carlo sampling system)
- ~350 lines (adapters and converters)
- ~50 lines (UI improvements)
- ~100 lines (cancel mechanism)

### Code Modified:
- ~200 lines (traditional expansion improvements)
- ~20 lines (API updates)
- ~10 lines (UI slider updates)

### Files Created: 6
### Files Modified: 4
### Total Impact: Significant improvement in control, efficiency, and capabilities

