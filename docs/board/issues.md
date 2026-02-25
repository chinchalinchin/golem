# Issue Board: Open Issues & Enhancements

See [Issue Archive](./closed/phases.md) for the project's closed issues.

## Issue 1: Pipeline Infrastructure Optimizations (Synchronous Data Loading)

**Status:** Open | **Priority:** Medium | **Opened**: 2026/02/20

**Description:**

The `DoomStreamingDataset` currently applies dynamic NumPy transposition, mirroring, and PyTorch tensor casting synchronously inside `__getitem__`. In `train.py`, the `DataLoader` is instantiated without multiprocessing or memory pinning. This creates an I/O bottleneck where the GPU idles while waiting for the CPU to transform the next batch.

**Proposed Solution:**

Refactor the `DataLoader` initialization in `train.py` to offload ETL transformations to background processes. Implement `num_workers` (e.g., 4), enable `pin_memory=True` for faster Host-to-Device memory transfers, and establish a `prefetch_factor`.

## Issue 2: Memory Overflow Risk in Dataset Loading (RAM Bottleneck)

**Status:** Open | **Priority:** Medium | **Opened**: 2026/02/21

**Description:**

In `dataset.py`, `DoomStreamingDataset` currently iterates through all `.npz` files and loads the raw numpy arrays directly into standard Python lists (`self.video_arrays.append(frames)`). As the dataset scales to hours of multi-modal gameplay (including dense spatial audio and thermal masks), this will exceed consumer RAM limits and cause catastrophic Out-Of-Memory (OOM) crashes before training even begins.

**Proposed Solution:**

Migrate the storage backend from compressed `.npz` archives to HDF5 (`h5py`) format, or utilize NumPy's `mmap_mode='r'` to memory-map the data on disk. This allows the `Dataset` to lazily stream tensor blocks directly from the NVMe/SSD without pre-loading the entire corpus into volatile memory.

## Issue 3: Audit Validation Leak & Redundancy (Train/Test Split)

**Status:** Open | **Priority:** Medium | **Opened**: 2026/02/21

**Description:**

The `audit` command currently evaluates the model against the `data/training/` directory. This causes a validation leak, resulting in an artificially inflated accuracy score (~97%) because the model is tested on its own training data. Furthermore, evaluating overlapping sliding windows inflates the sample count by a factor of 32.

**Proposed Solution:**

1. Establish a dedicated `data/validation/` directory. Update the ETL pipeline (`record.py`, `intervene.py`) to randomly route 10-15% of recorded episodes into this holdout folder.
2. Modify the `audit` command to strictly target this validation directory.

## Issue 4: Live Empirical Benchmarking Pipeline (Headless Rollouts)

**Status:** Open | **Priority:** High | **Opened**: 2026/02/24

**Description:**

Currently, models are exclusively evaluated using static classification metrics (Precision/Recall) via the `audit` command. While this confirms the agent learned to mimic keystrokes, it fails to measure true agentic performance within the continuous POMDP. A 95% imitation accuracy can still yield a 0% survival rate if the 5% error margin results in catastrophic environmental failure (e.g., walking into environmental hazards or failing to dodge).

**Proposed Solution:**

1. Implement an automated `benchmark` pipeline command that executes headless ViZDoom engine rollouts across $N$ episodes.
2. Track, aggregate, and report true environmental variables rather than classification logits (e.g., average survival time, Kill/Death ratio, ammo efficiency, and total damage taken). 
3. (Optional) Script an automated tournament utilizing the existing `docker-compose` VDAIC arena to establish an ELO rating for different Golem model iterations against legacy bots.

## Issue 5: Latent State Visualization (Mapping the Liquid Core)

**Status:** Open | **Priority:** Medium | **Opened**: 2026/02/24

**Description:**

The defining feature of the Liquid Neural Network is its continuous hidden state $x(t)$ and its input-dependent varying time-constant. However, this state is currently a black box during both training and inference. We lack the tooling to verify if the network is actually learning meaningful phenomenological abstractions, or if the latent space is just a uniform, un-clustered blob.

**Proposed Solution:**

1. Create a tracing mechanism within the `run` loop to capture the `hx` state vectors and CfC gating activations over the course of a live episode.
2. Integrate dimensionality reduction tooling (UMAP or t-SNE) to project the high-dimensional latent traces into 2D/3D space, color-coded by the active environmental context (e.g., combat vs. exploration), to visually confirm geometric state clustering.
3. Plot the network's time-constant derivatives to verify the agent's memory horizon correctly "stretches" during quiet maze navigation and "snaps" (becomes highly reactive) during sudden threat stimuli.