# Issue Board: Open Issues & Enhancements

## Issue 1: Pipeline Infrastructure Optimizations (Synchronous Data Loading)

**Status:** Open | **Priority:** Medium

**Description:**

The `DoomStreamingDataset` currently applies dynamic NumPy transposition, mirroring, and PyTorch tensor casting synchronously inside `__getitem__`. In `train.py`, the `DataLoader` is instantiated without multiprocessing or memory pinning. This creates an I/O bottleneck where the GPU idles while waiting for the CPU to transform the next batch.

**Proposed Solution:**

Refactor the `DataLoader` initialization in `train.py` to offload ETL transformations to background processes. Implement `num_workers` (e.g., 4), enable `pin_memory=True` for faster Host-to-Device memory transfers, and establish a `prefetch_factor`.

## Issue 2: Stateful Backpropagation Through Time (BPTT) Amnesia

**Status:** Open | **Priority:** High

**Description:**

The LNN's Closed-form Continuous (CfC) cells require a continuous flow of time to accurately accumulate evidence and trigger action potentials. Currently, the training loop implicitly initializes the hidden state `hx` as `None` (a zero-tensor) for every 32-frame sequence batch. This forces computational amnesia at the boundary of every sequence, breaking the mathematical continuity of the ODEs representing the agent's memory.

**Proposed Solution:**

Implement Stateful BPTT in `train.py`:

1. Disable `shuffle=True` across sequences of the same trajectory to ensure chronological streaming.
2. Retain the hidden state output `hx` from the previous batch.
3. Detach the state from the computational graph (`hx = hx.detach()`) to prevent backpropagating into infinite history.
4. Pass the detached state as the prior for the subsequent batch.

## Issue 3: Memory Overflow Risk in Dataset Loading (RAM Bottleneck)

**Status:** Open | **Priority:** Medium

**Description:**

In `dataset.py`, `DoomStreamingDataset` currently iterates through all `.npz` files and loads the raw numpy arrays directly into standard Python lists (`self.video_arrays.append(frames)`). As the dataset scales to hours of multi-modal gameplay (including dense spatial audio and thermal masks), this will exceed consumer RAM limits and cause catastrophic Out-Of-Memory (OOM) crashes before training even begins.

**Proposed Solution:**

Migrate the storage backend from compressed `.npz` archives to HDF5 (`h5py`) format, or utilize NumPy's `mmap_mode='r'` to memory-map the data on disk. This allows the `Dataset` to lazily stream tensor blocks directly from the NVMe/SSD without pre-loading the entire corpus into volatile memory.

## Issue 4: Phenomenological Saliency Mapping (Grad-CAM)

**Status:** Open | **Priority:** High

**Description:**

We currently lack explainable AI (XAI) tooling to verify that the agent's distinct sensor cortices (Visual, Depth, Thermal) are specializing as intended. We need visual proof that the Visual Cortex focuses on static geometry while the Thermal Cortex tracks dynamic threats.

**Proposed Solution:**

Integrate the `captum` library to generate Gradient-weighted Class Activation Mapping (Grad-CAM) heatmaps. Create an `examine` command that takes a single sequence from the dataset, runs `captum.attr.LayerGradCam` on the final convolutional layers of the respective cortices, and saves the upsampled heatmaps as side-by-side `.png` files. This will allow us to physically view the spatial stimuli responsible for triggering specific action logits.

## Issue 5: Audit Validation Leak & Redundancy (Train/Test Split)

**Status:** Open | **Priority:** Medium

**Description:**

The `audit` command currently evaluates the model against the `data/training/` directory. This causes a validation leak, resulting in an artificially inflated accuracy score (~97%) because the model is tested on its own training data. Furthermore, evaluating overlapping sliding windows inflates the sample count by a factor of 32.

**Proposed Solution:**

1. Establish a dedicated `data/validation/` directory. Update the ETL pipeline (`record.py`, `intervene.py`) to randomly route 10-15% of recorded episodes into this holdout folder.
2. Modify the `audit` command to strictly target this validation directory.
3. Add a `stride` parameter to `DoomStreamingDataset`. During `audit`, set `stride=32` so the dataloader yields non-overlapping sequences, evaluating each frame exactly once.
