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