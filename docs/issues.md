# Issue Board: Open Issues & Enhancements

## Issue 1: The "Hold W" Convergence Trap (Class Imbalance)

**Status:** Open | **Priority:** Medium

**Description:**

DOOM datasets—especially navigation-focused modules like `my_way_home.wad`—are inherently unbalanced. An agent will spend 90% of an episode holding `MOVE_FORWARD`, which causes standard Binary Cross-Entropy (`BCEWithLogitsLoss`) to converge to a local minimum. The agent learns that perpetually predicting `MOVE_FORWARD` and ignoring visual stimuli yields the lowest overall loss. 

**Proposed Solution:**

Replace standard BCE with a **Focal Loss** function, or apply **Dynamic Sample Weighting** within the `DoomStreamingDataset`. 

Focal loss adds a modulating factor $(1 - p_t)^\gamma$ to the standard cross-entropy criterion, dynamically scaling down the gradient of easily classified, high-frequency actions (like walking forward), and heavily penalizing the network when it misses rare, high-value actions (like `ATTACK` or `USE`).

**Implementation Notes:**

* Calculate dataset action distributions during the `transform` or `dataset` loading phase.
* Pass the resulting weights tensor to the loss function in `train.py`.

## Issue 2: Pipeline Infrastructure Optimizations (Synchronous Data Loading)

**Status:** Open | **Priority:** Medium

**Description:**

The `DoomStreamingDataset` currently applies dynamic NumPy transposition, mirroring, and PyTorch tensor casting synchronously inside `__getitem__`. In `train.py`, the `DataLoader` is instantiated without multiprocessing or memory pinning. This creates an I/O bottleneck where the GPU idles while waiting for the CPU to transform the next batch.

**Proposed Solution:**

Refactor the `DataLoader` initialization in `train.py` to offload ETL transformations to background processes. Implement `num_workers` (e.g., 4), enable `pin_memory=True` for faster Host-to-Device memory transfers, and establish a `prefetch_factor`.

## Issue 3: Stateful Backpropagation Through Time (BPTT) Amnesia

**Status:** Open | **Priority:** High

**Description:**

The LNN's Closed-form Continuous (CfC) cells require a continuous flow of time to accurately accumulate evidence and trigger action potentials. Currently, the training loop implicitly initializes the hidden state `hx` as `None` (a zero-tensor) for every 32-frame sequence batch. This forces computational amnesia at the boundary of every sequence, breaking the mathematical continuity of the ODEs representing the agent's memory.

**Proposed Solution:**

Implement Stateful BPTT in `train.py`:

1. Disable `shuffle=True` across sequences of the same trajectory to ensure chronological streaming.
2. Retain the hidden state output `hx` from the previous batch.
3. Detach the state from the computational graph (`hx = hx.detach()`) to prevent backpropagating into infinite history.
4. Pass the detached state as the prior for the subsequent batch.

## Issue 4: Pipeline Infrastructure Optimizations (GPU-Accelerated DSP)

**Status:** Open | **Priority:** High

**Description:**

In the Phase IV implementation, `torchaudio.transforms.MelSpectrogram` and `AmplitudeToDB` operations were added directly inside the `__getitem__` method of the `DoomStreamingDataset`. Because PyTorch `DataLoader` workers execute `__getitem__` on the CPU by default, this creates a massive I/O bottleneck. Short-Time Fourier Transforms (STFTs) are computationally expensive; executing them sequentially on the CPU forces the target hardware accelerator (CUDA/MPS) to idle while waiting for the next batched audio tensor. This severely exacerbates the existing synchronous data loading latency.

**Proposed Solution:**

Remove the `torchaudio` transforms from `dataset.py`. Instantiate the transforms directly on the target hardware accelerator (`.to(device)`) inside the training (`train.py`) and inference loops, applying the transformation to the batched `x_aud` tensor *after* it has been loaded into VRAM. Alternatively, embed the `torchaudio` transforms directly into the forward pass of `DoomLiquidNet` in `brain.py` to ensure the DSP logic is natively compiled into the model's computational graph and executed on the GPU.