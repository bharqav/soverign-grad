# Roadmap

This document tracks the implementation status and future direction of SovereignGrad.

---

## ✅ v0.1.0 - Implemented (August 2026)

### Core Framework
- [x] Multi-dimensional strided Tensor engine with `O(1)` views and transpositions
- [x] Dynamic reverse-mode automatic differentiation (tape-based DAG)
- [x] In-place mutation detection via version counters - prevents silent gradient corruption
- [x] `NoGradGuard` RAII context for inference mode
- [x] Finite-difference `gradcheck` utility for numerical gradient validation
- [x] Best-fit caching memory allocator with peak telemetry

### Compute Backends
- [x] CPU AVX2/FMA microkernel - 8×8 register-tiled matrix multiplication with L1/L2 cache blocking
- [x] CUDA GPU backend - 2D register-tiled GEMM (64×64 macro-blocks, 4×4 register tiles)
- [x] CUDA CPU emulation layer - compiles `.cu` kernels as C++ threads for GPU-free CI
- [x] OpenMP multi-threading (auto-detected)

### Neural Network Ecosystem
- [x] Layer zoo: `Linear`, `Conv2d`, `Conv1d`, `BatchNorm{1,2,3}d`, `LayerNorm`, `RMSNorm`, `MaxPool2d`, `AvgPool2d`, `Embedding`, `Dropout`, `Flatten`
- [x] Attention: `MultiheadAttention`, `CausalSelfAttention`, `FlashAttention`, `RotaryEmbedding`
- [x] SwiGLU gated MLP
- [x] Optimizers: `AdamW`, `Adam`, `SGD` with Nesterov, `CosineAnnealingLR`
- [x] Gradient clipping: `clip_grad_norm`
- [x] Loss functions: MSE, CrossEntropy, BCE, L1, SmoothL1, KLDiv

### Model Zoo
- [x] `DeepResNet` - Residual vision network with shortcut connections
- [x] `MiniGPT` / `TransformerBlock` - Decoder-only causal Transformer
- [x] `NanoLLaMA` - LLaMA-style: RMSNorm + SwiGLU + RoPE + CausalSelfAttention
- [x] `BidirectionalTransformer` - BERT-style encoder
- [x] `VAE` - Variational Autoencoder with reparameterization trick

### JIT & Quantization
- [x] JIT graph IR with operator fusion passes (Linear+Activation, Residual+Norm)
- [x] INT8 symmetric per-tensor quantization / dequantization
- [x] INT4 grouped quantization
- [x] `QuantizedLinear` layer

### Serialization & Interoperability
- [x] `.sgrd` binary checkpoint format (save/load full `nn::Module`)
- [x] Hugging Face SafeTensors (`.safetensors`) zero-copy parser and writer
- [x] `sgrd-cli` checkpoint inspection and conversion tool

### Infrastructure
- [x] Multi-OS CI (Ubuntu GCC, Ubuntu Clang, Windows MSVC, macOS AppleClang)
- [x] ASan/UBSan sanitizer CI job
- [x] PyTorch parity benchmark CI job
- [x] Multi-stage Docker build

---

## 🚧 v0.2.0 - Planned

### Memory & Persistence
- [ ] **Checkpoint log compaction** - snapshot state machine to avoid replaying full history on cold load
- [ ] **Memory-mapped tensor loading** - mmap-based zero-copy loading for large model weights
- [ ] **Mixed-precision training** - FP16 compute with FP32 master weights

### Compute
- [ ] **Batch matmul (`bmm`)** - batched matrix multiplication for 3D and 4D tensors
- [ ] **Strided conv backward optimization** - replace gather-scatter with direct im2col/col2im
- [ ] **Fused softmax + scale** - single-pass scaled softmax kernel

### Operators
- [ ] **`gather` / `scatter_add`** - needed for advanced embedding operations
- [ ] **`topk`** - for beam search decoding
- [ ] **`pad`** - general tensor padding with arbitrary fill values

---

## 🔮 v0.3.0 - Future

- [ ] **Multi-GPU training** - NCCL all-reduce for distributed data parallelism across physical GPUs
- [ ] **ONNX export** - export trained `nn::Module` to ONNX IR for deployment in third-party runtimes
- [ ] **TensorBoard event logging** - scalar, histogram, and graph logging compatible with TensorBoard
- [ ] **LSM-tree gradient checkpointing** - gradient checkpointing for training models larger than VRAM
- [ ] **Python bindings** - pybind11 interface to use SovereignGrad from Python without PyTorch

---

## Known Limitations (v0.1.0)

These are honest assessments of what is not production-ready in the current version:

| Limitation | Detail |
|------------|--------|
| No batch dimension in attention | `CausalSelfAttention` and `MultiheadAttention` operate on `[L, D]` sequences, not `[B, L, D]` batches |
| Serialization not adversarially hardened | `.sgrd` / `.safetensors` parsers do not guard against malformed inputs from untrusted sources |
| No gradient accumulation | Calling `optimizer.step()` before zeroing gradients accumulates - this is expected behavior but not explicitly documented |
| CUDA emulator is single-threaded | The CPU CUDA emulator runs kernels sequentially, not in parallel threads |
| No learning rate warm-up | `CosineAnnealingLR` has no linear warm-up phase |

See [CHANGELOG.md](CHANGELOG.md) for the complete v0.1.0 feature list.
