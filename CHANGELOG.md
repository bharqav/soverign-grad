# Changelog

All notable changes to SovereignGrad are documented here. This project adheres to [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### In Progress
- Log compaction / snapshot support for reducing cold-start load times of large checkpoints
- ONNX model export adapter

---

## [0.1.0] - 2026-08-28

### Added

**Core Tensor Engine**
- Multi-dimensional strided `Tensor` class over shared `Storage` buffers
- `O(1)` zero-copy `transpose()`, `view()`, `slice()`, `select()`, `narrow()`, `index_select()`
- `DType` support: `Float32`, `Float16` (cast-only), `Int32`, `Int64`
- Device placement: `Device::CPU` and `Device::CUDA` (hardware or emulated)
- Factory methods: `zeros`, `ones`, `full`, `randn`, `uniform`, `arange`, `tril`, `triu`
- In-place operators with version-counter mutation detection (`add_`, `mul_`)
- `item()` scalar extraction, `get_element` / `set_element` indexed access

**Dynamic Reverse-Mode Autograd**
- Topological DAG traversal engine (`AutogradEngine::backward`)
- `NoGradGuard` RAII context for inference-time graph suppression
- `retain_graph` support for second-order gradient computation
- In-place mutation version-counter detection during backward - throws `std::runtime_error` on version mismatch
- Gradient hooks via `register_hook`
- Finite-difference `gradcheck` utility for numerical gradient validation

**Operator Coverage (`ops.hpp` / `ops.cpp`)**
- Elementwise: `add`, `sub`, `mul`, `div`, `neg`, `exp`, `log`, `sqrt`, `pow`, `clamp`
- Activations: `relu`, `gelu`, `silu`, `sigmoid`, `tanh`
- Reductions: `sum`, `mean`, `max` (with dim and global variants)
- Normalization: `layer_norm`, `rms_norm`, `batch_norm2d`, `batch_norm1d`, `batch_norm3d`
- Convolutions: `conv2d`, `conv1d` with full backward passes
- Pooling: `max_pool2d`, `avg_pool2d`
- Attention: `flash_attention` (tiled SRAM computation), `rotary_embedding`
- Embedding lookup: `embedding` with gradient accumulation
- Loss functions: `mse_loss`, `cross_entropy_loss`, `softmax_cross_entropy`, `bce_with_logits_loss`, `l1_loss`, `smooth_l1_loss`, `kl_div_loss`

**Compute Backends**
- CPU AVX2 SIMD engine: 8×8 microkernel with L1/L2 cache-blocked matrix multiplication
- CUDA backend: 2D register-tiled GEMM kernel (64×64 macro-blocks, 4×4 register tiles)
- CUDA emulation layer (`cuda_mock.hpp`): transparently compiles `.cu` kernels as C++ threads for CI without a GPU
- OpenMP multi-threading (auto-detected at CMake configure time)

**Memory Allocator (`MemoryAllocator`)**
- Best-fit caching pool: allocated blocks returned to free list on release, merged with adjacent free blocks
- Separate CPU and CUDA pools with independent tracking
- Peak memory statistics and human-readable `get_memory_summary()`
- Thread-safe via `std::mutex`

**Neural Network Layers (`nn::` namespace)**
- `Linear`, `Conv2d`, `Conv1d`, `BatchNorm2d`, `BatchNorm1d`, `BatchNorm3d`
- `MaxPool2d`, `AvgPool2d`, `Flatten`, `Softmax`, `Dropout`
- `LayerNorm`, `RMSNorm`
- `Embedding`
- `MultiheadAttention`, `CausalSelfAttention`, `FlashAttention`
- `RotaryEmbedding` (RoPE)
- `SwiGLU` gated MLP block
- `Sequential` variadic container
- Loss modules: `MSELoss`, `CrossEntropyLoss`, `BCEWithLogitsLoss`, `L1Loss`, `SmoothL1Loss`, `KLDivLoss`
- `clip_grad_norm` global gradient clipping utility

**Optimizers**
- `AdamW` with native CUDA kernel step (zero PCIe bounce)
- `Adam` (AdamW alias with zero weight decay)
- `SGD` with momentum and Nesterov acceleration
- `CosineAnnealingLR` learning rate scheduler

**Reference Model Zoo (`models.hpp`)**
- `ResNetBlock` / `DeepResNet` - Residual convolutional vision network
- `MiniGPT` / `TransformerBlock` - Decoder-only Transformer with causal masking
- `NanoLLaMA` - Transformer with RMSNorm, SwiGLU, RoPE, and CausalSelfAttention
- `BidirectionalTransformer` - BERT-style encoder
- `VAE` - Variational Autoencoder with reparameterization trick

**JIT Graph Compiler (`jit::` namespace)**
- `Graph` IR with typed `Node` objects
- Fusion passes: `pass_fuse_linear_activation`, `pass_fuse_residual_norm`, `pass_fuse_scale_shift`
- `CompiledModule` execution engine

**Quantization (`quantization.hpp`)**
- INT8 symmetric per-tensor quantization / dequantization
- INT4 grouped quantization
- `QuantizedLinear` layer for deployment inference

**Serialization**
- `.sgrd` binary checkpoint format (magic header `SGRD`, version-stamped)
- `serialization::save_module` / `load_module` for full `nn::Module` checkpoint round-trips
- Hugging Face SafeTensors (`.safetensors`) zero-copy parser and writer
- Bidirectional `.sgrd` ↔ `.safetensors` conversion via `sgrd-cli`

**CLI Tool (`sgrd-cli`)**
- `inspect` - display checkpoint metadata and parameter table
- `verify` - validate magic header and file integrity
- `export-json` - export checkpoint structure as JSON
- `inspect-safetensors` - inspect SafeTensors metadata
- `to-safetensors` / `from-safetensors` - format conversion

**Applications & Demos**
- `sovereign_grad` - Multi-layer digit classifier with AdamW training loop
- `mnist_demo` - Convolutional MNIST classifier on synthetic data
- `models_demo` - 5-model architecture demonstration suite (NanoLLaMA, DeepResNet, MiniGPT, BidirectionalTransformer, VAE)
- `sovereign_grad_benchmark` - Micro-benchmark suite with PyTorch comparison output
- `sovereign_grad_test` - 928-line automated test suite with numerical gradchecks

**Build & CI**
- Zero-dependency CMake build (`cmake -B build -DUSE_CUDA=OFF`)
- Optional CUDA hardware path (`-DUSE_CUDA=ON`) with automatic fallback to CPU emulation
- Optional ASan/UBSan via `-DENABLE_SANITIZERS=ON`
- Optional AVX2/FMA acceleration (auto-detected by processor)
- Optional OpenMP (auto-detected)
- GitHub Actions CI: Ubuntu (GCC + Clang), Windows (MSVC), macOS (AppleClang)
- GitHub Actions ASan/UBSan sanitizer job
- GitHub Actions PyTorch parity benchmark job
- Multi-stage Docker build (`Dockerfile`)

**Documentation**
- Full C++ API reference (`docs/API.md`)
- SafeTensors / `.sgrd` serialization format specification (`docs/SERIALIZATION_SPEC.md`)
- Contributing guide with operator and layer addition workflows (`CONTRIBUTING.md`)
- MIT License
