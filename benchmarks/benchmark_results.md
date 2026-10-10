# SovereignGrad vs PyTorch Benchmark Results

| Operation | PyTorch (ms) | SovereignGrad (ms) | Speedup | Notes |
|-----------|--------------|--------------------|---------|-------|
| Linear Fwd/Bwd (128x1024) | 6.28 | 94.42 | 0.07x | B=128, IN=1024, OUT=1024 |
| RMSNorm Fwd/Bwd (128x1024) | 1.15 | 0.74 | 1.56x | D=1024 |
| Attention (L=512, D=64) | 76.34 | 184.49 | 0.41x | B=8, H=12, L=512 |
| Transformer Block Fwd/Bwd | 938.49 | 5471.31 | 0.17x | Hidden=768, L=128 |
| AdamW Step (10x 1024x1024) | 61.81 | 111.79 | 0.55x | 10 tensors of 1MB |
| MNIST Synthetic Training | 54.02 | 93.86 | 0.58x | 20 batches of 64 |
| Memory Allocator (1024x1024) | 23.37 | 26.16 | 0.89x | Malloc/Free caching |
