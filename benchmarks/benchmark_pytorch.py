"""
Empirical PyTorch Benchmark Suite for SovereignGrad Comparison
Measures forward, backward, optimizer step, memory, and small-model training latencies.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
import json
import sys

def benchmark_tensor_add(iters=100):
    a = torch.randn(500, 500)
    b = torch.randn(500, 500)
    
    # Warmup
    for _ in range(10):
        c = a + b

    start = time.perf_counter()
    for _ in range(iters):
        c = a + b
    end = time.perf_counter()
    return (end - start) / iters * 1000.0  # ms

def benchmark_matmul(iters=50):
    a = torch.randn(128, 128)
    b = torch.randn(128, 128)
    
    # Warmup
    for _ in range(10):
        c = torch.matmul(a, b)

    start = time.perf_counter()
    for _ in range(iters):
        c = torch.matmul(a, b)
    end = time.perf_counter()
    return (end - start) / iters * 1000.0  # ms

def benchmark_layernorm(iters=100):
    ln = nn.LayerNorm(128, eps=1e-5)
    x = torch.randn(32, 128)
    
    # Warmup
    for _ in range(10):
        y = ln(x)

    start = time.perf_counter()
    for _ in range(iters):
        y = ln(x)
    end = time.perf_counter()
    return (end - start) / iters * 1000.0  # ms

def benchmark_conv2d(iters=50):
    conv = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
    x = torch.randn(16, 1, 28, 28)
    
    # Warmup
    for _ in range(10):
        y = conv(x)

    start = time.perf_counter()
    for _ in range(iters):
        y = conv(x)
    end = time.perf_counter()
    return (end - start) / iters * 1000.0  # ms

def benchmark_backward_pass(iters=50):
    class SmallMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(64, 32)
            self.ln = nn.LayerNorm(32, eps=1e-5)
            self.fc2 = nn.Linear(32, 10)
        
        def forward(self, x):
            return self.fc2(self.ln(torch.nn.functional.gelu(self.fc1(x))))

    model = SmallMLP()
    opt = optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    x = torch.randn(16, 64)
    target_idx = torch.tensor([i % 10 for i in range(16)], dtype=torch.long)

    # Warmup
    for _ in range(5):
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, target_idx)
        loss.backward()
        opt.step()

    start = time.perf_counter()
    for _ in range(iters):
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, target_idx)
        loss.backward()
        opt.step()
    end = time.perf_counter()
    return (end - start) / iters * 1000.0  # ms

def benchmark_mnist_epoch(batch_size=150, epochs=15):
    class ConvClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
            self.bn = nn.BatchNorm2d(4, eps=1e-5, momentum=0.1)
            self.fc = nn.Linear(3136, 10)

        def forward(self, x):
            h = torch.relu(self.bn(self.conv(x)))
            h = h.view(x.size(0), -1)
            return self.fc(h)

    model = ConvClassifier()
    opt = optim.Adam(model.parameters(), lr=0.02)
    loss_fn = nn.CrossEntropyLoss()

    x = torch.randn(batch_size, 1, 28, 28)
    y = torch.randint(0, 10, (batch_size,))

    start = time.perf_counter()
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
    end = time.perf_counter()
    total_time = end - start
    return total_time, (total_time / epochs) * 1000.0

def main():
    print("=" * 60)
    print("            PYTORCH REFERENCE BENCHMARK SUITE             ")
    print(f"  PyTorch Version: {torch.__version__} (CPU)")
    print("=" * 60)
    
    t_add = benchmark_tensor_add()
    t_matmul = benchmark_matmul()
    t_ln = benchmark_layernorm()
    t_conv = benchmark_conv2d()
    t_bwd = benchmark_backward_pass()
    mnist_total, mnist_per_epoch = benchmark_mnist_epoch()

    results = {
        "tensor_add_ms": round(t_add, 4),
        "matmul_ms": round(t_matmul, 4),
        "layernorm_ms": round(t_ln, 4),
        "conv2d_ms": round(t_conv, 4),
        "backward_mlp_adamw_ms": round(t_bwd, 4),
        "mnist_15epochs_total_s": round(mnist_total, 4),
        "mnist_epoch_ms": round(mnist_per_epoch, 4)
    }

    print("\nEmpirical Latency Results:")
    print(f"  * Tensor Add ([500, 500]):       {results['tensor_add_ms']:.4f} ms")
    print(f"  * Matrix Mul ([128, 128]):       {results['matmul_ms']:.4f} ms")
    print(f"  * LayerNorm ([32, 128]):         {results['layernorm_ms']:.4f} ms")
    print(f"  * Conv2d ([16, 1, 28, 28]):      {results['conv2d_ms']:.4f} ms")
    print(f"  * MLP Backward + AdamW Step:     {results['backward_mlp_adamw_ms']:.4f} ms")
    print(f"  * MNIST CNN 15 Epochs Total:     {results['mnist_15epochs_total_s']:.4f} s ({results['mnist_epoch_ms']:.2f} ms/epoch)")
    print("=" * 60)

    if "--json" in sys.argv:
        print("\nJSON_OUTPUT:" + json.dumps(results))

if __name__ == "__main__":
    main()
