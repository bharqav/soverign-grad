# Production Multi-stage Docker Build for SovereignGrad
FROM nvidia/cuda:12.0.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install core build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy source repository
COPY . .

# Build with CUDA enabled and release optimizations
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON && \
    cmake --build build --config Release -j$(nproc)

# Production runtime container
FROM nvidia/cuda:12.0.0-runtime-ubuntu22.04 AS runner

WORKDIR /app

# Copy compiled binaries from builder stage
COPY --from=builder /workspace/build/sovereign_grad /app/
COPY --from=builder /workspace/build/sovereign_grad_test /app/
COPY --from=builder /workspace/build/sovereign_grad_benchmark /app/
COPY --from=builder /workspace/build/mnist_demo /app/
COPY --from=builder /workspace/build/models_demo /app/
COPY --from=builder /workspace/build/sgrd-cli /app/

# Default entrypoint runs full verification suite
ENTRYPOINT ["/app/sovereign_grad_test"]
