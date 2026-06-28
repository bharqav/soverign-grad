# Multi-stage Docker build for SovereignGrad
FROM nvidia/cuda:12.0.0-devel-ubuntu22.04 AS builder

# Install system dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set workspace
WORKDIR /workspace

# Copy codebase
COPY . .

# Build with CUDA enabled
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON && \
    cmake --build build --config Release

# Production runtime stage
FROM nvidia/cuda:12.0.0-runtime-ubuntu22.04 AS runner

WORKDIR /app

# Copy compiled binaries from builder stage
COPY --from=builder /workspace/build/sovereign_grad .

# Define entrypoint to run tests
ENTRYPOINT ["./sovereign_grad"]
