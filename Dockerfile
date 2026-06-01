FROM nvidia/cuda:13.0.2-devel-ubuntu24.04 AS builder

ARG CUDA_COMPUTE_CAP=86

ENV DEBIAN_FRONTEND=noninteractive
ENV CARGO_HOME=/usr/local/cargo
ENV RUSTUP_HOME=/usr/local/rustup
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:/usr/local/cargo/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64
ENV CC=gcc-13
ENV CXX=g++-13
ENV NVCC_CCBIN=/usr/bin/g++-13
ENV CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP}

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    clang \
    cmake \
    curl \
    gcc-13 \
    g++-13 \
    git \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain stable

WORKDIR /app

COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY scripts/runpod_entrypoint.sh ./scripts/runpod_entrypoint.sh

RUN cargo build --release --locked --features gpu-cuda

FROM nvidia/cuda:13.0.2-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/temporal_memory /usr/local/bin/temporal_memory
COPY --from=builder /app/scripts/runpod_entrypoint.sh /usr/local/bin/runpod_entrypoint.sh

RUN chmod +x /usr/local/bin/runpod_entrypoint.sh

ENV TEMPORAL_MEMORY_DEVICE=cuda
ENV TEMPORAL_MEMORY_HOST=0.0.0.0
ENV TEMPORAL_MEMORY_PORT=3000
ENV PORT=3000
ENV PORT_HEALTH=3000
ENV TELLODB_DATA_DIR=/runpod-volume/tellodb
ENV TEMPORAL_MEMORY_VECTOR_CHECKPOINT_SECS=30

EXPOSE 3000

ENTRYPOINT ["/usr/local/bin/runpod_entrypoint.sh"]
