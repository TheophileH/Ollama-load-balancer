# Build Stage
FROM rust:slim-bullseye as builder

WORKDIR /app

# create empty project to cache dependencies
RUN USER=root cargo new --bin rust_lb
WORKDIR /app/rust_lb
COPY Cargo.toml ./
# Need pkg-config and ssl for reqwest
RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*
# Build dependencies only
RUN cargo build --release
# Remove dummy build
RUN rm src/*.rs

# Copy source code
COPY src ./src
# Build application
# Force touch main.rs to ensure rebuild
RUN touch src/main.rs
RUN cargo build --release

# Runtime Stage
FROM debian:bullseye-slim

WORKDIR /app
# Install runtime dependencies
# Note: nvidia-smi/rocm-smi are accessed on host via Command::new, not in container
RUN apt-get update && apt-get install -y \
    libssl1.1 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*


COPY --from=builder /app/rust_lb/target/release/ollama-load-balancer-rust ./server


CMD ["./server"]
