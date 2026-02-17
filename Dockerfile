# Multi-stage build for minimal image size
FROM rust:1.87-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy workspace files
COPY Cargo.toml Cargo.lock ./
COPY llmg-core/Cargo.toml ./llmg-core/
COPY llmg-providers/Cargo.toml ./llmg-providers/
COPY llmg-gateway/Cargo.toml ./llmg-gateway/

# Copy source code
COPY llmg-core/src ./llmg-core/src
COPY llmg-providers/src ./llmg-providers/src
COPY llmg-gateway/src ./llmg-gateway/src

# Build release binary
RUN cargo build --release --bin llmg-gateway

# Runtime stage
FROM debian:bookworm-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy binary from builder
COPY --from=builder /app/target/release/llmg-gateway /usr/local/bin/llmg-gateway

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the gateway
ENTRYPOINT ["llmg-gateway"]
