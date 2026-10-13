# Semantic Matching in Rust

Local, offline semantic matching engine using ONNX Runtime embeddings (`all-MiniLM-L6-v2`) and cosine similarity (dot product on normalized vectors).

---

## 1. Setting Up Rust (One-Time)

```bash
# Install Rust compiler (rustc) and package manager (cargo)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Add cargo to current shell PATH
source "$HOME/.cargo/env"

# Install official linter and formatter
rustup component add clippy rustfmt

# (Optional) Install file watcher for instant feedback on save
cargo install cargo-watch
```

---

## 2. Managing & Downloading Model Resources

The `models/` directory is excluded from Git to prevent repository bloat. Model weights are distributed via GitHub Releases.

### Package local model for a GitHub Release:
```bash
# Compress model directory into an archive (~80 MB)
tar -czvf all-MiniLM-L6-v2.tar.gz -C models/all-MiniLM-L6-v2 .

# Generate SHA256 checksum for tracking
sha256sum all-MiniLM-L6-v2.tar.gz
```

### Download model on a fresh clone:
```bash
# Create target directory
mkdir -p models/all-MiniLM-L6-v2

# Download release archive
curl -fsSL -o /tmp/model.tar.gz \
  "https://github.com/<USER>/<REPO>/releases/download/model-v1/all-MiniLM-L6-v2.tar.gz"

# Extract directly into models directory and clean up
tar -xzf /tmp/model.tar.gz -C models/all-MiniLM-L6-v2
rm -f /tmp/model.tar.gz
```

---

## 3. Compiling

```bash
# Fast borrow/type checker (instant, skips binary generation)
cargo check

# Compile in Debug mode (fast build, unoptimized)
cargo build

# Compile in Release mode (essential for ML: SIMD/AVX2 optimizations enabled)
cargo build --release

# Format code
cargo fmt

# Run linter
cargo clippy
```

---

## 4. Running the Application

```bash
# Build and run directly in release mode
cargo run --release

# Or execute the compiled standalone binary directly
./target/release/rust

# Run with file watcher (auto-recompiles and runs on file change)
cargo watch -x "run --release"
```

---

## Input Format

The application expects an `input.json` file in the working directory:

```json
{
  "query": "How do I optimize CPU performance for low-level systems programming?",
  "candidates": [
    "Compiling with release optimizations and SIMD instructions can drastically improve execution speed.",
    "Efficient cache locality and avoiding false sharing between CPU threads maximizes throughput.",
    "A recipe for baking sourdough bread at home with simple ingredients."
  ]
}
```
