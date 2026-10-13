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

### Option A: Download automatically via script (Recommended on fresh clone)
```bash
./scripts/fetch_models.sh
```
*Auto-detects the repository remote (`hammaddaftab/sematic-search`), downloads `all-MiniLM-L6-v2.tar.gz` from the `model-v1` release, and extracts it into `models/all-MiniLM-L6-v2/`.*

---

### Packaging the Model for a GitHub Release:
When publishing or updating the model on GitHub:

```bash
# 1. Compress the local models directory (~80 MB)
tar -czvf all-MiniLM-L6-v2.tar.gz -C models/all-MiniLM-L6-v2 .

# 2. Upload to GitHub Releases
# Web: https://github.com/hammaddaftab/sematic-search/releases/new
# Create tag: model-v1
# Attach file: all-MiniLM-L6-v2.tar.gz
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
