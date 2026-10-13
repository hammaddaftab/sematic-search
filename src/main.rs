use anyhow::{Context, Result};
use fastembed::{
    InitOptionsUserDefined, Pooling, TextEmbedding, TokenizerFiles, UserDefinedEmbeddingModel,
};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::time::Instant;

#[derive(Debug, Deserialize)]
struct SemanticSearchInput {
    query: String,
    candidates: Vec<String>,
}

#[derive(Debug, Serialize)]
struct MatchResult<'a> {
    score: f32,
    candidate: &'a str,
}

/// Compute dot product between two float vectors.
/// Since embeddings are L2-normalized, dot product is equivalent to cosine similarity.
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Load the embedding model completely offline from a persistent directory.
/// Zero dynamic downloads, zero network requests, zero magic caching.
fn load_offline_model(model_dir: &Path) -> Result<TextEmbedding> {
    let t0 = Instant::now();
    println!("Loading model offline from: {}", model_dir.display());

    let onnx_file = fs::read(model_dir.join("model.onnx"))
        .with_context(|| format!("Missing model.onnx in {}", model_dir.display()))?;
    let tokenizer_file = fs::read(model_dir.join("tokenizer.json"))
        .with_context(|| format!("Missing tokenizer.json in {}", model_dir.display()))?;
    let config_file = fs::read(model_dir.join("config.json"))
        .with_context(|| format!("Missing config.json in {}", model_dir.display()))?;
    let special_tokens_map_file = fs::read(model_dir.join("special_tokens_map.json"))
        .with_context(|| format!("Missing special_tokens_map.json in {}", model_dir.display()))?;
    let tokenizer_config_file = fs::read(model_dir.join("tokenizer_config.json"))
        .with_context(|| format!("Missing tokenizer_config.json in {}", model_dir.display()))?;

    let tokenizer_files = TokenizerFiles {
        tokenizer_file,
        config_file,
        special_tokens_map_file,
        tokenizer_config_file,
    };

    // all-MiniLM-L6-v2 uses Mean Pooling across token hidden states
    let user_model = UserDefinedEmbeddingModel::new(onnx_file, tokenizer_files)
        .with_pooling(Pooling::Mean);

    let model = TextEmbedding::try_new_from_user_defined(
        user_model,
        InitOptionsUserDefined::default(),
    )?;

    println!("Model initialized in {:.2?}", t0.elapsed());
    Ok(model)
}

fn main() -> Result<()> {
    // 1. Read input JSON
    let json_data = fs::read_to_string("input.json")
        .context("Failed to read input.json")?;
    let input: SemanticSearchInput = serde_json::from_str(&json_data)
        .context("Failed to parse input.json")?;

    println!("Loaded query: \"{}\"", input.query);
    println!("Candidate pool: {} items\n", input.candidates.len());

    // 2. Load model directly from deterministic persistent storage
    let model_dir = Path::new("models/all-MiniLM-L6-v2");
    let mut model = load_offline_model(model_dir)?;

    // 3. Generate embeddings
    let t_infer = Instant::now();
    let query_embeddings = model.embed(vec![&input.query], None)?;
    let query_vec = &query_embeddings[0];

    let candidate_texts: Vec<&str> = input.candidates.iter().map(|s| s.as_str()).collect();
    let candidate_embeddings = model.embed(candidate_texts, None)?;
    println!("Inference completed in {:.2?}\n", t_infer.elapsed());

    // 4. Compute similarity and rank
    let mut results: Vec<MatchResult> = input
        .candidates
        .iter()
        .zip(candidate_embeddings.iter())
        .map(|(cand, cand_vec)| MatchResult {
            score: dot_product(query_vec, cand_vec),
            candidate: cand.as_str(),
        })
        .collect();

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    // 5. Display results
    println!("=== Top Matches (Dot Product / Cosine Similarity) ===");
    for (rank, res) in results.iter().enumerate() {
        println!("[#{:02}] Score: {:.4} | {}", rank + 1, res.score, res.candidate);
    }

    Ok(())
}
