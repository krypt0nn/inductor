use std::collections::HashSet;

use burn::prelude::*;

pub mod database;
pub mod train_samples;
pub mod train_batches;
pub mod model;

pub const EMBEDDING_DEFAULT_ONE_HOT_TOKENS_NUM: usize = 65536;
pub const EMBEDDING_DEFAULT_SIZE: usize = 64;
pub const EMBEDDING_DEFAULT_CONTEXT_RADIUS: usize = 3;
pub const EMBEDDING_DEFAULT_MINIMAL_OCCURENCES: usize = 2;
pub const EMBEDDING_DEFAULT_SUBSAMPLE_VALUE: f64 = 1e-5;

pub mod prelude {
    pub use super::database::Database as WordEmbeddingsDatabase;

    pub use super::train_samples::{
        WordEmbeddingSamplingParams,
        WordEmbeddingsTrainSamplesDataset,
        WordEmbeddingTrainSample
    };

    pub use super::train_batches::{
        WordEmbeddingTrainSamplesBatcher,
        WordEmbeddingTrainSamplesBatch
    };

    pub use super::model::WordEmbeddingModel;

    pub use super::{
        EMBEDDING_DEFAULT_ONE_HOT_TOKENS_NUM,
        EMBEDDING_DEFAULT_SIZE,
        EMBEDDING_DEFAULT_CONTEXT_RADIUS,
        EMBEDDING_DEFAULT_MINIMAL_OCCURENCES,
        EMBEDDING_DEFAULT_SUBSAMPLE_VALUE,

        cosine_similarity,
        one_hot_tensor
    };
}

/// Calculate cosine similarity between two vectors.
///
/// Return value in `[-1.0, 1.0]` range where 1.0 means fully equal.
pub fn cosine_similarity(word_1: &[f32], word_2: &[f32]) -> f32 {
    let mut distance = 0.0;
    let mut len_1 = 0.0;
    let mut len_2 = 0.0;

    let n = std::cmp::max(word_1.len(), word_2.len());

    for i in 0..n {
        let word_1 = word_1.get(i).copied().unwrap_or(0.0);
        let word_2 = word_2.get(i).copied().unwrap_or(0.0);

        distance += word_1 * word_2;

        len_1 += word_1.powi(2);
        len_2 += word_2.powi(2);
    }

    distance / (len_1.sqrt() * len_2.sqrt())
}

/// Get one-hot encoding tensor from input tokens.
pub fn one_hot_tensor<B: Backend>(tokens: &[usize], length: usize, device: &B::Device) -> Tensor<B, 1, Float> {
    let mut tensor = Tensor::<B, 1, Float>::zeros([length], device);

    for token in HashSet::<usize>::from_iter(tokens.iter().copied()) {
        tensor = tensor.select_assign(
            0,
            Tensor::from_ints([token], device),
            Tensor::from_floats([1.0], device)
        );
    }

    tensor
}
