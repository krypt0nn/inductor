use std::collections::HashSet;

use burn::prelude::*;

pub mod database;
pub mod train_samples;
pub mod train_batches;
pub mod model;

/// Maximal amount of tokens which can be learned by the word embeddings model.
pub const EMBEDDING_MAX_TOKENS: usize = 65536;

/// Amount of tokens around the target one to train embeddings model on.
pub const EMBEDDING_CONTEXT_RADIUS: usize = 6;

/// Amount of dimensions in the word embedding vector.
pub const EMBEDDING_SIZE: usize = 32;

pub mod prelude {
    pub use super::database::Database as WordEmbeddingsDatabase;

    pub use super::train_samples::{
        WordEmbeddingsTrainSamplesDataset,
        WordEmbeddingTrainSample
    };

    pub use super::train_batches::{
        WordEmbeddingTrainSamplesBatcher,
        WordEmbeddingTrainSamplesBatch
    };

    pub use super::model::WordEmbeddingModel;

    pub use super::{
        cosine_similarity,
        one_hot_tensor
    };

    pub use super::{
        EMBEDDING_MAX_TOKENS,
        EMBEDDING_CONTEXT_RADIUS,
        EMBEDDING_SIZE
    };
}

/// Calculate cosine similarity between two vectors.
///
/// Return value in `[-1.0, 1.0]` range where 1.0 means fully equal.
pub fn cosine_similarity<const N: usize>(word_1: &[f32], word_2: &[f32]) -> f32 {
    let mut distance = 0.0;
    let mut len_1 = 0.0;
    let mut len_2 = 0.0;

    for i in 0..N {
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
