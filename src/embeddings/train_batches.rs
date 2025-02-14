use burn::prelude::*;
use burn::data::dataloader::batcher::Batcher;

use super::prelude::*;

#[derive(Debug, Clone)]
/// Batched word embedding train samples.
pub struct WordEmbeddingTrainSamplesBatch<B: Backend> {
    pub contexts: Tensor<B, 2, Float>,
    pub targets: Tensor<B, 2, Float>
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// Struct that combines different word embedding train samples together
/// to utilize parallel model training.
pub struct WordEmbeddingTrainSamplesBatcher;

impl<B: Backend> Batcher<WordEmbeddingTrainSample<B>, WordEmbeddingTrainSamplesBatch<B>> for WordEmbeddingTrainSamplesBatcher {
    fn batch(&self, mut items: Vec<WordEmbeddingTrainSample<B>>) -> WordEmbeddingTrainSamplesBatch<B> {
        let mut contexts = Vec::with_capacity(items.len());
        let mut targets = Vec::with_capacity(items.len());

        for item in items.drain(..) {
            contexts.push(item.context);
            targets.push(item.target);
        }

        WordEmbeddingTrainSamplesBatch {
            contexts: Tensor::stack(contexts, 0),
            targets: Tensor::stack(targets, 0)
        }
    }
}
