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
    fn batch(&self, items: Vec<WordEmbeddingTrainSample<B>>) -> WordEmbeddingTrainSamplesBatch<B> {
        let (contexts, targets) = items.into_iter()
            .map(|sample| {
                let context = sample.context.reshape([1, -1]);
                let target = sample.target.reshape([1, -1]);

                (context, target)
            })
            .collect::<(Vec<_>, Vec<_>)>();

        WordEmbeddingTrainSamplesBatch {
            contexts: Tensor::<B, 2, Float>::cat(contexts, 0),
            targets: Tensor::<B, 2, Float>::cat(targets, 0)
        }
    }
}
