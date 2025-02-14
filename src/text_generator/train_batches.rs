use burn::prelude::*;
use burn::data::dataloader::batcher::Batcher;

use super::prelude::*;

#[derive(Debug, Clone)]
pub struct TextGeneratorTrainSamplesBatch<B: Backend> {
    pub contexts: Tensor<B, 2, Float>,
    pub targets: Tensor<B, 2, Float>
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextGeneratorTrainSamplesBatcher;

impl<B: Backend> Batcher<TextGeneratorTrainSample<B>, TextGeneratorTrainSamplesBatch<B>> for TextGeneratorTrainSamplesBatcher {
    fn batch(&self, mut items: Vec<TextGeneratorTrainSample<B>>) -> TextGeneratorTrainSamplesBatch<B> {
        let mut contexts = Vec::with_capacity(items.len());
        let mut targets = Vec::with_capacity(items.len());

        for item in items.drain(..) {
            contexts.push(item.context);
            targets.push(item.target);
        }

        TextGeneratorTrainSamplesBatch {
            contexts: Tensor::stack(contexts, 0),
            targets: Tensor::stack(targets, 0)
        }
    }
}
