use burn::prelude::*;
use burn::data::dataloader::batcher::Batcher;

use super::prelude::*;

#[derive(Debug, Clone)]
pub struct TextGeneratorTrainSamplesBatch<B: Backend> {
    pub contexts: Tensor<B, 3, Float>,
    pub targets: Tensor<B, 2, Float>
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextGeneratorTrainSamplesBatcher;

impl<B: Backend> Batcher<TextGeneratorTrainSample<B>, TextGeneratorTrainSamplesBatch<B>> for TextGeneratorTrainSamplesBatcher {
    fn batch(&self, items: Vec<TextGeneratorTrainSample<B>>) -> TextGeneratorTrainSamplesBatch<B> {
        let mut max_sequence_batch_len = 0;

        let (contexts, targets) = items.into_iter()
            .map(|sample| {
                let [context_windows_num, context_window_size] = sample.context.dims();

                max_sequence_batch_len = std::cmp::max(max_sequence_batch_len, context_windows_num);

                let context = sample.context.reshape([context_windows_num, context_window_size]);
                let target = sample.target.reshape([1, -1]);

                (context, target)
            })
            .collect::<(Vec<_>, Vec<_>)>();

        let contexts = contexts.into_iter()
            .map(|context| {
                let [context_windows_num, context_window_size] = context.dims();

                let padded_context = if max_sequence_batch_len > context_windows_num {
                    Tensor::cat(vec![
                        Tensor::zeros([max_sequence_batch_len - context_windows_num, context_window_size], &context.device()),
                        context
                    ], 0)
                } else {
                    context
                };

                padded_context.reshape([1, max_sequence_batch_len, context_window_size])
            })
            .collect::<Vec<_>>();

        TextGeneratorTrainSamplesBatch {
            contexts: Tensor::<B, 3, Float>::cat(contexts, 0),
            targets: Tensor::<B, 2, Float>::cat(targets, 0)
        }
    }
}
