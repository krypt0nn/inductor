use std::path::Path;

use burn::prelude::*;

use burn::nn::{Linear, LinearConfig};
use burn::nn::Initializer;
use burn::nn::loss::BinaryCrossEntropyLossConfig;
use burn::tensor::backend::AutodiffBackend;
use burn::train::{TrainStep, TrainOutput, ValidStep, MultiLabelClassificationOutput};
use burn::record::{BinGzFileRecorder, FullPrecisionSettings};

use crate::prelude::*;

#[derive(Debug, Module)]
pub struct WordEmbeddingModel<B: Backend> {
    encoder: Linear<B>,
    decoder: Linear<B>,
    one_hot_tokens: usize,
    embedding_size: usize
}

impl<B: Backend> WordEmbeddingModel<B> {
    /// Build new model with random weights.
    pub fn random(one_hot_tokens: usize, embedding_size: usize, device: &B::Device) -> Self {
        Self {
            encoder: LinearConfig::new(one_hot_tokens, embedding_size)
                .with_bias(false)
                .with_initializer(Initializer::XavierUniform { gain: 2.0 })
                .init(device),

            decoder: LinearConfig::new(embedding_size, one_hot_tokens)
                .with_bias(false)
                .with_initializer(Initializer::XavierUniform { gain: 2.0 })
                .init(device),

            one_hot_tokens,
            embedding_size
        }
    }

    /// Save model to a file.
    pub fn save(self, file: impl AsRef<Path>) -> anyhow::Result<()> {
        let recorder = BinGzFileRecorder::<FullPrecisionSettings>::new();

        Ok(self.save_file(file.as_ref(), &recorder)?)
    }

    /// Load model from a file.
    pub fn load(one_hot_tokens: usize, embedding_size: usize, file: impl AsRef<Path>, device: &B::Device) -> anyhow::Result<Self> {
        let recorder = BinGzFileRecorder::<FullPrecisionSettings>::new();

        let model = Self::random(one_hot_tokens, embedding_size, device)
            .load_file(file.as_ref(), &recorder, device)?;

        Ok(model)
    }

    #[inline]
    /// Encode token into embedding tensor.
    pub fn encode(&self, token: usize, device: &B::Device) -> Tensor<B, 1, Float> {
        self.encoder.forward(one_hot_tensor(&[token], self.one_hot_tokens, device))
    }

    fn forward_batch(&self, samples: WordEmbeddingTrainSamplesBatch<B>) -> MultiLabelClassificationOutput<B> {
        let embeddings = self.encoder.forward(samples.contexts);
        let predicted_targets = self.decoder.forward(embeddings);

        let loss = BinaryCrossEntropyLossConfig::new()
            .with_logits(true)
            .init(&predicted_targets.device())
            .forward(predicted_targets.clone(), samples.targets.clone().int());

        MultiLabelClassificationOutput::new(loss, predicted_targets, samples.targets.int())
    }
}

impl<B: AutodiffBackend> TrainStep<WordEmbeddingTrainSamplesBatch<B>, MultiLabelClassificationOutput<B>> for WordEmbeddingModel<B> {
    fn step(&self, samples: WordEmbeddingTrainSamplesBatch<B>) -> TrainOutput<MultiLabelClassificationOutput<B>> {
        let output = self.forward_batch(samples);

        TrainOutput::new(self, output.loss.backward(), output)
    }
}

impl<B: Backend> ValidStep<WordEmbeddingTrainSamplesBatch<B>, MultiLabelClassificationOutput<B>> for WordEmbeddingModel<B> {
    #[inline]
    fn step(&self, samples: WordEmbeddingTrainSamplesBatch<B>) -> MultiLabelClassificationOutput<B> {
        self.forward_batch(samples)
    }
}
