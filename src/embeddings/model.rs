use std::path::Path;

use burn::prelude::*;

use burn::nn::{Initializer, Linear, LinearConfig};
use burn::nn::loss::{MseLoss, Reduction};
use burn::tensor::backend::AutodiffBackend;
use burn::train::{TrainStep, TrainOutput, ValidStep, RegressionOutput};
use burn::record::{BinGzFileRecorder, FullPrecisionSettings};

use crate::prelude::*;

#[derive(Debug, Module)]
pub struct WordEmbeddingModel<B: Backend> {
    encoder: Linear<B>,
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

    fn forward_batch(&self, samples: WordEmbeddingTrainSamplesBatch<B>) -> RegressionOutput<B> {
        let predicted_embeddings = self.encoder.forward(samples.contexts);
        let target_embeddings = self.encoder.forward(samples.targets);

        let loss = MseLoss::new().forward(
            predicted_embeddings.clone(),
            target_embeddings.clone(),
            Reduction::Sum
        );

        RegressionOutput::new(loss, predicted_embeddings, target_embeddings)
    }
}

impl<B: AutodiffBackend> TrainStep<WordEmbeddingTrainSamplesBatch<B>, RegressionOutput<B>> for WordEmbeddingModel<B> {
    fn step(&self, samples: WordEmbeddingTrainSamplesBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let output = self.forward_batch(samples);

        TrainOutput::new(self, output.loss.backward(), output)
    }
}

impl<B: Backend> ValidStep<WordEmbeddingTrainSamplesBatch<B>, RegressionOutput<B>> for WordEmbeddingModel<B> {
    #[inline]
    fn step(&self, samples: WordEmbeddingTrainSamplesBatch<B>) -> RegressionOutput<B> {
        self.forward_batch(samples)
    }
}
