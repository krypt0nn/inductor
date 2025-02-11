use std::path::Path;

use burn::prelude::*;

use burn::nn::{Linear, LinearConfig};
use burn::nn::Initializer;
use burn::nn::loss::{MseLoss, Reduction};
use burn::tensor::backend::AutodiffBackend;
use burn::train::{TrainStep, TrainOutput, ValidStep, MultiLabelClassificationOutput};
use burn::record::{BinGzFileRecorder, FullPrecisionSettings};

use crate::prelude::*;

#[derive(Debug, Module)]
pub struct WordEmbeddingModel<B: Backend> {
    encoder: Linear<B>,
    decoder: Linear<B>
}

impl<B: Backend> WordEmbeddingModel<B> {
    /// Build new model with random weights.
    pub fn random(device: &B::Device) -> Self {
        Self {
            encoder: LinearConfig::new(EMBEDDING_MAX_TOKENS, EMBEDDING_SIZE)
                .with_bias(false)
                .with_initializer(Initializer::Uniform { min: -1.0, max: 1.0 })
                .init(device),

            decoder: LinearConfig::new(EMBEDDING_SIZE, EMBEDDING_MAX_TOKENS)
                .with_bias(false)
                .with_initializer(Initializer::Uniform { min: -1.0, max: 1.0 })
                .init(device)
        }
    }

    /// Save model to a file.
    pub fn save(self, file: impl AsRef<Path>) -> anyhow::Result<()> {
        let recorder = BinGzFileRecorder::<FullPrecisionSettings>::new();

        Ok(self.save_file(file.as_ref(), &recorder)?)
    }

    /// Load model from a file.
    pub fn load(file: impl AsRef<Path>, device: &B::Device) -> anyhow::Result<Self> {
        let recorder = BinGzFileRecorder::<FullPrecisionSettings>::new();

        Ok(Self::random(device).load_file(file.as_ref(), &recorder, device)?)
    }

    #[inline]
    /// Encode token into embedding tensor.
    pub fn encode(&self, token: usize, device: &B::Device) -> Tensor<B, 1, Float> {
        self.encoder.forward(one_hot_tensor(&[token], EMBEDDING_MAX_TOKENS, device))
    }

    fn forward_batch(&self, samples: WordEmbeddingTrainSamplesBatch<B>) -> MultiLabelClassificationOutput<B> {
        let embeddings = self.encoder.forward(samples.contexts);
        let predicted_targets = self.decoder.forward(embeddings);

        let loss = MseLoss::new().forward(
            predicted_targets.clone(),
            samples.targets.clone(),
            Reduction::Mean
        );

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
