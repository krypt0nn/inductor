use std::path::Path;

use burn::prelude::*;
use burn::nn::lstm::{Lstm, LstmConfig, LstmState};
use burn::nn::{Linear, LinearConfig};
use burn::nn::Initializer;
use burn::nn::loss::{MseLoss, Reduction};
use burn::tensor::backend::AutodiffBackend;
use burn::train::{TrainStep, TrainOutput, ValidStep, RegressionOutput};
use burn::record::{BinGzFileRecorder, FullPrecisionSettings};

use crate::prelude::*;

const INPUT_WINDOW_SIZE: usize = EMBEDDING_SIZE * TEXT_GENERATOR_CONTEXT_TOKENS_NUM;

#[derive(Debug, Module)]
pub struct TextGenerationModel<B: Backend> {
    encoder: Lstm<B>,
    decoder: Linear<B>
}

impl<B: Backend> TextGenerationModel<B> {
    /// Build new model with random weights.
    pub fn random(device: &B::Device) -> Self {
        Self {
            encoder: LstmConfig::new(INPUT_WINDOW_SIZE, INPUT_WINDOW_SIZE, true)
                .with_initializer(Initializer::XavierNormal { gain: 2.0 })
                .init(device),

            decoder: LinearConfig::new(INPUT_WINDOW_SIZE, EMBEDDING_SIZE)
                .with_bias(true)
                .with_initializer(Initializer::XavierNormal { gain: 2.0 })
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

    /// Get new tokens generation iterator.
    pub fn generate<'model>(&'model self, input: impl IntoIterator<Item = Tensor<B, 1, Float>>, device: &'model B::Device) -> TextGenerationIter<'model, B> {
        TextGenerationIter {
            model: self,
            device,

            history: input.into_iter()
                .map(|tensor| tensor.reshape([1, EMBEDDING_SIZE]))
                .collect(),

            state: None
        }
    }

    fn forward_batch(&self, samples: TextGeneratorTrainSamplesBatch<B>) -> RegressionOutput<B> {
        let (_, hidden) = self.encoder.forward(samples.contexts, None);

        let predicted_embedding = self.decoder.forward(hidden.hidden);

        let loss = MseLoss::new().forward(
            predicted_embedding.clone(),
            samples.targets.clone(),
            Reduction::Mean
        );

        RegressionOutput::new(loss, predicted_embedding, samples.targets)
    }
}

impl<B: AutodiffBackend> TrainStep<TextGeneratorTrainSamplesBatch<B>, RegressionOutput<B>> for TextGenerationModel<B> {
    fn step(&self, samples: TextGeneratorTrainSamplesBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let output = self.forward_batch(samples);

        TrainOutput::new(self, output.loss.backward(), output)
    }
}

impl<B: Backend> ValidStep<TextGeneratorTrainSamplesBatch<B>, RegressionOutput<B>> for TextGenerationModel<B> {
    #[inline]
    fn step(&self, samples: TextGeneratorTrainSamplesBatch<B>) -> RegressionOutput<B> {
        self.forward_batch(samples)
    }
}

pub struct TextGenerationIter<'model, B: Backend> {
    model: &'model TextGenerationModel<B>,
    device: &'model B::Device,
    history: Vec<Tensor<B, 2, Float>>,
    state: Option<LstmState<B, 2>>
}

impl<B: Backend> Iterator for TextGenerationIter<'_, B> {
    type Item = Tensor<B, 1, Float>;

    fn next(&mut self) -> Option<Self::Item> {
        let n = self.history.len();

        let context_window = if n < TEXT_GENERATOR_CONTEXT_TOKENS_NUM {
            let padding = Tensor::zeros([1, (TEXT_GENERATOR_CONTEXT_TOKENS_NUM - n) * EMBEDDING_SIZE], self.device);

            Tensor::cat(vec![padding, Tensor::cat(self.history[0..n].to_vec(), 0)], 0)
        }

        else {
            Tensor::cat(self.history[n - TEXT_GENERATOR_CONTEXT_TOKENS_NUM..n].to_vec(), 0)
        };

        let (hidden, state) = self.model.encoder.forward(context_window.reshape([1, 1, INPUT_WINDOW_SIZE]), self.state.take());
        let output = self.model.decoder.forward(hidden);

        self.history.push(output.reshape([1, EMBEDDING_SIZE]));
        self.state = Some(state);

        self.history.last()
            .cloned()
            .map(|tensor| tensor.reshape([EMBEDDING_SIZE]))
    }
}
