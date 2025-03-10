use std::path::Path;

use burn::prelude::*;
use burn::nn::{Initializer, Linear, LinearConfig, Dropout, DropoutConfig};
use burn::nn::loss::{MseLoss, Reduction};
use burn::tensor::backend::AutodiffBackend;
use burn::train::{TrainStep, TrainOutput, ValidStep, RegressionOutput};
use burn::record::{BinGzFileRecorder, FullPrecisionSettings};

use crate::prelude::*;

#[derive(Debug, Module)]
pub struct TextGenerationModel<B: Backend> {
    encoder: Linear<B>,
    decoder: Linear<B>,
    dropout: Dropout,
    embedding_size: usize,
    context_tokens_num: usize,
    position_encoding_period: usize
}

impl<B: Backend> TextGenerationModel<B> {
    /// Build new model with random weights.
    pub fn random(
        embedding_size: usize,
        context_tokens_num: usize,
        position_encoding_period: usize,
        device: &B::Device
    ) -> Self {
        let input_window_size = embedding_size * context_tokens_num;

        Self {
            encoder: LinearConfig::new(input_window_size, input_window_size)
                .with_bias(true)
                .with_initializer(Initializer::XavierNormal { gain: 2.0 })
                .init(device),

            decoder: LinearConfig::new(input_window_size, embedding_size)
                .with_bias(true)
                .with_initializer(Initializer::XavierNormal { gain: 2.0 })
                .init(device),

            dropout: DropoutConfig::new(0.1).init(),

            embedding_size,
            context_tokens_num,
            position_encoding_period
        }
    }

    /// Save model to a file.
    pub fn save(self, file: impl AsRef<Path>) -> anyhow::Result<()> {
        let recorder = BinGzFileRecorder::<FullPrecisionSettings>::new();

        Ok(self.save_file(file.as_ref(), &recorder)?)
    }

    /// Load model from a file.
    pub fn load(
        embedding_size: usize,
        context_tokens_num: usize,
        position_encoding_period: usize,
        file: impl AsRef<Path>,
        device: &B::Device
    ) -> anyhow::Result<Self> {
        let recorder = BinGzFileRecorder::<FullPrecisionSettings>::new();

        let model = Self::random(embedding_size, context_tokens_num, position_encoding_period, device)
            .load_file(file.as_ref(), &recorder, device)?;

        Ok(model)
    }

    /// Get new tokens generation iterator.
    pub fn generate<'model>(&'model self, input: impl IntoIterator<Item = Tensor<B, 1, Float>>, device: &B::Device) -> TextGenerationIter<'model, B> {
        TextGenerationIter {
            model: self,
            history: (0..self.context_tokens_num)
                .map(|i| encode_position(Tensor::zeros([self.embedding_size], device), i, self.position_encoding_period))
                .chain({
                    input.into_iter()
                        .enumerate()
                        .map(|(i, tensor)| encode_position(tensor, self.context_tokens_num + i, self.position_encoding_period))
                })
                .collect()
        }
    }

    fn forward_batch(&self, samples: TextGeneratorTrainSamplesBatch<B>) -> RegressionOutput<B> {
        let hidden = self.encoder.forward(samples.contexts);
        let dropped_hidden = self.dropout.forward(hidden);
        let predicted = self.decoder.forward(dropped_hidden);

        let loss = MseLoss::new().forward(
            predicted.clone(),
            samples.targets.clone(),
            Reduction::Sum
        );

        RegressionOutput::new(loss, predicted, samples.targets)
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
    history: Vec<Tensor<B, 1, Float>>
}

impl<B: Backend> Iterator for TextGenerationIter<'_, B> {
    type Item = Tensor<B, 1, Float>;

    fn next(&mut self) -> Option<Self::Item> {
        let n = self.history.len();

        let context = Tensor::cat(self.history[n - self.model.context_tokens_num..].to_vec(), 0);

        let hidden = self.model.encoder.forward(context);
        let predicted = self.model.decoder.forward(hidden);

        self.history.push(encode_position(predicted.clone(), n, self.model.position_encoding_period));

        Some(predicted)
    }
}
