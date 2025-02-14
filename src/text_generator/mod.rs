use burn::prelude::*;

pub mod train_samples;
pub mod train_batches;
pub mod model;

pub mod prelude {
    pub use super::train_samples::{
        TextGeneratorTrainSample,
        TextGeneratorTrainSamplesDataset
    };

    pub use super::train_batches::{
        TextGeneratorTrainSamplesBatch,
        TextGeneratorTrainSamplesBatcher
    };

    pub use super::model::TextGenerationModel;

    pub use super::encode_position;
}

/// Add sines to all the tensor's values.
pub fn encode_position<B: Backend>(tensor: Tensor<B, 1, Float>, position: usize, period: usize) -> Tensor<B, 1, Float> {
    // Don't change input tensor if position or period is zero.
    if period == 0 || position == 0 {
        return tensor;
    }

    const DOUBLE_PI: f32 = 2.0 * std::f32::consts::PI;

    // Length of the input tensor.
    let len = tensor.dims()[0];

    // Argument of the sin function repeated `len` times.
    //
    // sin(2pi * curr / total)
    let args = vec![DOUBLE_PI * position as f32 / period as f32; len];

    // Convert arguments from vector into tensor.
    let args = Tensor::from_floats(args.as_slice(), &tensor.device());

    // Get shifts of the sin function arguments.
    let args_shifts = Tensor::arange(1..(len as i64 + 1), &tensor.device()).float();

    // Apply position encoding to the input tensor.
    //
    // Encoding is calculated as:
    //
    // encoding[k] = sin(args[k] * args_shifts[k]) = sin(2pi * curr / total * (k + 1))
    tensor + (args * args_shifts).sin()
}
