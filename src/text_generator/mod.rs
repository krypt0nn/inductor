pub mod train_samples;
pub mod train_batches;
pub mod model;

/// Amount of tokens used to generate the next one.
pub const TEXT_GENERATOR_CONTEXT_TOKENS_NUM: usize = 8;

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

    pub use super::TEXT_GENERATOR_CONTEXT_TOKENS_NUM;
}
