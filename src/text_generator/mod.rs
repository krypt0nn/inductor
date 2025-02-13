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
}
