use std::path::{Path, PathBuf};

use serde::{Serialize, Deserialize};

use crate::prelude::*;

#[inline]
/// Try reading config from the given file.
pub fn load(path: impl AsRef<Path>) -> anyhow::Result<CliConfig> {
    let path = path.as_ref();

    let config = std::fs::read_to_string(path)?;
    let mut config = toml::from_str::<CliConfig>(&config)?;

    if let Some(config_folder) = path.parent() {
        if config.documents.database_path.is_relative() {
            config.documents.database_path = config_folder.join(config.documents.database_path);
        }

        if config.tokens.database_path.is_relative() {
            config.tokens.database_path = config_folder.join(config.tokens.database_path);
        }

        if config.embeddings.database_path.is_relative() {
            config.embeddings.database_path = config_folder.join(config.embeddings.database_path);
        }

        if config.embeddings.model_path.is_relative() {
            config.embeddings.model_path = config_folder.join(config.embeddings.model_path);
        }

        if config.embeddings.logs_path.is_relative() {
            config.embeddings.logs_path = config_folder.join(config.embeddings.logs_path);
        }

        if config.text_generator.model_path.is_relative() {
            config.text_generator.model_path = config_folder.join(config.text_generator.model_path);
        }

        if config.text_generator.logs_path.is_relative() {
            config.text_generator.logs_path = config_folder.join(config.text_generator.logs_path);
        }
    }

    Ok(config)
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CliConfig {
    pub documents: CliConfigDocuments,
    pub tokens: CliConfigTokens,
    pub embeddings: CliConfigEmbeddings,
    pub text_generator: CliConfigTextGenerator
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CliConfigDocuments {
    /// Path to the tokens sqlite database.
    pub database_path: PathBuf,

    /// SQLite database cache size.
    ///
    /// Positive value sets cache size in bytes, negative - in sqlite pages.
    pub ram_cache: i64,

    /// Convert content of the documents to lowercase.
    pub lowercase: bool
}

impl Default for CliConfigDocuments {
    #[inline]
    fn default() -> Self {
        Self {
            database_path: PathBuf::from("documents.db"),
            ram_cache: 1024 * 1024 * 128,
            lowercase: false
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CliConfigTokens {
    /// Path to the tokens sqlite database.
    pub database_path: PathBuf,

    /// SQLite database cache size.
    ///
    /// Positive value sets cache size in bytes, negative - in sqlite pages.
    pub ram_cache: i64,

    /// Convert content of the documents to lowercase.
    pub lowercase: bool,

    /// Strip punctuation from the documents.
    pub strip_punctuation: bool,

    /// Save whitespace characters as tokens.
    pub whitespace_tokens: bool
}

impl Default for CliConfigTokens {
    #[inline]
    fn default() -> Self {
        Self {
            database_path: PathBuf::from("tokens.db"),
            ram_cache: 1024 * 1024 * 16,
            lowercase: false,
            strip_punctuation: false,
            whitespace_tokens: false
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CliConfigEmbeddings {
    /// Path to the word embeddings sqlite database.
    pub database_path: PathBuf,

    /// Path to the word embeddings model.
    pub model_path: PathBuf,

    /// Path to the word embeddings model training logs.
    pub logs_path: PathBuf,

    /// SQLite database cache size.
    ///
    /// Positive value sets cache size in bytes, negative - in sqlite pages.
    pub ram_cache: i64,

    /// Method used to learn word embeddings.
    pub sampling_method: WordEmbeddingSamplingMethod,

    /// Maximal amount of tokens which can be encoded by the model.
    pub one_hot_tokens: usize,

    /// Amount of dimensions in a word embedding.
    pub embedding_size: usize,

    /// Amount or tokens to the left and right of the current one used to train the model.
    pub context_radius: usize,

    /// Skip tokens which occured less times than the specified amount.
    pub minimal_occurences: usize,

    /// Used to calculate probability of skipping word from training samples.
    pub subsampling_value: f64,

    /// Word embeddings model learning parameters.
    pub learning: CliConfigLearning
}

impl Default for CliConfigEmbeddings {
    #[inline]
    fn default() -> Self {
        Self {
            database_path: PathBuf::from("embeddings.db"),
            model_path: PathBuf::from("embeddings"),
            logs_path: PathBuf::from("embeddings-logs"),
            ram_cache: 1024 * 1024 * 64,
            sampling_method: WordEmbeddingSamplingMethod::default(),
            one_hot_tokens: 32768,
            embedding_size: 16,
            context_radius: 3,
            minimal_occurences: 2,
            subsampling_value: 1e-5,
            learning: CliConfigLearning::default()
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CliConfigTextGenerator {
    /// Path to the text generation model.
    pub model_path: PathBuf,

    /// Path to the text generation model training logs.
    pub logs_path: PathBuf,

    /// Amount of tokens used to predict the next one.
    pub context_tokens_num: usize,

    /// Amount of tokens after which position encoding will start repeating.
    ///
    /// If set to 0 no positional encoding is applied.
    pub position_encoding_period: usize,

    /// Maximal amount of tokens to generate.
    pub max_generated_tokens: usize,

    /// Text generation model learning parameters.
    pub learning: CliConfigLearning
}

impl Default for CliConfigTextGenerator {
    #[inline]
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("text-generator"),
            logs_path: PathBuf::from("text-generator-logs"),
            context_tokens_num: 16,
            position_encoding_period: 5000,
            max_generated_tokens: 512,
            learning: CliConfigLearning::default()
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CliConfigLearning {
    /// Number of epochs to train the model.
    pub epochs: usize,

    /// Initial learn rate of the model training.
    pub initial_learn_rate: f64,

    /// Final learn rate of the model training.
    pub final_learn_rate: f64,

    /// Amount of samples to train at one iteration. Increases memory use.
    pub batch_size: usize,

    /// Average last iterations before updating the model's weights.
    pub accumulate_gradients: usize,

    /// Amount of workers to prepare train and validation datasets in parallel.
    pub dataset_workers_num: usize,

    /// Addresses of remote devices used for training.
    pub remote_devices: Vec<String>
}

impl Default for CliConfigLearning {
    #[inline]
    fn default() -> Self {
        Self {
            epochs: 10,
            initial_learn_rate: 3e-2,
            final_learn_rate: 3e-5,
            batch_size: 64,
            accumulate_gradients: 2,
            dataset_workers_num: 4,
            remote_devices: vec![]
        }
    }
}
