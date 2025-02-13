use std::path::PathBuf;
use std::sync::Arc;
use std::io::Write;

use clap::Parser;
use colorful::Colorful;
use rand::SeedableRng;

use burn::prelude::*;
use burn::backend::{Autodiff, Wgpu, wgpu::WgpuDevice};
use burn::data::dataloader::{Dataset, DataLoaderBuilder};
use burn::data::dataset::transform::{ShuffledDataset, PartialDataset};
use burn::train::LearnerBuilder;
use burn::train::metric::{LossMetric, CpuUse, CpuMemory};
use burn::optim::AdamWConfig;
use burn::lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig;

type Backend = Wgpu;
type AutodiffBackend = Autodiff<Backend>;
type Device = WgpuDevice;

use crate::prelude::*;

#[derive(Parser)]
pub enum TextGeneratorCLI {
    /// Train text generation model on provided documents dataset.
    Train {
        #[arg(long, short)]
        /// Path to the instructed documents database.
        documents: PathBuf,

        #[arg(long, short)]
        /// Path to the word embeddings database.
        embeddings: PathBuf,

        #[arg(long, default_value_t = 1024 * 1024 * 32)]
        /// SQLite database cache size.
        ///
        /// Positive value sets cache size in bytes, negative - in sqlite pages.
        cache_size: i64,

        #[arg(long, short)]
        /// Convert content of the documents to lowercase.
        lowercase: bool,

        #[arg(long, short)]
        /// Strip punctuation from the documents.
        strip_punctuation: bool,

        #[arg(long, default_value_t = 10)]
        /// Number of epochs to train the word embeddings model.
        epochs: usize,

        #[arg(long, default_value_t = 0.0015)]
        /// Learn rate of the model training.
        learn_rate: f64
    },

    Generate {
        #[arg(long, short)]
        /// Path to the word embeddings database.
        embeddings: PathBuf,

        #[arg(long, default_value_t = 1024 * 1024 * 8)]
        /// SQLite database cache size.
        ///
        /// Positive value sets cache size in bytes, negative - in sqlite pages.
        cache_size: i64,

        #[arg(long, short)]
        /// Convert content of the documents to lowercase.
        lowercase: bool,

        #[arg(long, short)]
        /// Strip punctuation from the documents.
        strip_punctuation: bool,

        #[arg(long, short)]
        /// Context for which the model should generate the output.
        context: Option<String>,

        #[arg(long, short, default_value_t = 500)]
        /// Maximal amount of tokens to generate.
        max_tokens: usize
    }
}

impl TextGeneratorCLI {
    #[inline]
    pub fn execute(self, model: PathBuf) -> anyhow::Result<()> {
        match self {
            Self::Train { documents, embeddings, cache_size, lowercase, strip_punctuation, epochs, learn_rate } => {
                let documents = documents.canonicalize().unwrap_or(documents);
                let embeddings = embeddings.canonicalize().unwrap_or(embeddings);

                let model = model.canonicalize().unwrap_or(model);

                let model_folder = model.parent().ok_or_else(|| anyhow::anyhow!("Failed to get parent folder of the model path"))?;
                let model_name = model.file_name().ok_or_else(|| anyhow::anyhow!("Failed to get model file name"))?.to_string_lossy();

                let model_logs_folder = model_folder.join(format!("{model_name}-logs"));

                if !model_folder.exists() {
                    std::fs::create_dir_all(model_folder)?;
                } else if model_logs_folder.exists() {
                    std::fs::remove_dir_all(&model_logs_folder)?;
                }

                println!("⏳ Opening documents database in {documents:?}...");

                let documents = match DocumentsDatabase::open(&documents, cache_size) {
                    Ok(documents) => Arc::new(Box::new(documents) as Box<dyn Dataset<Document>>),
                    Err(err) => {
                        eprintln!("{}", format!("🧯 Failed to open documents database: {err}").red());

                        return Ok(());
                    }
                };

                println!("⏳ Opening word embeddings database in {embeddings:?}...");

                let embeddings = match WordEmbeddingsDatabase::open(&embeddings, cache_size) {
                    Ok(embeddings) => Arc::new(embeddings),
                    Err(err) => {
                        eprintln!("{}", format!("🧯 Failed to open word embeddings database: {err}").red());

                        return Ok(());
                    }
                };

                println!("⏳ Preparing training datasets...");

                let parser = DocumentsParser::new(lowercase, strip_punctuation);
                let device = Device::default();

                // Backend::seed(fastrand::u64(..));
                // AutodiffBackend::seed(fastrand::u64(..));

                let mut rng = rand::rngs::StdRng::seed_from_u64(fastrand::u64(..));

                let train_samples_dataset = TextGeneratorTrainSamplesDataset::<AutodiffBackend>::new(
                    documents.clone(),
                    embeddings.clone(),
                    parser,
                    device.clone()
                );

                let validate_samples_dataset = TextGeneratorTrainSamplesDataset::<Backend>::new(
                    documents.clone(),
                    embeddings.clone(),
                    parser,
                    device.clone()
                );

                let train_samples_dataset = ShuffledDataset::new(train_samples_dataset, &mut rng);
                let validate_samples_dataset = ShuffledDataset::new(validate_samples_dataset, &mut rng);

                let validate_dataset_len = (train_samples_dataset.len() as f32 * 0.15) as usize;

                let validate_samples_dataset = PartialDataset::new(validate_samples_dataset, 0, validate_dataset_len);

                let train_samples_dataset = DataLoaderBuilder::new(TextGeneratorTrainSamplesBatcher)
                    .num_workers(4)
                    .batch_size(16)
                    .build(train_samples_dataset);

                let validate_samples_dataset = DataLoaderBuilder::new(TextGeneratorTrainSamplesBatcher)
                    .num_workers(4)
                    .batch_size(16)
                    .build(validate_samples_dataset);

                println!("⏳ Opening the model...");

                let text_generation_model = TextGenerationModel::<AutodiffBackend>::load(&model, &device)
                    .unwrap_or_else(|_| TextGenerationModel::<AutodiffBackend>::random(&device));

                println!("⏳ Training the model...");

                let learner = LearnerBuilder::new(model_logs_folder)
                    // .metric_train_numeric(AccuracyMetric::new())
                    // .metric_valid_numeric(AccuracyMetric::new())
                    .metric_train_numeric(LossMetric::new())
                    .metric_valid_numeric(LossMetric::new())
                    .metric_train_numeric(CpuUse::new())
                    .metric_valid_numeric(CpuUse::new())
                    .metric_train_numeric(CpuMemory::new())
                    .metric_valid_numeric(CpuMemory::new())
                    .devices(vec![device.clone()])
                    .grads_accumulation(4)
                    .num_epochs(epochs)
                    .summary()
                    .build(
                        text_generation_model,
                        AdamWConfig::new().init(),
                        CosineAnnealingLrSchedulerConfig::new(learn_rate, 10).init().unwrap()
                    );

                let text_generation_model = learner.fit(train_samples_dataset, validate_samples_dataset);

                println!("{}", "✅ Model trained".green());
                println!("⏳ Saving the model...");

                text_generation_model.save(model)?;

                println!("{}", "✅ Model saved".green());
            }

            Self::Generate { embeddings, cache_size, lowercase, strip_punctuation, context, max_tokens } => {
                let embeddings = embeddings.canonicalize().unwrap_or(embeddings);

                let model = model.canonicalize().unwrap_or(model);

                let model_folder = model.parent().ok_or_else(|| anyhow::anyhow!("Failed to get parent folder of the model path"))?;
                let model_name = model.file_name().ok_or_else(|| anyhow::anyhow!("Failed to get model file name"))?.to_string_lossy();

                let model_logs_folder = model_folder.join(format!("{model_name}-logs"));

                if !model_folder.exists() {
                    std::fs::create_dir_all(model_folder)?;
                } else if model_logs_folder.exists() {
                    std::fs::remove_dir_all(&model_logs_folder)?;
                }

                println!("⏳ Opening word embeddings database in {embeddings:?}...");

                let embeddings = match WordEmbeddingsDatabase::open(&embeddings, cache_size) {
                    Ok(embeddings) => Arc::new(embeddings),
                    Err(err) => {
                        eprintln!("{}", format!("🧯 Failed to open word embeddings database: {err}").red());

                        return Ok(());
                    }
                };

                println!("⏳ Opening the model...");

                let parser = DocumentsParser::new(lowercase, strip_punctuation);
                let device = Device::default();

                // Backend::seed(fastrand::u64(..));
                // AutodiffBackend::seed(fastrand::u64(..));

                let text_generation_model = TextGenerationModel::<Backend>::load(&model, &device)
                    .unwrap_or_else(|_| TextGenerationModel::<Backend>::random(&device));

                let stdin = std::io::stdin();
                let mut stdout = std::io::stdout();

                stdout.write_all(b"\n")?;
                stdout.flush()?;

                loop {
                    stdout.write_all(format!("{} ", "Input:".yellow()).as_bytes())?;
                    stdout.flush()?;

                    let mut line = String::new();

                    stdin.read_line(&mut line)?;

                    stdout.write_all(b"\n")?;
                    stdout.flush()?;

                    let mut document = Document::new(line);

                    if let Some(context) = &context {
                        document = document.with_context(context);
                    }

                    let input_embeddings = parser.read_document(document)
                        .filter(|token| token != DocumentsParser::OUTPUT_CLOSE_TAG)
                        .map(|token| {
                            match embeddings.query_embedding(token) {
                                Ok(Some(embedding)) => Ok(Some(Tensor::from_floats(embedding.as_slice(), &device))),
                                Ok(None) => Ok(None),
                                Err(err) => Err(err)
                            }
                        })
                        .collect::<anyhow::Result<Option<Vec<Tensor<Backend, 1, Float>>>>>();

                    let Some(input_embeddings) = input_embeddings? else {
                        stdout.write_all("📖 Some input word is not indexed\n\n".as_bytes())?;
                        stdout.flush()?;

                        continue;
                    };

                    for (i, output_embedding) in text_generation_model.generate(input_embeddings, &device).enumerate() {
                        let output_embedding = output_embedding.into_data();

                        let output_embedding = output_embedding.as_slice().map_err(|err| {
                            anyhow::anyhow!("Failed to cast generated tensor into embedding vector: {err:?}")
                        })?;

                        let output_token = embeddings.find_token(output_embedding)?
                            .ok_or_else(|| anyhow::anyhow!("Failed to find token from the embedding vector"))?;

                        if output_token == DocumentsParser::OUTPUT_CLOSE_TAG {
                            break;
                        }

                        stdout.write_all(output_token.as_bytes())?;
                        stdout.flush()?;

                        if i >= max_tokens {
                            break;
                        }
                    }

                    stdout.write_all(b"\n\n")?;
                    stdout.flush()?;
                }
            }
        }

        Ok(())
    }
}
