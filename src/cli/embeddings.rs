use std::path::PathBuf;
use std::fs::File;
use std::io::{BufWriter, Write};

use clap::Parser;
use colorful::Colorful;
use rand::SeedableRng;

use burn::backend::{Autodiff, Wgpu, wgpu::WgpuDevice};
use burn::data::dataloader::{Dataset, DataLoaderBuilder};
use burn::data::dataset::transform::{ComposedDataset, ShuffledDataset, PartialDataset};
use burn::train::LearnerBuilder;
use burn::train::metric::{LossMetric, CpuUse, CpuMemory};
use burn::optim::AdamWConfig;
use burn::lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig;

type Backend = Wgpu;
type AutodiffBackend = Autodiff<Backend>;
type Device = WgpuDevice;

use crate::prelude::*;

#[derive(Parser)]
pub enum EmbeddingsCLI {
    /// Create new word embeddings database.
    Create,

    /// Train word embeddings model on provided documents dataset.
    Train {
        #[arg(long, short)]
        /// Path to the documents database.
        documents: PathBuf,

        #[arg(long, short)]
        /// Convert content of the documents to lowercase.
        lowercase: bool,

        #[arg(long, short)]
        /// Strip punctuation from the documents.
        strip_punctuation: bool,

        #[arg(long, short)]
        /// Save whitespace characters as tokens.
        whitespace_tokens: bool,

        #[arg(long, short)]
        /// Path to the word tokens database.
        tokens: PathBuf,

        #[arg(long, short)]
        /// Path to the word embeddings model.
        model: PathBuf,

        #[arg(long, short, default_value_t = 10)]
        /// Number of epochs to train the word embeddings model.
        epochs: usize,

        #[arg(long, default_value_t = 0.0015)]
        /// Learn rate of the model training.
        learn_rate: f64
    },

    /// Update embeddings for all tokens from the database using provided model.
    Update {
        #[arg(long, short)]
        /// Path to the word tokens database.
        tokens: PathBuf,

        #[arg(long, short)]
        /// Path to the word embeddings model.
        model: PathBuf
    },

    /// Compare words to each other using their embeddings.
    Compare {
        #[arg(long, short, default_value_t = 10)]
        /// Amount of closest tokens to return.
        top_n: usize
    },

    /// Export word embeddings into a CSV file.
    Export {
        #[arg(long, short)]
        /// Path to the CSV file.
        csv: PathBuf
    }
}

impl EmbeddingsCLI {
    #[inline]
    pub fn execute(self, database: PathBuf, cache_size: i64) -> anyhow::Result<()> {
        match self {
            Self::Create => {
                let database = database.canonicalize().unwrap_or(database);

                println!("⏳ Creating word embeddings database in {database:?}...");

                match WordEmbeddingsDatabase::open(&database, cache_size) {
                    Ok(_) => println!("{}", "🚀 Database created".green()),
                    Err(err) => eprintln!("{}", format!("🧯 Failed to create database: {err}").red())
                }
            }

            Self::Train { documents, lowercase, strip_punctuation, whitespace_tokens, tokens, model, epochs, learn_rate } => {
                let embeddings = database.canonicalize().unwrap_or(database);
                let documents = documents.canonicalize().unwrap_or(documents);
                let tokens = tokens.canonicalize().unwrap_or(tokens);
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
                    Ok(embeddings) => embeddings,
                    Err(err) => {
                        eprintln!("{}", format!("🧯 Failed to open word embeddings database: {err}").red());

                        return Ok(());
                    }
                };

                println!("⏳ Opening documents database in {documents:?}...");

                let documents = match DocumentsDatabase::open(&documents, cache_size) {
                    Ok(documents) => documents,
                    Err(err) => {
                        eprintln!("{}", format!("🧯 Failed to open documents database: {err}").red());

                        return Ok(());
                    }
                };

                println!("⏳ Opening tokens database in {tokens:?}...");

                let tokens = match TokensDatabase::open(&tokens, cache_size) {
                    Ok(tokens) => tokens,
                    Err(err) => {
                        eprintln!("{}", format!("🧯 Failed to open tokens database: {err}").red());

                        return Ok(());
                    }
                };

                println!("⏳ Preparing training datasets...");

                let parser = DocumentsParser::new(lowercase, strip_punctuation, whitespace_tokens);
                let device = Device::default();

                // Backend::seed(fastrand::u64(..));
                // AutodiffBackend::seed(fastrand::u64(..));

                let mut train_samples_dataset = Vec::new();
                let mut validate_samples_dataset = Vec::new();

                documents.for_each(|document| {
                    let train_dataset = WordEmbeddingsTrainSamplesDataset::<AutodiffBackend>::from_document(
                        document.clone(),
                        &parser,
                        &tokens,
                        device.clone()
                    )?;

                    let validate_dataset = WordEmbeddingsTrainSamplesDataset::<Backend>::from_document(
                        document,
                        &parser,
                        &tokens,
                        device.clone()
                    )?;

                    train_samples_dataset.push(train_dataset);
                    validate_samples_dataset.push(validate_dataset);

                    Ok(())
                })?;

                let mut rng = rand::rngs::StdRng::seed_from_u64(fastrand::u64(..));

                let train_samples_dataset = ComposedDataset::new(train_samples_dataset);
                let train_samples_dataset = ShuffledDataset::new(train_samples_dataset, &mut rng);

                let validate_samples_dataset = ComposedDataset::new(validate_samples_dataset);
                let validate_samples_dataset = ShuffledDataset::new(validate_samples_dataset, &mut rng);

                let validate_dataset_len = (train_samples_dataset.len() as f32 * 0.15) as usize;

                let validate_samples_dataset = PartialDataset::new(validate_samples_dataset, 0, validate_dataset_len);

                let train_samples_dataset = DataLoaderBuilder::new(WordEmbeddingTrainSamplesBatcher)
                    .num_workers(4)
                    .batch_size(32)
                    .build(train_samples_dataset);

                let validate_samples_dataset = DataLoaderBuilder::new(WordEmbeddingTrainSamplesBatcher)
                    .num_workers(4)
                    .batch_size(32)
                    .build(validate_samples_dataset);

                println!("⏳ Opening the model...");

                let embeddings_model = WordEmbeddingModel::<AutodiffBackend>::load(&model, &device)
                    .unwrap_or_else(|_| WordEmbeddingModel::<AutodiffBackend>::random(&device));

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
                        embeddings_model,
                        AdamWConfig::new().init(),
                        CosineAnnealingLrSchedulerConfig::new(learn_rate, 10).init().unwrap()
                    );

                let embeddings_model = learner.fit(train_samples_dataset, validate_samples_dataset);

                println!("{}", "✅ Model trained".green());
                println!("⏳ Updating token embeddings...");

                let tokens = tokens.for_each(|token_id, token| {
                    let embedding = embeddings_model.encode(token_id as usize, &device)
                        .to_data();

                    let embedding = embedding.as_slice().map_err(|err| anyhow::anyhow!("Failed to cast tensor into floats slice: {err:?}"))?;

                    embeddings.insert_embedding(token, embedding)
                })?;

                println!("✅ Updated {} embeddings", tokens.to_string().yellow());
                println!("⏳ Saving the model...");

                embeddings_model.save(model)?;

                println!("{}", "✅ Model saved".green());
            }

            Self::Update { tokens, model } => {
                let embeddings = database.canonicalize().unwrap_or(database);
                let tokens = tokens.canonicalize().unwrap_or(tokens);
                let model = model.canonicalize().unwrap_or(model);

                println!("⏳ Opening word embeddings database in {embeddings:?}...");

                let embeddings = match WordEmbeddingsDatabase::open(&embeddings, cache_size) {
                    Ok(embeddings) => embeddings,
                    Err(err) => {
                        eprintln!("{}", format!("🧯 Failed to open word embeddings database: {err}").red());

                        return Ok(());
                    }
                };

                println!("⏳ Opening tokens database in {tokens:?}...");

                let tokens = match TokensDatabase::open(&tokens, cache_size) {
                    Ok(tokens) => tokens,
                    Err(err) => {
                        eprintln!("{}", format!("🧯 Failed to open tokens database: {err}").red());

                        return Ok(());
                    }
                };

                println!("⏳ Opening the model...");

                let device = Device::default();

                let embeddings_model = WordEmbeddingModel::<AutodiffBackend>::load(&model, &device)
                    .unwrap_or_else(|_| WordEmbeddingModel::<AutodiffBackend>::random(&device));

                println!("⏳ Updating token embeddings...");

                let tokens = tokens.for_each(|token_id, token| {
                    let embedding = embeddings_model.encode(token_id as usize, &device)
                        .to_data();

                    let embedding = embedding.as_slice().map_err(|err| anyhow::anyhow!("Failed to cast tensor into floats slice: {err:?}"))?;

                    embeddings.insert_embedding(token, embedding)
                })?;

                println!("✅ Updated {} embeddings", tokens.to_string().yellow());
            }

            Self::Compare { top_n } => {
                let embeddings = database.canonicalize().unwrap_or(database);

                println!("⏳ Opening word embeddings database in {embeddings:?}...");

                let embeddings = match WordEmbeddingsDatabase::open(&embeddings, cache_size) {
                    Ok(embeddings) => embeddings,
                    Err(err) => {
                        eprintln!("{}", format!("🧯 Failed to open word embeddings database: {err}").red());

                        return Ok(());
                    }
                };

                let stdin = std::io::stdin();
                let mut stdout = std::io::stdout();

                stdout.write_all(b"\n")?;
                stdout.flush()?;

                loop {
                    stdout.write_all(format!("{} ", "Word:".yellow()).as_bytes())?;
                    stdout.flush()?;

                    let mut line = String::new();

                    stdin.read_line(&mut line)?;

                    stdout.write_all(b"\n")?;
                    stdout.flush()?;

                    let Some(target_embedding) = embeddings.query_embedding(line.trim())? else {
                        stdout.write_all("📖 Word is not indexed\n\n".as_bytes())?;
                        stdout.flush()?;

                        continue;
                    };

                    let mut best_tokens = Vec::new();

                    embeddings.for_each(|token, embedding| {
                        best_tokens.push((token, cosine_similarity::<EMBEDDING_SIZE>(&target_embedding, &embedding)));

                        Ok(())
                    })?;

                    best_tokens.sort_by(|a, b| {
                        if b.1 > a.1 {
                            std::cmp::Ordering::Greater
                        } else if b.1 < a.1 {
                            std::cmp::Ordering::Less
                        } else {
                            std::cmp::Ordering::Equal
                        }
                    });

                    for (token, distance) in best_tokens.into_iter().take(top_n) {
                        stdout.write_all(format!("- {} [{distance:.08}]\n", format!("\"{token}\"").blue()).as_bytes())?;
                    }

                    stdout.write_all(b"\n")?;
                    stdout.flush()?;
                }
            }

            Self::Export { csv } => {
                let embeddings = database.canonicalize().unwrap_or(database);
                let csv = csv.canonicalize().unwrap_or(csv);

                println!("⏳ Opening word embeddings database in {embeddings:?}...");

                let embeddings = match WordEmbeddingsDatabase::open(&embeddings, cache_size) {
                    Ok(embeddings) => embeddings,
                    Err(err) => {
                        eprintln!("{}", format!("🧯 Failed to open word embeddings database: {err}").red());

                        return Ok(());
                    }
                };

                let mut file = match File::create(&csv) {
                    Ok(file) => BufWriter::new(file),
                    Err(err) => {
                        eprintln!("{}", format!("🧯 Failed to create csv file: {err}").red());

                        return Ok(());
                    }
                };

                println!("⏳ Exporting tokens into {csv:?}...");

                let mut has_header = false;

                let result = embeddings.for_each(|token, embedding| {
                    if let Some(first_char) = token.chars().next() {
                        if first_char.is_alphanumeric() || (first_char.is_ascii_punctuation() && !['"', '\\'].contains(&first_char)) {
                            if !has_header {
                                file.write_all(b"\"token\"")?;

                                for i in 1..=embedding.len() {
                                    file.write_all(format!(",\"embedding{i}\"").as_bytes())?;
                                }

                                file.write_all(b"\n")?;

                                has_header = true;
                            }

                            file.write_all(format!("\"{token}\"").as_bytes())?;

                            for value in embedding {
                                file.write_all(format!(",\"{value}\"").as_bytes())?;
                            }

                            file.write_all(b"\n")?;
                        }
                    }

                    file.flush()?;

                    Ok(())
                });

                match result {
                    Ok(tokens) => println!("✅ Exported {} tokens", tokens.to_string().yellow()),
                    Err(err) => eprintln!("{}", format!("🧯 Failed to export tokens: {err}").red())
                }
            }
        }

        Ok(())
    }
}
