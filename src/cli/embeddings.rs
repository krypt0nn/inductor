use std::path::PathBuf;
use std::fs::File;
use std::io::{BufWriter, Write};

use clap::Parser;
use colorful::Colorful;
use rand::SeedableRng;

use burn::prelude::*;
use burn::backend::{Autodiff, Wgpu, wgpu::WgpuDevice, RemoteBackend, remote::RemoteDevice};
use burn::data::dataloader::{Dataset, DataLoaderBuilder};
use burn::data::dataset::transform::{ComposedDataset, ShuffledDataset, PartialDataset};
use burn::train::LearnerBuilder;
use burn::train::metric::{LossMetric, CpuUse, CpuMemory};
use burn::optim::AdamWConfig;
use burn::lr_scheduler::linear::LinearLrSchedulerConfig;

use crate::prelude::*;

#[derive(Parser)]
pub enum EmbeddingsCli {
    /// Train word embeddings model on provided documents dataset.
    Train,

    /// Update embeddings for all tokens from the database using provided model.
    Update,

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

impl EmbeddingsCli {
    #[inline]
    pub fn execute(self, config: super::config::CliConfig) -> anyhow::Result<()> {
        match self {
            Self::Train => {
                if config.embeddings.logs_path.exists() {
                    std::fs::remove_dir_all(&config.embeddings.logs_path)?;
                }

                println!("⏳ Opening documents database in {:?}...", config.documents.database_path);

                let documents = match DocumentsDatabase::open(&config.documents.database_path, config.documents.ram_cache) {
                    Ok(documents) => documents,
                    Err(err) => {
                        eprintln!("{}", format!("🧯 Failed to open documents database: {err}").red());

                        return Ok(());
                    }
                };

                println!("⏳ Opening tokens database in {:?}...", config.tokens.database_path);

                let tokens = match TokensDatabase::open(&config.tokens.database_path, config.tokens.ram_cache) {
                    Ok(tokens) => tokens,
                    Err(err) => {
                        eprintln!("{}", format!("🧯 Failed to open tokens database: {err}").red());

                        return Ok(());
                    }
                };

                println!("⏳ Opening word embeddings database in {:?}...", config.embeddings.database_path);

                let embeddings = match WordEmbeddingsDatabase::open(&config.embeddings.database_path, config.embeddings.ram_cache) {
                    Ok(embeddings) => embeddings,
                    Err(err) => {
                        eprintln!("{}", format!("🧯 Failed to open word embeddings database: {err}").red());

                        return Ok(());
                    }
                };

                let parser = DocumentsParser::new(
                    config.tokens.lowercase,
                    config.tokens.strip_punctuation,
                    config.tokens.whitespace_tokens
                );

                struct TrainParams<B: Backend> {
                    pub documents: DocumentsDatabase,
                    pub tokens: TokensDatabase,
                    pub embeddings: WordEmbeddingsDatabase,
                    pub parser: DocumentsParser,
                    pub config: super::config::CliConfig,
                    pub devices: Vec<B::Device>
                }

                fn train<B: Backend>(mut params: TrainParams<B>) -> anyhow::Result<()> {
                    let device = params.devices.first()
                        .cloned()
                        .ok_or_else(|| anyhow::anyhow!("No devices supplied"))?;

                    println!("⏳ Preparing training datasets...");

                    let mut train_samples_dataset = Vec::new();
                    let mut validate_samples_dataset = Vec::new();

                    let sampling_params = WordEmbeddingSamplingParams {
                        sampling_method: params.config.embeddings.sampling_method,
                        one_hot_tokens: params.config.embeddings.one_hot_tokens,
                        context_radius: params.config.embeddings.context_radius,
                        min_occurences: params.config.embeddings.minimal_occurences,
                        subsample_value: params.config.embeddings.subsampling_value
                    };

                    params.documents.for_each(|document| {
                        let train_dataset = WordEmbeddingsTrainSamplesDataset::<Autodiff<B>>::from_document(
                            document.clone(),
                            &params.parser,
                            &mut params.tokens,
                            device.clone(),
                            sampling_params
                        )?;

                        let validate_dataset = WordEmbeddingsTrainSamplesDataset::<B>::from_document(
                            document,
                            &params.parser,
                            &mut params.tokens,
                            device.clone(),
                            sampling_params
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

                    let validate_dataset_len = std::cmp::min((train_samples_dataset.len() as f32 * 0.15) as usize, 10000);

                    let validate_samples_dataset = PartialDataset::new(validate_samples_dataset, 0, validate_dataset_len);

                    let train_samples_dataset = DataLoaderBuilder::new(WordEmbeddingTrainSamplesBatcher)
                        .num_workers(params.config.embeddings.learning.dataset_workers_num)
                        .batch_size(params.config.embeddings.learning.batch_size)
                        .build(train_samples_dataset);

                    let validate_samples_dataset = DataLoaderBuilder::new(WordEmbeddingTrainSamplesBatcher)
                        .num_workers(params.config.embeddings.learning.dataset_workers_num)
                        .batch_size(params.config.embeddings.learning.batch_size)
                        .build(validate_samples_dataset);

                    println!("⏳ Opening the model...");

                    let embeddings_model = WordEmbeddingModel::<Autodiff<B>>::load(
                        params.config.embeddings.one_hot_tokens,
                        params.config.embeddings.embedding_size,
                        &params.config.embeddings.model_path,
                        &device
                    ).unwrap_or_else(|_| WordEmbeddingModel::<Autodiff<B>>::random(
                        params.config.embeddings.one_hot_tokens,
                        params.config.embeddings.embedding_size,
                        &device
                    ));

                    println!("⏳ Training the model...");

                    let learner = LearnerBuilder::new(params.config.embeddings.logs_path)
                        // .metric_train_numeric(AccuracyMetric::new())
                        // .metric_valid_numeric(AccuracyMetric::new())
                        .metric_train_numeric(LossMetric::new())
                        .metric_valid_numeric(LossMetric::new())
                        .metric_train_numeric(CpuUse::new())
                        .metric_valid_numeric(CpuUse::new())
                        .metric_train_numeric(CpuMemory::new())
                        .metric_valid_numeric(CpuMemory::new())
                        .devices(params.devices)
                        .grads_accumulation(params.config.embeddings.learning.accumulate_gradients)
                        .num_epochs(params.config.embeddings.learning.epochs)
                        .build(
                            embeddings_model,
                            AdamWConfig::new().init(),
                            LinearLrSchedulerConfig::new(
                                params.config.embeddings.learning.initial_learn_rate,
                                params.config.embeddings.learning.final_learn_rate,
                                params.config.embeddings.learning.epochs
                            ).init().unwrap()
                        );

                    let embeddings_model = learner.fit(train_samples_dataset, validate_samples_dataset);

                    println!("{}", "✅ Model trained".green());
                    println!("⏳ Updating token embeddings...");

                    let tokens = params.tokens.for_each(|token| {
                        let embedding = embeddings_model.encode(token.id as usize, &device)
                            .to_data();

                        let embedding = embedding.as_slice().map_err(|err| anyhow::anyhow!("Failed to cast tensor into floats slice: {err:?}"))?;

                        params.embeddings.insert_embedding(token.value, embedding)
                    })?;

                    println!("✅ Updated {} embeddings", tokens.to_string().yellow());
                    println!("⏳ Saving the model...");

                    embeddings_model.save(params.config.embeddings.model_path)?;

                    println!("{}", "✅ Model saved".green());

                    Ok(())
                }

                let result = if config.embeddings.learning.remote_devices.is_empty() {
                    train::<Wgpu>(TrainParams {
                        documents,
                        tokens,
                        embeddings,
                        parser,
                        config,
                        devices: vec![WgpuDevice::default()]
                    })
                }

                else {
                    train::<RemoteBackend>(TrainParams {
                        documents,
                        tokens,
                        embeddings,
                        parser,

                        devices: config.embeddings.learning.remote_devices.iter()
                            .map(|url| RemoteDevice::new(url))
                            .collect(),

                        config
                    })
                };

                if let Err(err) = result {
                    eprintln!("{}", format!("🧯 Failed to train the model: {err}").red());
                }
            }

            Self::Update => {
                println!("⏳ Opening tokens database in {:?}...", config.tokens.database_path);

                let tokens = match TokensDatabase::open(&config.tokens.database_path, config.tokens.ram_cache) {
                    Ok(tokens) => tokens,
                    Err(err) => {
                        eprintln!("{}", format!("🧯 Failed to open tokens database: {err}").red());

                        return Ok(());
                    }
                };

                println!("⏳ Opening word embeddings database in {:?}...", config.embeddings.database_path);

                let embeddings = match WordEmbeddingsDatabase::open(&config.embeddings.database_path, config.embeddings.ram_cache) {
                    Ok(embeddings) => embeddings,
                    Err(err) => {
                        eprintln!("{}", format!("🧯 Failed to open word embeddings database: {err}").red());

                        return Ok(());
                    }
                };

                println!("⏳ Opening the model...");

                let device = WgpuDevice::default();

                let embeddings_model = WordEmbeddingModel::<Autodiff<Wgpu>>::load(
                    config.embeddings.one_hot_tokens,
                    config.embeddings.embedding_size,
                    &config.embeddings.model_path,
                    &device
                ).unwrap_or_else(|_| WordEmbeddingModel::<Autodiff<Wgpu>>::random(
                    config.embeddings.one_hot_tokens,
                    config.embeddings.embedding_size,
                    &device
                ));

                println!("⏳ Updating token embeddings...");

                let tokens = tokens.for_each(|token| {
                    let embedding = embeddings_model.encode(token.id as usize, &device)
                        .to_data();

                    let embedding = embedding.as_slice().map_err(|err| anyhow::anyhow!("Failed to cast tensor into floats slice: {err:?}"))?;

                    embeddings.insert_embedding(token.value, embedding)
                })?;

                println!("✅ Updated {} embeddings", tokens.to_string().yellow());
            }

            Self::Compare { top_n } => {
                println!("⏳ Opening word embeddings database in {:?}...", config.embeddings.database_path);

                let embeddings = match WordEmbeddingsDatabase::open(&config.embeddings.database_path, config.embeddings.ram_cache) {
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
                        best_tokens.push((token, cosine_similarity(&target_embedding, &embedding)));

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
                let csv = csv.canonicalize().unwrap_or(csv);

                println!("⏳ Opening word embeddings database in {:?}...", config.embeddings.database_path);

                let embeddings = match WordEmbeddingsDatabase::open(&config.embeddings.database_path, config.embeddings.ram_cache) {
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
