use std::sync::Arc;
use std::io::Write;

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
pub enum TextGeneratorCli {
    /// Train text generation model on provided documents dataset.
    Train,

    Generate {
        #[arg(long)]
        /// Context for which the model should generate the output.
        context: Option<String>
    }
}

impl TextGeneratorCli {
    #[inline]
    pub fn execute(self, config: super::config::CliConfig) -> anyhow::Result<()> {
        match self {
            Self::Train => {
                if config.text_generator.logs_path.exists() {
                    std::fs::remove_dir_all(&config.text_generator.logs_path)?;
                }

                println!("⏳ Opening documents database in {:?}...", config.documents.database_path);

                let documents = match DocumentsDatabase::open(&config.documents.database_path, config.documents.ram_cache) {
                    Ok(documents) => documents,
                    Err(err) => {
                        eprintln!("{}", format!("🧯 Failed to open documents database: {err}").red());

                        return Ok(());
                    }
                };

                println!("⏳ Opening word embeddings database in {:?}...", config.embeddings.database_path);

                let embeddings = match WordEmbeddingsDatabase::open(&config.embeddings.database_path, config.embeddings.ram_cache) {
                    Ok(embeddings) => Arc::new(embeddings),
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
                    pub embeddings: Arc<WordEmbeddingsDatabase>,
                    pub parser: DocumentsParser,
                    pub config: super::config::CliConfig,
                    pub devices: Vec<B::Device>
                }

                fn train<B: Backend>(params: TrainParams<B>) -> anyhow::Result<()> {
                    let device = params.devices.first()
                        .cloned()
                        .ok_or_else(|| anyhow::anyhow!("No devices supplied"))?;

                    println!("⏳ Preparing training datasets...");

                    let mut train_samples_dataset = Vec::new();
                    let mut validate_samples_dataset = Vec::new();

                    params.documents.for_each(|document| {
                        let train_dataset = TextGeneratorTrainSamplesDataset::<Autodiff<B>>::from_document(
                            document.clone(),
                            &params.parser,
                            params.embeddings.clone(),
                            params.config.embeddings.embedding_size,
                            params.config.text_generator.context_tokens_num,
                            params.config.text_generator.position_encoding_period,
                            device.clone()
                        );

                        let validate_dataset = TextGeneratorTrainSamplesDataset::<B>::from_document(
                            document,
                            &params.parser,
                            params.embeddings.clone(),
                            params.config.embeddings.embedding_size,
                            params.config.text_generator.context_tokens_num,
                            params.config.text_generator.position_encoding_period,
                            device.clone()
                        );

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

                    let train_samples_dataset = DataLoaderBuilder::new(TextGeneratorTrainSamplesBatcher)
                        .num_workers(params.config.text_generator.learning.dataset_workers_num)
                        .batch_size(params.config.text_generator.learning.batch_size)
                        .build(train_samples_dataset);

                    let validate_samples_dataset = DataLoaderBuilder::new(TextGeneratorTrainSamplesBatcher)
                        .num_workers(params.config.text_generator.learning.dataset_workers_num)
                        .batch_size(params.config.text_generator.learning.batch_size)
                        .build(validate_samples_dataset);

                    println!("⏳ Opening the model...");

                    let text_generation_model = TextGenerationModel::<Autodiff<B>>::load(
                        params.config.embeddings.embedding_size,
                        params.config.text_generator.context_tokens_num,
                        params.config.text_generator.position_encoding_period,
                        &params.config.text_generator.model_path,
                        &device
                    ).unwrap_or_else(|_| TextGenerationModel::<Autodiff<B>>::random(
                        params.config.embeddings.embedding_size,
                        params.config.text_generator.context_tokens_num,
                        params.config.text_generator.position_encoding_period,
                        &device
                    ));

                    println!("⏳ Training the model...");

                    let learner = LearnerBuilder::new(&params.config.text_generator.model_path)
                        // .metric_train_numeric(AccuracyMetric::new())
                        // .metric_valid_numeric(AccuracyMetric::new())
                        .metric_train_numeric(LossMetric::new())
                        .metric_valid_numeric(LossMetric::new())
                        .metric_train_numeric(CpuUse::new())
                        .metric_valid_numeric(CpuUse::new())
                        .metric_train_numeric(CpuMemory::new())
                        .metric_valid_numeric(CpuMemory::new())
                        .devices(params.devices)
                        .grads_accumulation(params.config.text_generator.learning.accumulate_gradients)
                        .num_epochs(params.config.text_generator.learning.epochs)
                        .build(
                            text_generation_model,
                            AdamWConfig::new().init(),
                            LinearLrSchedulerConfig::new(
                                params.config.text_generator.learning.initial_learn_rate,
                                params.config.text_generator.learning.final_learn_rate,
                                params.config.text_generator.learning.epochs
                            ).init().unwrap()
                        );

                    let text_generation_model = learner.fit(train_samples_dataset, validate_samples_dataset);

                    println!("{}", "✅ Model trained".green());
                    println!("⏳ Saving the model...");

                    text_generation_model.save(params.config.text_generator.model_path)?;

                    println!("{}", "✅ Model saved".green());

                    Ok(())
                }

                let result = if config.text_generator.learning.remote_devices.is_empty() {
                    train::<Wgpu>(TrainParams {
                        documents,
                        embeddings,
                        parser,
                        config,
                        devices: vec![WgpuDevice::default()]
                    })
                }

                else {
                    train::<RemoteBackend>(TrainParams {
                        documents,
                        embeddings,
                        parser,

                        devices: config.text_generator.learning.remote_devices.iter()
                            .map(|url| RemoteDevice::new(url))
                            .collect(),

                        config
                    })
                };

                if let Err(err) = result {
                    eprintln!("{}", format!("🧯 Failed to train the model: {err}").red());
                }
            }

            Self::Generate { context } => {
                if config.text_generator.logs_path.exists() {
                    std::fs::remove_dir_all(&config.text_generator.logs_path)?;
                }

                println!("⏳ Opening word embeddings database in {:?}...", config.embeddings.database_path);

                let embeddings = match WordEmbeddingsDatabase::open(&config.embeddings.database_path, config.embeddings.ram_cache) {
                    Ok(embeddings) => Arc::new(embeddings),
                    Err(err) => {
                        eprintln!("{}", format!("🧯 Failed to open word embeddings database: {err}").red());

                        return Ok(());
                    }
                };

                println!("⏳ Opening the model...");

                let parser = DocumentsParser::new(
                    config.tokens.lowercase,
                    config.tokens.strip_punctuation,
                    config.tokens.whitespace_tokens
                );

                let device = WgpuDevice::default();

                // Backend::seed(fastrand::u64(..));
                // AutodiffBackend::seed(fastrand::u64(..));

                let text_generation_model = TextGenerationModel::<Wgpu>::load(
                    config.embeddings.embedding_size,
                    config.text_generator.context_tokens_num,
                    config.text_generator.position_encoding_period,
                    &config.text_generator.model_path,
                    &device
                ).unwrap_or_else(|_| TextGenerationModel::<Wgpu>::random(
                    config.embeddings.embedding_size,
                    config.text_generator.context_tokens_num,
                    config.text_generator.position_encoding_period,
                    &device
                ));

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

                    let mut document = Document::default()
                        .with_input(line.trim());

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
                        .collect::<anyhow::Result<Option<Vec<Tensor<Wgpu, 1, Float>>>>>();

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

                        if !config.tokens.whitespace_tokens {
                            stdout.write_all(b" ")?;
                        }

                        stdout.flush()?;

                        if config.text_generator.max_generated_tokens > 0 && i >= config.text_generator.max_generated_tokens {
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
