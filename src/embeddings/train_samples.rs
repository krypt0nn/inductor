use std::sync::Arc;

use serde::{Serialize, Deserialize};

use burn::prelude::*;
use burn::data::dataset::Dataset;

use crate::prelude::*;

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WordEmbeddingSamplingMethod {
    #[default]
    /// Continuous Bag of Words (CBOW) - learn to predict one word from its
    /// surrounding context (many to one).
    ///
    /// - Trains faster than skip-gram.
    /// - Uses smaller context window.
    /// - Requires larger dataset.
    Cbow,

    /// Skip-Gram - learn to predict surrouding context from one
    /// target word (one to many).
    ///
    /// - Trains slower than CBOW.
    /// - Uses larger context window.
    /// - Can be used on smaller dataset with many rare tokens.
    SkipGram
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WordEmbeddingSamplingParams {
    /// Method used to learn word embeddings.
    pub sampling_method: WordEmbeddingSamplingMethod,

    /// Maximal amount of tokens which can be learned by the model.
    ///
    /// Directly affects RAM usage.
    pub one_hot_tokens: usize,

    /// Amount of tokens around the target one to learn embeddings from.
    pub context_radius: usize,

    /// Use uniformal random distribution for context radius,
    /// in range `[1, context_radius]`.
    pub dynamic_context_radius: bool,

    /// Skip tokens which occured less times than the specified amount.
    pub min_occurences: usize,

    /// Used to calculate probability of skipping word from training samples.
    ///
    /// Probability of keeping word in train samples is calculated as:
    ///
    /// ```text,ignore
    /// P_keep(token) = sqrt(token_frequency / subsample_value + 1) * subsample_value / token_frequency
    /// ```
    pub subsample_value: f64
}

#[derive(Debug, Clone)]
/// Single word embedding train sample.
pub struct WordEmbeddingTrainSample<B: Backend> {
    pub context: Tensor<B, 1, Float>,
    pub target: Tensor<B, 1, Float>
}

#[derive(Debug, Clone)]
/// Read word embedding train samples from the parsed document.
pub struct WordEmbeddingsTrainSamplesDataset<B: Backend> {
    document_tokens: Arc<Vec<usize>>,
    sampled_tokens: Arc<Vec<usize>>,
    device: B::Device,
    params: WordEmbeddingSamplingParams
}

impl<B: Backend> WordEmbeddingsTrainSamplesDataset<B> {
    /// Split given document into words and convert them into tokens.
    pub fn from_document(
        document: Document,
        parser: &DocumentsParser,
        tokens_db: &mut TokensDatabase,
        device: B::Device,
        params: WordEmbeddingSamplingParams
    ) -> anyhow::Result<Self> {
        let mut tokens = parser.read_document(document)
            .map(|word| {
                match tokens_db.query_token_by_value(&word) {
                    Ok(Some(token)) => Ok(Some(token)),

                    // Insert token if it didn't exist in tokens database.
                    _ => {
                        let transaction = tokens_db.insert_tokens()?;

                        transaction.insert_token(&word)?;
                        transaction.commit()?;

                        // Query token again to include its frequency.
                        tokens_db.query_token_by_value(word)
                    }
                }
            })
            .collect::<anyhow::Result<Option<Vec<TokensDatabaseRecord>>>>()?
            .ok_or_else(|| anyhow::anyhow!("Failed to query token for one of words within the document"))?;

        let n = tokens.len();

        let document_tokens = tokens.iter()
            .map(|token| token.id as usize)
            .collect::<Vec<usize>>();

        if n > params.context_radius * 2 {
            let mut sampled_tokens = Vec::with_capacity(n - params.context_radius * 2);

            for (i, target_token) in tokens.drain(params.context_radius..n - params.context_radius).enumerate() {
                // Skip token if it occured too few times in input documents.
                if (target_token.occurences as usize) < params.min_occurences {
                    continue;
                }

                // Calculate probability of skipping this token.
                // let skip_probability = 1.0 - (params.subsample_value / target_token.frequency).sqrt().clamp(0.0, 1.0);
                let skip_probability = 1.0 - ((target_token.frequency / params.subsample_value + 1.0).sqrt() * (params.subsample_value / target_token.frequency)).clamp(0.0, 1.0);

                // Randomly skip it.
                if fastrand::f64() < skip_probability {
                    continue;
                }

                sampled_tokens.push(params.context_radius + i);
            }

            return Ok(Self {
                document_tokens: Arc::new(document_tokens),
                sampled_tokens: Arc::new(sampled_tokens),
                device,
                params
            });
        }

        Ok(Self {
            document_tokens: Arc::new(vec![]),
            sampled_tokens: Arc::new(vec![]),
            device,
            params
        })
    }
}

impl<B: Backend> Dataset<WordEmbeddingTrainSample<B>> for WordEmbeddingsTrainSamplesDataset<B> {
    fn get(&self, index: usize) -> Option<WordEmbeddingTrainSample<B>> {
        let i = self.sampled_tokens.get(index).copied()?;

        let target_token = self.document_tokens[i];

        let context_radius = if self.params.dynamic_context_radius {
            fastrand::usize(1..=self.params.context_radius)
        } else {
            self.params.context_radius
        };

        let mut context_tokens = vec![0; context_radius * 2];

        context_tokens[..context_radius].copy_from_slice(&self.document_tokens[i - context_radius..i]);
        context_tokens[context_radius..].copy_from_slice(&self.document_tokens[i + 1..i + context_radius + 1]);

        let context_tensor = one_hot_tensor(&context_tokens, self.params.one_hot_tokens, &self.device);
        let target_tensor = one_hot_tensor(&[target_token], self.params.one_hot_tokens, &self.device);

        match self.params.sampling_method {
            WordEmbeddingSamplingMethod::Cbow => Some(WordEmbeddingTrainSample {
                context: context_tensor,
                target: target_tensor
            }),

            WordEmbeddingSamplingMethod::SkipGram => Some(WordEmbeddingTrainSample {
                context: target_tensor,
                target: context_tensor
            })
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.sampled_tokens.len()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.sampled_tokens.is_empty()
    }
}
