use std::sync::Arc;

use burn::prelude::*;
use burn::data::dataset::Dataset;

use crate::prelude::*;

#[derive(Debug, Clone)]
/// Read word embedding train samples from the parsed document.
pub struct WordEmbeddingsTrainSamplesDataset<B: Backend> {
    tokens: Arc<Vec<usize>>,
    context_radius: usize,
    one_hot_length: usize,
    device: B::Device
}

impl<B: Backend> WordEmbeddingsTrainSamplesDataset<B> {
    /// Split given document into words and convert them into tokens.
    pub fn from_document(
        document: &Document,
        parser: &DocumentsParser,
        tokens_db: &TokensDatabase,
        device: B::Device
    ) -> anyhow::Result<Self> {
        let tokens = parser.parse(document, true)
            .into_iter()
            .map(|word| tokens_db.insert_token(word).map(|token| token as usize))
            .collect::<anyhow::Result<Vec<usize>>>()?;

        Ok(Self {
            tokens: Arc::new(tokens),
            context_radius: EMBEDDING_CONTEXT_RADIUS,
            one_hot_length: EMBEDDING_MAX_TOKENS,
            device
        })
    }

    #[inline]
    pub fn with_context_radius(mut self, context_radius: usize) -> Self {
        self.context_radius = context_radius;

        self
    }

    #[inline]
    pub fn with_one_hot_length(mut self, one_hot_length: usize) -> Self {
        self.one_hot_length = one_hot_length;

        self
    }
}

#[derive(Debug, Clone)]
/// Single word embedding train sample.
pub struct WordEmbeddingTrainSample<B: Backend> {
    pub context: Tensor<B, 1, Float>,
    pub target: Tensor<B, 1, Float>
}

impl<B: Backend> Dataset<WordEmbeddingTrainSample<B>> for WordEmbeddingsTrainSamplesDataset<B> {
    fn get(&self, index: usize) -> Option<WordEmbeddingTrainSample<B>> {
        let i = self.context_radius + index;

        self.tokens.get(i)?;

        let mut context = vec![0; self.context_radius * 2];
        let target = self.tokens[i];

        context[..self.context_radius].copy_from_slice(&self.tokens[i - self.context_radius..i]);
        context[self.context_radius..].copy_from_slice(&self.tokens[i + 1..i + self.context_radius + 1]);

        Some(WordEmbeddingTrainSample {
            context: one_hot_tensor(&context, self.one_hot_length, &self.device),
            target: one_hot_tensor(&[target], self.one_hot_length, &self.device)
        })
    }

    fn len(&self) -> usize {
        let n = self.tokens.len();
        let d = self.context_radius * 2;

        n.checked_sub(d).unwrap_or_default()
    }

    fn is_empty(&self) -> bool {
        let n = self.tokens.len();
        let d = self.context_radius * 2;

        n > d
    }
}
