use std::sync::Arc;

use burn::prelude::*;
use burn::data::dataset::Dataset;

use crate::prelude::*;

#[derive(Debug, Clone)]
pub struct TextGeneratorTrainSample<B: Backend> {
    /// Concatenated previous N tokens' embeddings.
    pub context: Tensor<B, 1, Float>,

    /// Target token's embedding.
    pub target: Tensor<B, 1, Float>
}

#[derive(Clone)]
pub struct TextGeneratorTrainSamplesDataset<B: Backend> {
    tokens: Arc<Vec<String>>,
    embeddings: Arc<WordEmbeddingsDatabase>,
    embedding_size: usize,
    context_tokens_num: usize,
    position_encoding_period: usize,
    device: B::Device
}

impl<B: Backend> TextGeneratorTrainSamplesDataset<B> {
    #[inline]
    pub fn from_document(
        document: Document,
        parser: &DocumentsParser,
        embeddings: Arc<WordEmbeddingsDatabase>,
        embedding_size: usize,
        context_tokens_num: usize,
        position_encoding_period: usize,
        device: B::Device
    ) -> Self {
        let tokens = parser.read_document(document).collect();

        Self {
            tokens: Arc::new(tokens),
            embeddings,
            embedding_size,
            context_tokens_num,
            position_encoding_period,
            device
        }
    }
}

impl<B: Backend> Dataset<TextGeneratorTrainSample<B>> for TextGeneratorTrainSamplesDataset<B> {
    fn get(&self, index: usize) -> Option<TextGeneratorTrainSample<B>> {
        // Get target token.
        let target_token = self.tokens.get(index)?;

        // Get target token's embedding.
        let Ok(Some(target_token)) = self.embeddings.query_embedding(target_token) else {
            return None;
        };

        // If we can take all context tokens.
        if index >= self.context_tokens_num {
            let context_tokens = self.tokens[index - self.context_tokens_num..index].iter()
                .enumerate()
                .map(|(i, token)| {
                    match self.embeddings.query_embedding(token) {
                        Ok(Some(embedding)) => Some(encode_position(
                            Tensor::from_floats(embedding.as_slice(), &self.device),
                            index + i, // unintuitive but trust me
                            self.position_encoding_period
                        )),
                        _ => None
                    }
                })
                .collect::<Option<Vec<Tensor<B, 1, Float>>>>()?;

            Some(TextGeneratorTrainSample {
                context: Tensor::cat(context_tokens, 0),
                target: Tensor::from_floats(target_token.as_slice(), &self.device)
            })
        }

        // Otherwise we will take what we can and fill everything else with zeros.
        else {
            let mut padding_tokens = (0..self.context_tokens_num - index)
                .map(|i| encode_position(Tensor::zeros([self.embedding_size], &self.device), i, self.position_encoding_period))
                .collect::<Vec<Tensor<B, 1, Float>>>();

            let context_tokens = self.tokens[0..index].iter()
                .enumerate()
                .map(|(i, token)| {
                    match self.embeddings.query_embedding(token) {
                        Ok(Some(embedding)) => Some(encode_position(
                            Tensor::from_floats(embedding.as_slice(), &self.device),
                            index + i, // unintuitive but trust me
                            self.position_encoding_period
                        )),
                        _ => None
                    }
                })
                .collect::<Option<Vec<Tensor<B, 1, Float>>>>()?;

            padding_tokens.extend(context_tokens);

            Some(TextGeneratorTrainSample {
                context: Tensor::cat(padding_tokens, 0),
                target: Tensor::from_floats(target_token.as_slice(), &self.device)
            })
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.tokens.len()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}
