use std::sync::Arc;

use burn::prelude::*;
use burn::data::dataset::Dataset;

use crate::prelude::*;

#[derive(Debug, Clone)]
pub struct TextGeneratorTrainSample<B: Backend> {
    /// Series of input tokens windowed by the context through time.
    ///
    /// ```text,ignore
    /// context[timestep][token_n * embedding_size + embedding_i]
    /// ```
    ///
    /// Example text:
    ///
    /// ```text,ignore
    /// ["this", "is", "an", "example", "text"]
    /// ```
    ///
    /// With context window of 2 tokens this tensor will have:
    ///
    /// 1. \["this", "is"\]
    /// 2. \["ia", "an"\]
    /// 3. \["an", "example"\]
    ///
    /// And the target token is "text".
    pub context: Tensor<B, 2, Float>,

    /// Token that should be returned after the series of context windows.
    ///
    /// ```text,ignore
    /// target[embedding_i]
    /// ```
    pub target: Tensor<B, 1, Float>
}

#[derive(Clone)]
pub struct TextGeneratorTrainSamplesDataset<B: Backend> {
    documents: Arc<Box<dyn Dataset<Document>>>,
    embeddings: Arc<WordEmbeddingsDatabase>,
    parser: DocumentsParser,
    embedding_size: usize,
    context_tokens_num: usize,
    device: B::Device
}

impl<B: Backend> TextGeneratorTrainSamplesDataset<B> {
    #[inline]
    pub fn new(
        documents: Arc<Box<dyn Dataset<Document>>>,
        embeddings: Arc<WordEmbeddingsDatabase>,
        parser: DocumentsParser,
        embedding_size: usize,
        context_tokens_num: usize,
        device: B::Device
    ) -> Self {
        Self {
            documents,
            embeddings,
            parser,
            embedding_size,
            context_tokens_num,
            device
        }
    }
}

impl<B: Backend> Dataset<TextGeneratorTrainSample<B>> for TextGeneratorTrainSamplesDataset<B> {
    fn get(&self, index: usize) -> Option<TextGeneratorTrainSample<B>> {
        let document = self.documents.get(index)?;

        let mut context_embeddings = self.parser.read_document(document)
            .map(|token| {
                match self.embeddings.query_embedding(token) {
                    Ok(Some(embedding)) => Tensor::from_floats(embedding.as_slice(), &self.device),
                    _ => Tensor::zeros([self.embedding_size], &self.device)
                }
            })
            .collect::<Vec<Tensor<B, 1, Float>>>();

        let target_embedding = context_embeddings.pop()?;

        let n = context_embeddings.len();

        if n <= self.context_tokens_num {
            let mut zero_padding = vec![
                Tensor::zeros([(self.context_tokens_num - n) * self.embedding_size], &self.device)
            ];

            zero_padding.extend(context_embeddings);

            let timestep_sequence = Tensor::cat(zero_padding, 0);

            Some(TextGeneratorTrainSample {
                context: timestep_sequence.reshape([1, -1]),
                target: target_embedding
            })
        }

        else {
            let m = n - self.context_tokens_num;

            let mut context = Vec::with_capacity(m);

            for i in 0..m {
                let tensor = Tensor::cat(context_embeddings[i..i + self.context_tokens_num].to_vec(), 0);

                context.push(tensor.reshape([1, -1]));
            }

            Some(TextGeneratorTrainSample {
                context: Tensor::cat(context, 0),
                target: target_embedding
            })
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.documents.len()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }
}
