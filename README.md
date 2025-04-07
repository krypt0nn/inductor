# Inductor

Logical continuation of my [markov-chains](https://github.com/krypt0nn/markov-chains) project for text generation using neural networks.

## Get started

```bash
cargo build --release
```

## 0. Create new project

Every time you want to experiment with some model you need to create a new project. This is a separate folder with
a TOML formatted config file, bunch of sqlite databases and compressed neural network models. Config file, by default
named `inductor.toml`, contains parameters used by the tokens parser, word embeddings and text generation models,
sqlite databases and so on.

```bash
inductor --config 'path/to/inductor.toml' init
```

By default `./inductor.toml` path is assumed. It's **generally recommended** to look into this file and tweak
parameters depending on your needs because defaults are not meant to be good for every usecase.

## 1. Prepare documents

Document is a structured block of text which contains "input", "context" and "output" sections divided by XML tags.
By default input and context of a document are empty but you can manually assign them. Documents are compressed and
stored in SQLite database to save your disk space.

### Insert documents to the dataset

| Optional flags              | Meaning                                                                  |
| --------------------------- | ------------------------------------------------------------------------ |
| `--discord-chat`            | Assume given document path is a discord chat history dump in JSON format |
| `--discord-split-documents` | Split messages into separate documents                                   |
| `--discord-last-n`          | Export only given amount of last messages                                |

```bash
inductor documents insert --document my_document.txt
```

## 2. Create tokens database

Token is a minimal undividable entity of a document. Depending on documents parser configuration it can contain
one or many characters, including whitespaces. Tokens database indexes unique tokens which is needed in later steps.

### Update tokens database from documents dataset

```bash
inductor tokens update
```

## 3. Train word embeddings model

Word embeddings model maps each token into a multi-dimensional vector. During training model
tried to figure out relations between the words in all the documents you provided it with
and distribute all the tokens in a vast space. If two words have similar meaning - they will
be much closer to each other in this space than other words, which will greatly improve
text generation quality since the prediction error will not be so easily observable.

### Train the model on given documents

During training model will try to learn relative positions of tokens in natural text documents. To do this
it will read `embedding_context_radius * 2` tokens around the target token and learn to predict it using
surrounding tokens. To improve model's performance we skip tokens with too few occurences in the documents
(with less than `minimal_occurences`) and randomly skip tokens based on their frequency and `subsampling_value`.

Probability of keeping word in train samples is calculated as:

```
P_keep(token) = clamp(sqrt(token_frequency / subsample_value + 1) * subsample_value / token_frequency)
```

Where `clamp` ensures that `sqrt` value is within `[0.0, 1.0]` range.
Lower subsample value means less train samples.

```bash
inductor embeddings train
```

### Update embeddinds using pre-trained model

After the model is trained you need to update the database of all the embeddings
from the input documents.  This is needed to optimize embeddings querying during
text generation.

> Note that  in previous versions  word embeddings  were  updated  automatically
> after the training is finished, but not you need to run this command yourself.

```bash
inductor embeddings update
```

### Compare words to each other

This method allows you to use word embeddings database to find words with meaning
closest to your input word. Useful to debug your model's training results.

```bash
inductor embeddings compare
```

### Export word embeddings from the database

With this method you can export all the tokens and their embedding vectors to a CSV table
to analyze them manually, e.g. by using [this website](https://www.csvplot.com).

| Optional flags | Meaning                                                    |
| -------------- | ---------------------------------------------------------- |
| `--csv`        | Path to the CSV file where to save all the word embeddings |

```bash
inductor embeddings export --csv embeddings.csv
```

## 4. Train text generation model

Text generation model uses `context_tokens_num` tokens to predict the following one.
It also uses positional encoding which adds sines with different properties to the tokens' embeddings.
Theoretically positional encoding should allow model to differ the same word (embedding) placed
on different positions within the text. You can disable positional encoding by setting
`position_encoding_period` to 0.

### Train the model on given documents and word embeddings

```bash
inductor text-generator train
```

### Generate text using the trained model

| Optional flags | Meaning                                                    |
| -------------- | ---------------------------------------------------------- |
| `--context`    | Optional context string applied to the generating document |

```bash
inductor text-generator generate
```

## Bonus: host your device for remote model training

If you have access to remove devices - you can host their computation resources
to train the model remotely. Good for making sparse computations pool.

Used in combination with `--remote-device` flags.

| Optional flags | Meaning         |
| -------------- | --------------- |
| `--port`       | Connection port |

```bash
inductor serve
```

> Note: burn doesn't support remote devices yet. This is a placeholder for potential future support.

Author: [Nikita Podvirnyi](https://github.com/krypt0nn)\
Licensed under [GPL-3.0](LICENSE)
