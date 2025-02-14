# Inductor

Logical continuation of my [markov-chains](https://github.com/krypt0nn/markov-chains) project for text generation using neural networks.

## Get started

```bash
cargo build --release
```

## 1. Prepare documents

Document is a structured block of text which contains "input", "context" and "output" sections divided by XML tags.
By default input and context of a document are empty but you can manually assign them. Documents are compressed and
stored in SQLite database to save your disk space.

### Create new documents database

```bash
inductor documents --database documents.db create
```

### Insert documents to the dataset

| Optional flags              | Meaning                                                                  |
| --------------------------- | ------------------------------------------------------------------------ |
| `--lowercase`               | Convert document text to lowercase                                       |
| `--discord-chat`            | Assume given document path is a discord chat history dump in JSON format |
| `--discord-split-documents` | Split messages into separate documents                                   |
| `--discord-last-n`          | Export only given amount of last messages                                |

```bash
inductor documents --database documents.db insert --document my_document.txt
```

## 2. Create tokens database

Token is a minimal undividable entity of a document. Depending on documents parser configuration it can contain
one or many characters, including whitespaces. Tokens database indexes unique tokens which is needed in later steps.

### Create tokens database

```bash
inductor tokens --database tokens.db create
```

### Update tokens database from documents dataset

| Optional flags        | Meaning                                                       |
| --------------------- | ------------------------------------------------------------- |
| `--lowercase`         | Convert document text to lowercase                            |
| `--strip-punctuation` | Remove all punctuation characters. Can easily break your text |
| `--whitespace-tokens` | Make whitespace characters separate tokens                    |

```bash
inductor tokens --database tokens.db update --documents documents.db
```

## 3. Train word embeddings model

Word embeddings model maps each token into a multi-dimensional vector. During training model
tried to figure out relations between the words in all the documents you provided it with
and distribute all the tokens in a vast space. If two words have similar meaning - they will
be much closer to each other in this space than other words, which will greatly improve
text generation quality since the prediction error will not be so easily observable.

### Create word embeddings model

```bash
inductor embeddings --database embeddings.db create
```

### Train the model on given documents

| Optional flags               | Meaning                                                                           |
| ---------------------------- | --------------------------------------------------------------------------------- |
| `--one-hot-tokens`           | Maximal amount of tokens which can be encoded by the model                        |
| `--embedding-size`           | Amount of dimensions in a word embedding                                          |
| `--embedding-context-radius` | Amount or tokens to the left and right of the current one used to train the model |
| `--lowercase`                | Convert document text to lowercase                                                |
| `--strip-punctuation`        | Remove all punctuation characters. Can easily break your text                     |
| `--whitespace-tokens`        | Make whitespace characters separate tokens                                        |
| `--remote-device`            | URL to a remote device. Can be set multiple times                                 |
| `--epochs`                   | Amount of epochs to train the model                                               |
| `--initial-learn-rate`       | Initial learn rate of the model training. Should be relatively large              |
| `--final-learn-rate`         | Final learn rate of the model training. Should be relatively small                |
| `--batch-size`               | Amount of sequences to train at one iteration. Increases memory use               |
| `--accumulate-gradients`     | Average last iterations before updating the model's weights                       |

```bash
inductor embeddings --database embeddings.db train --documents documents.db --tokens tokens.db --model embeddings-model
```

### Update embeddinds using pre-trained model

After model's training embeddings database is updated automatically, but if you
downloaded the pre-trained model - you can use this method and your own tokens database
to create word embeddings database.

| Optional flags        | Meaning                                                    |
| --------------------- | ---------------------------------------------------------- |
| `--one-hot-tokens`    | Maximal amount of tokens which can be encoded by the model |
| `--embedding-size`    | Amount of dimensions in a word embedding                   |

```bash
inductor embeddings --database embeddings.db update --tokens tokens.db --model embeddings-model
```

### Compare words to each other

This method allows you to use word embeddings database to find words with meaning
closest to your input word. Useful to debug your model's training results.

```bash
inductor embeddings --database embeddings.db compare
```

### Export word embeddings from the database

With this method you can export all the tokens and their embedding vectors to a CSV table
to analyze them manually, e.g. by using [this website](https://www.csvplot.com).

```bash
inductor embeddings --database embeddings.db export --csv embeddings.csv
```

## 4. Train text generation model

Currently text generation model uses 2 layers neural network: recurrent LSTM layer with last 8 tokens
window, and a feed-forward dense layer.

### Train the model on given documents and word embeddings

| Optional flags           | Meaning                                                              |
| ------------------------ | -------------------------------------------------------------------- |
| `--embedding-size`       | Amount of dimensions in a word embedding                             |
| `--context-tokens-num`   | Amount of tokens used to predict the next one                        |
| `--lowercase`            | Convert document text to lowercase                                   |
| `--strip-punctuation`    | Remove all punctuation characters. Can easily break your text        |
| `--whitespace-tokens`    | Make whitespace characters separate tokens                           |
| `--remote-device`        | URL to a remote device. Can be set multiple times                    |
| `--epochs`               | Amount of epochs to train the model                                  |
| `--initial-learn-rate`   | Initial learn rate of the model training. Should be relatively large |
| `--final-learn-rate`     | Final learn rate of the model training. Should be relatively small   |
| `--batch-size`           | Amount of sequences to train at one iteration. Increases memory use  |
| `--accumulate-gradients` | Average last iterations before updating the model's weights          |

```bash
inductor text-generator --model text-generator-model train --documents documents.db --embeddings embeddings.db
```

### Generate text using the trained model

| Optional flags         | Meaning                                                       |
| ---------------------- | ------------------------------------------------------------- |
| `--embedding-size`     | Amount of dimensions in a word embedding                      |
| `--context-tokens-num` | Amount of tokens used to predict the next one                 |
| `--lowercase`          | Convert document text to lowercase                            |
| `--strip-punctuation`  | Remove all punctuation characters. Can easily break your text |
| `--whitespace-tokens`  | Make whitespace characters separate tokens                    |
| `--context`            | Optional context string applied to the generating document    |
| `--max-tokens`         | Maximal amount of tokens to generate                          |

```bash
inductor text-generator --model text-generator-model generate --embeddings embeddings.db
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

Author: [Nikita Podvirnyi](https://github.com/krypt0nn)\
Licensed under [GPL-3.0](LICENSE)
