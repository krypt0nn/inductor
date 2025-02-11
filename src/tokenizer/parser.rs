use crate::prelude::*;

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Parser {
    /// Convert all words to lowercase.
    pub lowercase: bool,

    /// Strip all puncturaion characters.
    pub strip_punctuation: bool
}

impl Parser {
    pub const INPUT_OPEN_TAG: &'static str = "<input>";
    pub const INPUT_CLOSE_TAG: &'static str = "</input>";

    pub const CONTEXT_OPEN_TAG: &'static str = "<context>";
    pub const CONTEXT_CLOSE_TAG: &'static str = "</context>";

    pub const OUTPUT_OPEN_TAG: &'static str = "<output>";
    pub const OUTPUT_CLOSE_TAG: &'static str = "</output>";

    #[inline]
    pub fn new(lowercase: bool, strip_punctuation: bool) -> Self {
        Self {
            lowercase,
            strip_punctuation
        }
    }

    /// Return vector of separate words and symbols (tokens) from the given document.
    ///
    /// If `include_ending = true` then `</output>` tag is added to the final vector.
    pub fn parse(&self, document: &Document, include_ending: bool) -> Vec<String> {
        fn parse_text(text: &str, lowercase: bool, strip_punctuation: bool) -> Vec<String> {
            let mut tokens = Vec::new();

            let mut i = 0;
            let mut j = 0;

            let text: Vec<char> = if lowercase {
                text.to_lowercase().chars().collect()
            } else {
                text.chars().collect()
            };

            let n = text.len();

            while j < n {
                // Continue collecting alpha-numerics (literal values built from letters and numbers).
                if text[j].is_alphanumeric() {
                    j += 1;
                }

                // Skip whitespaces.
                else if text[j].is_whitespace() {
                    // Store the word before whitespace.
                    if i < j {
                        tokens.push(text[i..j].iter().collect());
                    }

                    // Skip all the following whitespaces as well.
                    while j < n && text[j].is_whitespace() {
                        tokens.push(text[j].to_string());

                        j += 1;
                    }

                    // Set cursor to the whitespace's end.
                    i = j;
                }

                // XML tags.
                else if text[j] == '<' {
                    // Store the word before the symbol.
                    if i < j {
                        tokens.push(text[i..j].iter().collect());
                    }

                    i = j;

                    // If there's more symbols after "<".
                    if j + 1 < n {
                        j += 1;

                        // Skip "/" right after "<" (intended syntax).
                        if text[j] == '/' {
                            j += 1;
                        }

                        // Iterate over the content of the tag.
                        while j < n && (text[j].is_alphanumeric() || ['_', '-'].contains(&text[j])) {
                            j += 1;
                        }

                        // If it was properly closed - store it as a single tag.
                        if j < n && text[j] == '>' {
                            tokens.push(text[i..=j].iter().collect());
                        }

                        // Otherwise store "<", "/" and literal value separately.
                        else {
                            tokens.push(String::from("<"));

                            // Store "/" and literal separately.
                            if text[i + 1] == '/' {
                                tokens.push(String::from("/"));
                                tokens.push(text[i + 2..j].iter().collect());
                            }

                            // Store just the literal if there weren't "/".
                            else {
                                tokens.push(text[i + 1..j].iter().collect());
                            }

                            // Store whatever other symbol we got (if we got any).
                            if j < n {
                                tokens.push(text[j].to_string());
                            }
                        }

                        j += 1;
                        i = j;
                    }

                    // Just store this random "<" as a token.
                    else {
                        tokens.push(text[j].to_string());
                    }
                }

                // Store special symbol (non-alpha-numeric value).
                else {
                    // Store the word before the symbol.
                    if i < j {
                        tokens.push(text[i..j].iter().collect());
                    }

                    // Store the symbol.
                    if !strip_punctuation || !text[j].is_ascii_punctuation() {
                        tokens.push(text[j].to_string());
                    }

                    // Update cursors.
                    j += 1;
                    i = j;
                }
            }

            // Store remaining word.
            if i < j {
                tokens.push(text[i..j].iter().collect());
            }

            tokens
        }

        let mut input_tokens = parse_text(&document.input, self.lowercase, self.strip_punctuation);
        let mut context_tokens = parse_text(&document.context, self.lowercase, self.strip_punctuation);
        let mut output_tokens = parse_text(&document.output, self.lowercase, self.strip_punctuation);

        let mut tokens = Vec::with_capacity(input_tokens.len() + context_tokens.len() + output_tokens.len() + 10);

        // <input>...</input>
        tokens.push(Self::INPUT_OPEN_TAG.to_string());
        tokens.append(&mut input_tokens);
        tokens.push(Self::INPUT_CLOSE_TAG.to_string());

        // <context>...</context>
        tokens.push(Self::CONTEXT_OPEN_TAG.to_string());
        tokens.append(&mut context_tokens);
        tokens.push(Self::CONTEXT_CLOSE_TAG.to_string());

        // <output>...</output>
        tokens.push(Self::OUTPUT_OPEN_TAG.to_string());
        tokens.append(&mut output_tokens);

        if include_ending {
            tokens.push(Self::OUTPUT_CLOSE_TAG.to_string());
        }

        tokens
    }

    /// Try to reconstruct document from the given tokens slice.
    ///
    /// Return `None` if provided tokens have invalid format.
    pub fn join(&self, tokens: &[String]) -> Option<Document> {
        let document = Document::default();

        fn parse_section<'a>(tokens: &'a [String], open: &'static str, close: &'static str) -> Option<(String, &'a [String])> {
            // <tag>...</tag>
            if tokens.first().map(String::as_str) != Some(open) {
                return None;
            }

            let mut i = 1;
            let n = tokens.len();

            while i < n {
                if tokens[i] == close {
                    return Some((tokens[1..i].concat(), &tokens[i + 1..]));
                }

                i += 1;
            }

            None
        }

        let (input,   tokens) = parse_section(tokens, Self::INPUT_OPEN_TAG,   Self::INPUT_CLOSE_TAG)?;
        let (context, tokens) = parse_section(tokens, Self::CONTEXT_OPEN_TAG, Self::CONTEXT_CLOSE_TAG)?;
        let (output, _tokens) = parse_section(tokens, Self::OUTPUT_OPEN_TAG,  Self::OUTPUT_CLOSE_TAG)?;

        Some(document.with_input(input)
            .with_context(context)
            .with_output(output))
    }
}

#[test]
fn test_document_tokenizer() {
    let document = Document::new("Example document")
        .with_input("With <very> =special11- #@%\"<-input->\"!")
        .with_context("</and_potentially> broken <xml @ tags>");

    let tokens = Parser::default().parse(&document, true);

    assert_eq!(tokens, &[
        "<input>", "With", " ", "<very>", " ", "=", "special11", "-", " ", "#", "@", "%", "\"", "<-input->", "\"", "!", "</input>",
        "<context>", "</and_potentially>", " ", "broken", " ", "<", "xml", " ", "@", " ", "tags", ">", "</context>",
        "<output>", "Example", " ", "document", "</output>"
    ]);

    let detokenized = Parser::default()
        .join(&tokens)
        .unwrap();

    assert_eq!(document, detokenized);
}
