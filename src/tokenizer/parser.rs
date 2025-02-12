use std::collections::VecDeque;
use std::iter::FusedIterator;

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

    /// Get tokens reader from the given string.
    pub fn read_text(&self, text: impl ToString) -> TokensReader {
        let mut text = text.to_string();

        if self.lowercase {
            text = text.to_lowercase();
        }

        TokensReader {
            text: text.chars()
                .filter(|char| !self.strip_punctuation || !char.is_ascii_punctuation())
                .collect::<VecDeque<char>>(),

            current: 0
        }
    }

    /// Get tokens reader from the given document.
    pub fn read_document(&self, document: Document) -> impl Iterator<Item = String> {
        [Self::INPUT_OPEN_TAG.to_string()].into_iter()
            .chain(self.read_text(document.input))
            .chain([Self::INPUT_CLOSE_TAG.to_string()])
            .chain([Self::CONTEXT_OPEN_TAG.to_string()])
            .chain(self.read_text(document.context))
            .chain([Self::CONTEXT_CLOSE_TAG.to_string()])
            .chain([Self::OUTPUT_OPEN_TAG.to_string()])
            .chain(self.read_text(document.output))
            .chain([Self::OUTPUT_CLOSE_TAG.to_string()])
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

pub struct TokensReader {
    text: VecDeque<char>,
    current: usize
}

impl Iterator for TokensReader {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        let len = self.text.len();

        while self.current < len {
            // Continue collecting alpha-numerics (literal values built from letters and numbers).
            if self.text[self.current].is_alphanumeric() {
                self.current += 1;
            }

            // Skip whitespaces.
            else if self.text[self.current].is_whitespace() {
                // Store the word before whitespace.
                if self.current > 0 {
                    let literal = self.text.drain(0..self.current)
                        .collect::<String>();

                    self.current = 0;

                    return Some(literal);
                }

                // Store whitespace characters.
                else {
                    let whitespace = self.text.pop_front()?.to_string();

                    return Some(whitespace);
                }
            }

            // Store XML tags as single tokens.
            else if self.text[self.current] == '<' {
                // Store the word before the "<".
                if self.current > 0 {
                    let literal = self.text.drain(0..self.current)
                        .collect::<String>();

                    self.current = 0;

                    return Some(literal);
                }

                // Store the XML token if it's valid.
                else if self.current + 1 < len {
                    self.current += 1;

                    // Skip "/" right after "<" (intended syntax).
                    if self.text[self.current] == '/' {
                        self.current += 1;
                    }

                    // Iterate over the content of the tag.
                    while self.current < len && (self.text[self.current].is_alphanumeric() || ['_', '-'].contains(&self.text[self.current])) {
                        self.current += 1;
                    }

                    // If it was properly closed - store it as a single tag.
                    if self.current < len && self.text[self.current] == '>' {
                        let tag = self.text.drain(0..=self.current)
                            .collect::<String>();

                        self.current = 0;

                        return Some(tag);
                    }

                    // Otherwise store "<" and let tokenizer check the string again.
                    else {
                        let char = self.text.pop_front()?.to_string();

                        self.current = 0;

                        return Some(char);
                    }
                }

                // Just store "<" as token.
                else {
                    let char = self.text.pop_front()?.to_string();

                    return Some(char);
                }
            }

            // Store special symbol (non-alpha-numeric value).
            else {
                // Store the word before the symbol.
                if self.current > 0 {
                    let literal = self.text.drain(0..self.current)
                        .collect::<String>();

                    self.current = 0;

                    return Some(literal);
                }

                // Store the symbol itself.
                else {
                    let symbol = self.text.pop_front()?.to_string();

                    return Some(symbol);
                }
            }
        }

        // Store remaining literal.
        if self.current > 0 {
            let literal = self.text.drain(0..self.current)
                .collect::<String>();

            self.current = 0;

            return Some(literal);
        }

        None
    }
}

impl FusedIterator for TokensReader {}

#[test]
fn test_document_tokenizer() {
    let document = Document::new("Example document")
        .with_input("With <very> =special11- #@%\"<-input->\"!")
        .with_context("</and_potentially> broken <xml @ tags>");

    let tokens = Parser::default()
        .read_document(document.clone())
        .collect::<Vec<String>>();

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
