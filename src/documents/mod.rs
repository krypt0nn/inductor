pub mod document;
pub mod database;

pub mod prelude {
    pub use super::document::Document;
    pub use super::database::Database as DocumentsDatabase;
}
