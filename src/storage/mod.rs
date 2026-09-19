pub mod entity_resolver;
pub mod manager;
pub mod tenant;
pub mod types;

pub use manager::TenantDatabaseManager;
pub use repo::ingest::{CommitOutcome, IngestBatches, IngestFactRegistration};
pub use tenant::TenantStore;
pub use types::*;
pub mod repo;
pub use repo::traits::{QueryRepo, RetrospectiveRepo, VectorRepo};
