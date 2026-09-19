use anyhow::Result;
use std::collections::{HashMap, HashSet};

use crate::graph::Direction;
use crate::storage::tenant::TenantStore;
use crate::storage::types::{
    AgentObservation, FactVersionRow, LedgerTurn, MemoryCard, MemoryCardSearchHit,
    MemoryCardSearchInput, SessionRouterSearchHit,
};
use crate::vector_index::VectorIndex;

pub trait MemoryRepo: Send + Sync {
    fn insert_observations_batch(
        &self,
        items: &[(u64, String, AgentObservation)],
    ) -> Result<Vec<Option<u64>>>;
    fn lookup_by_memory_id(&self, memory_id: &str) -> Result<Option<(u64, Option<u64>)>>;
    fn lookup_by_vector_ids_batch(&self, vector_ids: &[u64]) -> Result<Vec<Option<(u64, String)>>>;
    fn lookup_by_memory_ids_batch(
        &self,
        memory_ids: &[String],
    ) -> Result<HashMap<String, (u64, u64)>>;
    fn memory_identity_batch(
        &self,
        memory_ids: &[String],
    ) -> Result<HashMap<String, (String, u32)>>;
    fn get_observation(&self, timestamp: u64, memory_id: &str) -> Result<Option<AgentObservation>>;
    fn get_observations_batch(
        &self,
        keys: &[(u64, String)],
    ) -> Result<HashMap<String, AgentObservation>>;
    fn stored_content_hashes(&self, memory_ids: &[String]) -> Result<HashMap<String, String>>;
    fn existing_content_hashes(&self, hashes: &[String]) -> Result<HashSet<String>>;
    fn update_embeddings(
        &self,
        updates: &[(String, Vec<f32>)],
    ) -> Result<Vec<(u64, String, Vec<f32>)>>;
}

pub trait CardRepo: Send + Sync {
    fn ingest_cards(&self, cards: &[MemoryCard]) -> Result<()>;
    fn get_memory_card_by_source(&self, source_memory_id: &str) -> Result<Option<MemoryCard>>;
    fn get_memory_cards_batch(&self, card_ids: &[String]) -> Result<HashMap<String, MemoryCard>>;
    fn search_memory_cards(
        &self,
        query: &MemoryCardSearchInput<'_>,
    ) -> Result<Vec<MemoryCardSearchHit>>;
}

pub trait FactRepo: Send + Sync {
    fn fact_versions_for_memories(
        &self,
        memory_ids: &[String],
    ) -> Result<HashMap<String, FactVersionRow>>;
    fn get_current_fact_value(&self, entity_id: &str, fact_key: &str) -> Result<Option<String>>;
    fn invalidated_set(&self, memory_ids: &[String]) -> Result<HashSet<String>>;
    fn invalidated_set_at_time(
        &self,
        point_in_time_ms: u64,
        memory_ids: &[String],
    ) -> Result<HashSet<String>>;
}

pub trait SessionRepo: Send + Sync {
    fn search_session_router(
        &self,
        entity_id: &str,
        query: &str,
        lexical_terms: &[String],
        temporal_terms: &[String],
        entities: &[String],
        limit: usize,
    ) -> Result<Vec<SessionRouterSearchHit>>;
    fn sessions_in_time_window(
        &self,
        entity_id: &str,
        start_ms: u64,
        end_ms: u64,
    ) -> Result<Vec<SessionRouterSearchHit>>;
    fn entity_pivot_sessions(
        &self,
        entity_id: &str,
        subject_entities: &[String],
    ) -> Result<Vec<SessionRouterSearchHit>>;
    fn get_ledger_turns_batch(&self, turn_ids: &[String]) -> Result<HashMap<String, LedgerTurn>>;
    fn get_turn_window(
        &self,
        entity_id: &str,
        session_id: &str,
        turn_index: u32,
        radius: u32,
    ) -> Result<Vec<LedgerTurn>>;
}

pub trait GraphRepo: Send + Sync {
    fn graph_query_edges(
        &self,
        entity: &str,
        label: Option<&str>,
        direction: Direction,
        limit: usize,
    ) -> Result<Vec<crate::storage::tenant::GraphEdge>>;
    fn graph_edge_summaries_for_label(
        &self,
        entity_id: &str,
        label: &str,
        limit: usize,
    ) -> Result<Vec<String>>;
    fn get_edge_cluster_neighbors_batch(
        &self,
        memory_ids: &[String],
        edge_type_filter: Option<&str>,
        limit: usize,
        max_node_degree: usize,
    ) -> Result<HashMap<String, Vec<crate::storage::tenant::EdgeNeighbour>>>;
}

pub trait FtsRepo: Send + Sync {
    fn fts_search(
        &self,
        query: &str,
        limit: usize,
        entity_id: Option<&str>,
    ) -> Result<Vec<(String, f32)>>;
    fn fts_index_batch(&self, batch: &[(String, String, String)]) -> Result<()>;
}

pub trait ProfileRepo: Send + Sync {
    fn get_core_profile(&self, entity_id: &str) -> Result<Option<String>>;
}

pub trait VectorRepo: Send + Sync {
    fn vectors(&self) -> Result<&VectorIndex>;
}

pub trait QueryRepo:
    MemoryRepo + CardRepo + FactRepo + SessionRepo + GraphRepo + ProfileRepo
{
}

impl<T> QueryRepo for T where
    T: MemoryRepo + CardRepo + FactRepo + SessionRepo + GraphRepo + ProfileRepo
{
}

pub trait RetrospectiveRepo: MemoryRepo + FtsRepo + VectorRepo {}

impl<T> RetrospectiveRepo for T where T: MemoryRepo + FtsRepo + VectorRepo {}

impl MemoryRepo for TenantStore {
    fn insert_observations_batch(
        &self,
        items: &[(u64, String, AgentObservation)],
    ) -> Result<Vec<Option<u64>>> {
        TenantStore::insert_observations_batch(self, items)
    }

    fn lookup_by_memory_id(&self, memory_id: &str) -> Result<Option<(u64, Option<u64>)>> {
        TenantStore::lookup_by_memory_id(self, memory_id)
    }

    fn lookup_by_vector_ids_batch(&self, vector_ids: &[u64]) -> Result<Vec<Option<(u64, String)>>> {
        TenantStore::lookup_by_vector_ids_batch(self, vector_ids)
    }

    fn lookup_by_memory_ids_batch(
        &self,
        memory_ids: &[String],
    ) -> Result<HashMap<String, (u64, u64)>> {
        TenantStore::lookup_by_memory_ids_batch(self, memory_ids)
    }

    fn memory_identity_batch(
        &self,
        memory_ids: &[String],
    ) -> Result<HashMap<String, (String, u32)>> {
        TenantStore::memory_identity_batch(self, memory_ids)
    }

    fn get_observation(&self, timestamp: u64, memory_id: &str) -> Result<Option<AgentObservation>> {
        TenantStore::get_observation(self, timestamp, memory_id)
    }

    fn get_observations_batch(
        &self,
        keys: &[(u64, String)],
    ) -> Result<HashMap<String, AgentObservation>> {
        TenantStore::get_observations_batch(self, keys)
    }

    fn stored_content_hashes(&self, memory_ids: &[String]) -> Result<HashMap<String, String>> {
        TenantStore::stored_content_hashes(self, memory_ids)
    }

    fn existing_content_hashes(&self, hashes: &[String]) -> Result<HashSet<String>> {
        TenantStore::existing_content_hashes(self, hashes)
    }

    fn update_embeddings(
        &self,
        updates: &[(String, Vec<f32>)],
    ) -> Result<Vec<(u64, String, Vec<f32>)>> {
        TenantStore::update_embeddings(self, updates)
    }
}

impl CardRepo for TenantStore {
    fn ingest_cards(&self, cards: &[MemoryCard]) -> Result<()> {
        TenantStore::ingest_cards(self, cards)
    }

    fn get_memory_card_by_source(&self, source_memory_id: &str) -> Result<Option<MemoryCard>> {
        TenantStore::get_memory_card_by_source(self, source_memory_id)
    }

    fn get_memory_cards_batch(&self, card_ids: &[String]) -> Result<HashMap<String, MemoryCard>> {
        TenantStore::get_memory_cards_batch(self, card_ids)
    }

    fn search_memory_cards(
        &self,
        query: &MemoryCardSearchInput<'_>,
    ) -> Result<Vec<MemoryCardSearchHit>> {
        TenantStore::search_memory_cards(self, query)
    }
}

impl FactRepo for TenantStore {
    fn fact_versions_for_memories(
        &self,
        memory_ids: &[String],
    ) -> Result<HashMap<String, FactVersionRow>> {
        TenantStore::fact_versions_for_memories(self, memory_ids)
    }

    fn get_current_fact_value(&self, entity_id: &str, fact_key: &str) -> Result<Option<String>> {
        TenantStore::get_current_fact_value(self, entity_id, fact_key)
    }

    fn invalidated_set(&self, memory_ids: &[String]) -> Result<HashSet<String>> {
        TenantStore::invalidated_set(self, memory_ids)
    }

    fn invalidated_set_at_time(
        &self,
        point_in_time_ms: u64,
        memory_ids: &[String],
    ) -> Result<HashSet<String>> {
        TenantStore::invalidated_set_at_time(self, point_in_time_ms, memory_ids)
    }
}

impl SessionRepo for TenantStore {
    fn search_session_router(
        &self,
        entity_id: &str,
        query: &str,
        lexical_terms: &[String],
        temporal_terms: &[String],
        entities: &[String],
        limit: usize,
    ) -> Result<Vec<SessionRouterSearchHit>> {
        TenantStore::search_session_router(
            self,
            entity_id,
            query,
            lexical_terms,
            temporal_terms,
            entities,
            limit,
        )
    }

    fn sessions_in_time_window(
        &self,
        entity_id: &str,
        start_ms: u64,
        end_ms: u64,
    ) -> Result<Vec<SessionRouterSearchHit>> {
        TenantStore::sessions_in_time_window(self, entity_id, start_ms, end_ms)
    }

    fn entity_pivot_sessions(
        &self,
        entity_id: &str,
        subject_entities: &[String],
    ) -> Result<Vec<SessionRouterSearchHit>> {
        TenantStore::entity_pivot_sessions(self, entity_id, subject_entities)
    }

    fn get_ledger_turns_batch(&self, turn_ids: &[String]) -> Result<HashMap<String, LedgerTurn>> {
        TenantStore::get_ledger_turns_batch(self, turn_ids)
    }

    fn get_turn_window(
        &self,
        entity_id: &str,
        session_id: &str,
        turn_index: u32,
        radius: u32,
    ) -> Result<Vec<LedgerTurn>> {
        TenantStore::get_turn_window(self, entity_id, session_id, turn_index, radius)
    }
}

impl GraphRepo for TenantStore {
    fn graph_query_edges(
        &self,
        entity: &str,
        label: Option<&str>,
        direction: Direction,
        limit: usize,
    ) -> Result<Vec<crate::storage::tenant::GraphEdge>> {
        TenantStore::graph_query_edges(self, entity, label, direction, limit)
    }

    fn graph_edge_summaries_for_label(
        &self,
        entity_id: &str,
        label: &str,
        limit: usize,
    ) -> Result<Vec<String>> {
        TenantStore::graph_edge_summaries_for_label(self, entity_id, label, limit)
    }

    fn get_edge_cluster_neighbors_batch(
        &self,
        memory_ids: &[String],
        edge_type_filter: Option<&str>,
        limit: usize,
        max_node_degree: usize,
    ) -> Result<HashMap<String, Vec<crate::storage::tenant::EdgeNeighbour>>> {
        TenantStore::get_edge_cluster_neighbors_batch(
            self,
            memory_ids,
            edge_type_filter,
            limit,
            max_node_degree,
        )
    }
}

impl FtsRepo for TenantStore {
    fn fts_search(
        &self,
        query: &str,
        limit: usize,
        entity_id: Option<&str>,
    ) -> Result<Vec<(String, f32)>> {
        TenantStore::fts_search(self, query, limit, entity_id)
    }

    fn fts_index_batch(&self, batch: &[(String, String, String)]) -> Result<()> {
        TenantStore::fts_index_batch(self, batch)
    }
}

impl ProfileRepo for TenantStore {
    fn get_core_profile(&self, entity_id: &str) -> Result<Option<String>> {
        TenantStore::get_core_profile(self, entity_id)
    }
}

impl VectorRepo for TenantStore {
    fn vectors(&self) -> Result<&VectorIndex> {
        TenantStore::vectors(self)
    }
}

impl TenantStore {
    pub fn query_repo(&self) -> &dyn QueryRepo {
        self
    }

    pub fn retrospective_repo(&self) -> &dyn RetrospectiveRepo {
        self
    }

    pub fn vector_repo(&self) -> &dyn VectorRepo {
        self
    }
}
