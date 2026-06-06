#![allow(dead_code)]
use crate::storage::tenant::TenantStore;
use crate::api::plan::types::QueryIntent;
use rusqlite::params;

pub struct InferredConclusion {
    pub entity_id: String,
    pub conclusion_text: String,
    pub confidence: f32,
    pub pattern: &'static str,
}

pub fn run_graph_inference(
    tenant: &TenantStore,
    entity_id: &str,
    query_intent: QueryIntent,
) -> Vec<InferredConclusion> {
    if !matches!(query_intent, QueryIntent::Inference | QueryIntent::Recommendation) {
        return Vec::new();
    }

    let mut conclusions = Vec::new();

    let conn_res = tenant.get_conn();
    let conn = match conn_res {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!("Failed to get database connection for graph inference: {:?}", e);
            return Vec::new();
        }
    };

    // Rule 2: PREFERENCE_CHAIN
    if let Ok(mut stmt) = conn.prepare_cached(
        "SELECT e1.target AS preferred, e2.target AS cause
         FROM edges e1
         JOIN edges e2 ON e1.source = e2.source
         WHERE e1.source = ?1
           AND e1.edge_type = 'prefers'
           AND e2.edge_type = 'caused_by'"
    ) {
        if let Ok(mut rows) = stmt.query(params![entity_id]) {
            while let Ok(Some(row)) = rows.next() {
                if let (Ok(pref), Ok(cause)) = (row.get::<_, String>(0), row.get::<_, String>(1)) {
                    conclusions.push(InferredConclusion {
                        entity_id: entity_id.to_string(),
                        conclusion_text: format!("{} prefers {} possibly because of {}.", entity_id, pref, cause),
                        confidence: 0.60,
                        pattern: "preference_chain",
                    });
                }
            }
        }
    }

    // Rule 3: CAUSAL_SEQUENCE
    if let Ok(mut stmt) = conn.prepare_cached(
        "SELECT e1.source AS root_cause, e2.target AS final_effect, e1.target AS intermediate
         FROM edges e1
         JOIN edges e2 ON e1.target = e2.source
         WHERE e1.source = ?1
           AND e1.edge_type = 'leads_to'
           AND e2.edge_type = 'leads_to'"
    ) {
        if let Ok(mut rows) = stmt.query(params![entity_id]) {
            while let Ok(Some(row)) = rows.next() {
                if let (Ok(rc), Ok(fe), Ok(im)) = (row.get::<_, String>(0), row.get::<_, String>(1), row.get::<_, String>(2)) {
                    conclusions.push(InferredConclusion {
                        entity_id: entity_id.to_string(),
                        conclusion_text: format!("{}: {} is a root cause of {} via {}.", entity_id, rc, fe, im),
                        confidence: 0.55,
                        pattern: "causal_sequence",
                    });
                }
            }
        }
    }

    conclusions
}
