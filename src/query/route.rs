use super::*;

pub(crate) fn collect_edge_cluster_scores_for_seeds(
    tenant: &TenantStore,
    seeds: &[String],
    max_depth: usize,
    max_node_degree: usize,
    edge_type_filter: Option<&str>,
    intent: Option<crate::api::plan::types::QueryIntent>,
) -> (Vec<HashMap<String, f32>>, u64) {
    const NEIGHBORS_PER_NODE: usize = 50;
    const DEPTH_DECAY: f32 = 0.6;

    let mut neighbours: HashMap<String, Vec<(String, f32, String)>> = HashMap::new();
    let mut accumulated: Vec<HashMap<String, f32>> = vec![HashMap::new(); seeds.len()];
    let mut frontiers: Vec<Vec<String>> = seeds.iter().map(|seed| vec![seed.clone()]).collect();
    let mut expanded = 0u64;
    let mut path_weight = 1.0f32;
    for _depth in 0..max_depth {
        let mut missing: Vec<String> = frontiers
            .iter()
            .flatten()
            .filter(|node| !neighbours.contains_key(*node))
            .cloned()
            .collect();
        missing.sort();
        missing.dedup();
        if !missing.is_empty() {
            expanded += missing.len() as u64;
            match tenant.get_edge_cluster_neighbors_batch(
                &missing,
                edge_type_filter,
                NEIGHBORS_PER_NODE,
                max_node_degree,
            ) {
                Ok(found) => neighbours.extend(found),
                Err(err) => {
                    tracing::warn!(error = ?err, "edge cluster expansion failed");
                    break;
                }
            }
        }

        let mut any_next = false;
        for (frontier, scores) in frontiers.iter_mut().zip(accumulated.iter_mut()) {
            let mut next = Vec::new();
            let mut seen_next = HashSet::new();
            for node in frontier.iter() {
                for (linked_mid, weight, edge_type) in neighbours.get(node).into_iter().flatten() {
                    let intent_mult = intent_weight_for_edge(edge_type.as_str(), intent);
                    *scores.entry(linked_mid.clone()).or_insert(0.0) +=
                        path_weight * weight * intent_mult;
                    if seen_next.insert(linked_mid.as_str()) {
                        next.push(linked_mid.clone());
                    }
                }
            }
            any_next |= !next.is_empty();
            *frontier = next;
        }
        if !any_next {
            break;
        }
        path_weight *= DEPTH_DECAY;
    }

    (accumulated, expanded)
}

/// Returns 1.5 for edges that align with the query intent, 1.0 otherwise.
pub(crate) fn intent_weight_for_edge(
    edge_type: &str,
    intent: Option<crate::api::plan::types::QueryIntent>,
) -> f32 {
    use crate::api::plan::types::QueryIntent;
    let et = edge_type.to_ascii_lowercase();
    match intent {
        Some(QueryIntent::Inference) | Some(QueryIntent::Recommendation) => {
            if et == "caused_by" || et == "leads_to" || et == "prefers" {
                1.5
            } else {
                1.0
            }
        }
        Some(QueryIntent::PeripheralMention) => {
            if et == "updates" || et == "supports" || et == "contradicts" {
                1.5
            } else {
                1.0
            }
        }
        _ => 1.0,
    }
}

pub(crate) fn route_phase(s: &mut QueryPipelineState) {
    if !s.state.config.features.enabled(Feature::SessionRouter)
        || !s.state.config.lanes.enabled(Lane::Route)
    {
        return;
    }
    let route_probe_queries = if s.budget.route_probe_query_limit == 0 {
        Vec::new()
    } else if s.plan.coverage_facets.is_empty() {
        vec![s.plan.fts_queries.first().cloned().unwrap_or_else(|| s.query_text.clone())]
    } else {
        s.plan
            .coverage_facets
            .iter()
            .take(s.budget.route_probe_query_limit)
            .map(|facet| facet.text.clone())
            .collect::<Vec<_>>()
    };
    let mut route_probe_results = Vec::new();
    let tenant_for_probes = s.tenant.clone();
    let eid_for_probes = s.payload.entity_id.clone();
    let hit_limit = s.budget.route_probe_hit_limit;
    std::thread::scope(|sc| {
        let mut handles = Vec::new();
        for (probe_idx, probe_query) in route_probe_queries.iter().enumerate() {
            let tenant = tenant_for_probes.clone();
            let eid = eid_for_probes.clone();
            handles.push(sc.spawn(move || {
                let hits = tenant
                    .fts_search(probe_query.as_str(), hit_limit, eid.as_deref())
                    .unwrap_or_else(|err| {
                        // An empty lane and a failed lane look identical
                        // downstream, and the difference is a recall bug.
                        tracing::warn!(error = %err, probe = %probe_query, "route probe FTS failed");
                        Vec::new()
                    });
                (probe_idx, hits)
            }));
        }
        for handle in handles {
            if let Ok(res) = handle.join() {
                route_probe_results.push(res);
            }
        }
    });

    // Lanes for RRF fusion. Each lane is independently ranked.
    let mut lanes: Vec<Vec<(String, f32)>> = Vec::with_capacity(5);

    // Lane 1: entity-anchored pivot (resolved subjects → sessions). Strongest
    // signal for multi-hop and cross-entity questions.
    if !s.plan.subject_entities.is_empty() {
        let tenant = s.tenant.clone();
        let eid = s.payload.entity_id.clone();
        let subjects = s.plan.subject_entities.clone();
        if let Ok(hits) = tenant.entity_pivot_sessions(eid.as_deref().unwrap_or(""), &subjects) {
            lanes.push(hits.into_iter().map(|h| (h.session_id, h.score)).collect());
        }
    }

    // Lane 2: FTS session router (router_text OR of query terms).
    let s_router_lane: Vec<(String, f32)> =
        s.route.session_scores.iter().map(|(sid, score)| (sid.clone(), *score)).collect();
    if !s_router_lane.is_empty() {
        lanes.push(s_router_lane);
    }

    // Lane 3: time-window hits.
    // Already merged into s.route.session_scores via the plan_phase time-window
    // pass; we re-extract the contribution by snapshot before any further
    // mutations. For now this lane is implicit in lane 2.

    // Lane 4: FTS probe votes.
    let mut probe_lane: Vec<(String, f32)> = Vec::new();
    let probe_ids: Vec<String> = route_probe_results
        .iter()
        .flat_map(|(_, hits)| {
            hits.iter().take(s.budget.route_probe_hit_limit).map(|(m, _)| m.clone())
        })
        .collect();
    let probe_identity = s.tenant.memory_identity_batch(&probe_ids).unwrap_or_else(|err| {
        tracing::warn!(error = ?err, "route probe identity lookup failed");
        HashMap::new()
    });
    for (probe_idx, route_probe_hits) in route_probe_results {
        let vote_weight: f32 = match probe_idx {
            0 => 3.0,
            1 => 2.0,
            _ => 1.0,
        };
        let mut seen_in_probe: std::collections::HashSet<String> = std::collections::HashSet::new();
        for (memory_id, _) in route_probe_hits.iter().take(s.budget.route_probe_hit_limit) {
            let session_id = probe_identity
                .get(memory_id)
                .map(|(session, _)| session.clone())
                .filter(|session| !session.is_empty())
                .or_else(|| routed_session_from_memory_id(memory_id));
            if let Some(session_id) = session_id {
                if seen_in_probe.insert(session_id.clone()) {
                    probe_lane.push((session_id, vote_weight));
                }
            }
        }
    }
    if !probe_lane.is_empty() {
        lanes.push(probe_lane);
    }

    // Lane 5: if we have no entity-anchored hits, fall back to raw FTS probe on
    // the plain question text. Cheap insurance.
    if lanes.is_empty() {
        let tenant = s.tenant.clone();
        let q = s.query_text.clone();
        let eid = s.payload.entity_id.clone();
        if let Ok(hits) = tenant.search_session_router(
            eid.as_deref().unwrap_or(""),
            &q,
            &[],
            &[],
            &s.plan.subject_entities,
            24,
        ) {
            let lane: Vec<(String, f32)> =
                hits.into_iter().map(|h| (h.session_id, h.score)).collect();
            if !lane.is_empty() {
                lanes.push(lane);
            }
        }
    }

    // Fuse with RRF (c=60, standard constant).
    let fused = rrf_fuse(&lanes, 60.0);

    // Also seed the legacy aggregate for downstream diagnostics.
    s.route.session_scores = fused.iter().cloned().collect();

    let route_take = if s.plan.needs_decomposition || s.plan.cross_entity {
        s.budget.route_take_hard
    } else {
        s.budget.route_take_simple
    };
    s.adaptive_profile.route_sessions =
        fused.into_iter().take(route_take).map(|(session, _)| session).collect();
    s.adaptive_profile.route_strength = if s.adaptive_profile.route_sessions.is_empty() {
        0.0
    } else if s.plan.needs_decomposition || s.plan.cross_entity {
        1.0
    } else {
        0.65
    };
    s.diag.routed_sessions = s.adaptive_profile.route_sessions.len() as u64;
    (s.diag.route_ms, s.diag.route_us) = elapsed_ms_and_us(s.route_start);
}
