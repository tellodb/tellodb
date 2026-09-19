use super::*;

pub(crate) fn score_hydrate(s: &mut QueryPipelineState) -> EngineResult<()> {
    let observation_keys: Vec<(u64, String)> =
        s.fused.items.iter().map(|(mid, ts, _)| (*ts, mid.clone())).collect();
    let observation_memory_ids: Vec<String> =
        observation_keys.iter().map(|(_, mid)| mid.clone()).collect();
    let pit = s.payload.point_in_time_ms;
    let tenant = s.tenant.as_ref();

    fn timed<T>(f: impl FnOnce() -> anyhow::Result<T>) -> (anyhow::Result<T>, Duration) {
        let start = Instant::now();
        let out = f();
        (out, start.elapsed())
    }

    let (obs, cards, invalid) = std::thread::scope(|scope| {
        let obs = scope.spawn(|| timed(|| tenant.get_observations_batch(&observation_keys)));
        let cards =
            scope.spawn(|| timed(|| tenant.get_memory_cards_batch(&observation_memory_ids)));
        let invalid = scope.spawn(|| {
            timed(|| match pit {
                Some(pit_ms) => tenant.invalidated_set_at_time(pit_ms, &observation_memory_ids),
                None => tenant.invalidated_set(&observation_memory_ids),
            })
        });
        fn join<T>(name: &'static str, r: std::thread::Result<T>) -> EngineResult<T> {
            r.map_err(|_panic| EngineError::internal(format!("hydrate stage panicked: {name}")))
        }
        Ok::<_, EngineError>((
            join("observations", obs.join())?,
            join("cards", cards.join())?,
            join("invalidated", invalid.join())?,
        ))
    })?;

    let fail = |name: &'static str| {
        move |err: anyhow::Error| EngineError::Other(anyhow::anyhow!("{name}: {err}"))
    };
    let us = |d: Duration| (d.as_millis() as u64, d.as_micros() as u64);
    s.scoring.observations = obs.0.map_err(fail("observations"))?;
    s.scoring.scorables = s
        .scoring
        .observations
        .iter()
        .map(|(memory_id, observation)| {
            (memory_id.clone(), ScorableObservation::new(&observation.textual_content))
        })
        .collect();
    (s.diag.fetch_obs_ms, s.diag.fetch_obs_us) = us(obs.1);
    s.scoring.memory_cards = cards.0.map_err(fail("cards"))?;
    (s.diag.fetch_cards_ms, s.diag.fetch_cards_us) = us(cards.1);
    s.scoring.invalidated_facts = invalid.0.map_err(fail("invalidated"))?;
    (s.diag.fetch_invalid_ms, s.diag.fetch_invalid_us) = us(invalid.1);

    // Graph, link and edge lanes traverse tenant-wide structures, so they can
    // surface another entity's memories. Enforce the requested scope once,
    // here, where every candidate's owner is known.
    if let Some(scope) = s.payload.entity_id.clone() {
        let (observations, cards) = (&s.data.scoring.observations, &s.data.scoring.memory_cards);
        let in_scope = |mid: &String| {
            observations.get(mid).map_or(true, |o| o.entity_id == scope)
                && cards.get(mid).map_or(true, |c| c.entity_id == scope)
        };
        s.data.fused.items.retain(|(mid, _, _)| in_scope(mid));
        let kept: HashSet<String> =
            s.data.fused.items.iter().map(|(mid, _, _)| mid.clone()).collect();
        s.data.scoring.observations.retain(|mid, _| kept.contains(mid));
        s.data.scoring.scorables.retain(|mid, _| kept.contains(mid));
    }
    Ok(())
}

pub(crate) fn score_loop(s: &mut QueryPipelineState) -> Vec<EvidenceCard> {
    let loop_start = Instant::now();
    let mut scored = Vec::new();
    let primary_qembed = &s.primary_qembed;
    let graph_scores = &s.scoring.graph_scores;
    let memory_cards = &s.scoring.memory_cards;
    let observations = &s.scoring.observations;
    let invalidated_facts = &s.scoring.invalidated_facts;
    let scorables = &s.scoring.scorables;
    let plan = &s.plan;
    let plan_intent = s.plan.intent;
    let query_text = &s.query_text;
    let now_ms = s.now_ms;
    let adaptive_profile = &s.adaptive_profile;
    let session_route_scores = &s.route.session_scores;

    for (mid, ts, rrf_score) in &s.fused.items {
        if MemoryId::parse(mid).ok().is_some_and(|id| id.tags().contains(&Tag::SyntheticQuery)) {
            continue;
        }
        let Some(obs) = observations.get(mid) else {
            continue;
        };
        let is_stale_fact = (obs.kind == MemoryKind::Fact
            || obs.kind == MemoryKind::Preference
            || obs.kind == MemoryKind::Decision)
            && invalidated_facts.contains(mid);
        let created_at_ms = if obs.created_at_ms > 0 { obs.created_at_ms } else { *ts };
        if let Some(pit) = s.payload.point_in_time_ms {
            if created_at_ms > pit {
                continue;
            }
        }
        let Some(scorable) = scorables.get(mid) else {
            continue;
        };
        let entity_hits = entity_hit_count(scorable, plan);
        let lexical_hits = lexical_hit_count(scorable, plan);
        let temporal_hits = temporal_hit_count(scorable, plan);
        let facet_mask = facet_match_mask(scorable, plan);
        // Cross-encoder scores are unbounded logits; they enter through their
        // own lane in the rank fusion above. Using them directly here put
        // reranked candidates on a different scale from everything else.
        let mut base_score = *rrf_score;
        if base_score <= 0.001
            && !primary_qembed.is_empty()
            && obs.embedding.len() == primary_qembed.len()
        {
            base_score = cosine_similarity(primary_qembed, &obs.embedding).max(0.0);
        }
        let lifecycle =
            memory_cards.get(mid).and_then(|card| card.lifecycle.as_ref()).cloned().unwrap_or_else(
                || {
                    let mut lifecycle = crate::lifecycle::evaluate_lifecycle(
                        &obs.textual_content,
                        obs.kind,
                        created_at_ms,
                        None,
                        false,
                    );
                    // No stored lifecycle means no known storage time; never
                    // expire on the event timestamp alone.
                    lifecycle.expires_at_ms = None;
                    lifecycle
                },
            );
        let Some(lifecycle_adjustment) = lifecycle_rank_adjustment(&lifecycle, obs.kind, now_ms)
        else {
            continue;
        };
        let mut fs = apply_decay_with_policy(base_score, created_at_ms, obs.kind, now_ms);
        fs += lifecycle_adjustment;
        let superseded_card = memory_cards.get(mid).is_some_and(|card| !card.is_latest);
        if is_stale_fact || (plan.prefers_latest && superseded_card) {
            fs *= s.weights.stale_fact_decay;
        }
        fs -= attractor_negative_penalty(
            scorable,
            plan,
            query_text,
            entity_hits,
            lexical_hits,
            temporal_hits,
            facet_mask,
        );
        fs += kind_query_bonus(obs.kind, plan, scorable);
        fs += lexical_overlap_bonus(scorable, plan);
        fs += entity_coverage_bonus(scorable, plan);
        fs += numeric_signal_bonus(&scorable.lower, &scorable.numeric_tokens, plan_intent);
        fs += ordinal_signal_bonus(obs.kind, scorable, plan);

        let graph_score = graph_scores.get(mid).copied().unwrap_or(0.0);
        let temporal_adjust = if s.state.config.temporal.recency_scoring {
            temporal_consistency_adjustment(obs.kind, created_at_ms, now_ms, plan_intent)
        } else {
            0.0
        };
        let confidence_signal: f32 = if is_stale_fact {
            s.weights.rerank_stale_penalty
        } else if lifecycle.stability_score > 0.7 {
            s.weights.rerank_confidence_stable
        } else if lifecycle.confidence_score > 0.7 {
            s.weights.rerank_confidence_high
        } else {
            0.0
        };

        let weights = FourSignalWeights::for_intent(plan_intent);
        let semantic_signal = base_score.max(0.0);
        let temporal_signal = temporal_adjust.max(0.0);
        let reweighted = fuse_four_signals(
            semantic_signal,
            temporal_signal,
            confidence_signal.max(0.0),
            graph_score.max(0.0),
            &weights,
        );
        fs = fs * (1.0 - s.weights.four_signal_temporal_weight)
            + reweighted * s.weights.four_signal_temporal_weight;
        if adaptive_profile.route_strength > 0.0 {
            let routed_sid = memory_cards
                .get(mid)
                .map(|card| card.source_session_id.clone())
                .or_else(|| Some(obs.session_id.clone()).filter(|s| !s.is_empty()));
            if let Some(sid) = routed_sid {
                if adaptive_profile.route_sessions.contains(&sid) {
                    let route_score = session_route_scores.get(&sid).copied().unwrap_or(0.0);
                    fs += if plan.needs_decomposition || plan.cross_entity {
                        s.weights.route_boost_hard
                    } else {
                        s.weights.route_boost_simple
                    };
                    fs += route_score.min(0.35) * 0.18;
                } else if !(plan.needs_decomposition || plan.cross_entity) {
                    fs += s.weights.route_penalty;
                }
            }
        }
        let (source_memory_id, source_session_id) = if let Some(card) = memory_cards.get(mid) {
            if card.source_memory_id != *mid {
                (card.source_memory_id.clone(), card.source_session_id.clone())
            } else {
                (mid.clone(), obs.session_id.clone())
            }
        } else {
            (mid.clone(), obs.session_id.clone())
        };

        scored.push(EvidenceCard {
            claim_text: obs.textual_content.clone(),
            source_memory_id,
            source_session_id,
            card_id: if memory_cards.contains_key(mid) { Some(mid.clone()) } else { None },
            semantic_rank: None,
            semantic_score: base_score,
            bm25_rank: None,
            bm25_score: 0.0,
            session_router_rank: None,
            session_router_score: 0.0,
            card_score: 0.0,
            reranker_score: base_score,
            entity_hits,
            lexical_hits,
            temporal_hits,
            facet_mask,
            graph_score,
            child_score: 0.0,
            is_latest: false,
            card_type: format!("{:?}", obs.kind),
            final_score: fs,
            inference_notes: None,
            internal_kind: obs.kind,
            created_at_ms,
            entity_id: obs.entity_id.clone(),
            source_turn_index: obs.turn_index as usize,
        });
    }
    if plan.prefers_latest {
        apply_latest_preference(&mut scored, s.state.config.retrieval.latest_recency_weight);
    }
    (s.diag.scoring_loop_ms, s.diag.scoring_loop_us) = elapsed_ms_and_us(loop_start);
    scored
}

/// For current-value questions, adds a bonus that grows with how recent a
/// candidate is relative to the other candidates, scaled by the score spread
/// and by the candidate's own (squared) relative score, so newer versions of
/// relevant facts win without promoting recent unrelated memories
/// (`TELLODB_LATEST_RECENCY_WEIGHT`, default 0.35).
pub(crate) fn apply_latest_preference(scored: &mut [EvidenceCard], weight: f32) {
    let (Some(oldest), Some(newest)) = (
        scored.iter().map(|c| c.created_at_ms).min(),
        scored.iter().map(|c| c.created_at_ms).max(),
    ) else {
        return;
    };
    if newest == oldest {
        return;
    }
    let (lo, hi) = scored
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), c| (lo.min(c.final_score), hi.max(c.final_score)));
    let spread = (hi - lo).max(f32::EPSILON);
    for card in scored.iter_mut() {
        let recency = (card.created_at_ms - oldest) as f32 / (newest - oldest) as f32;
        // Gate by relevance: recency decides between relevant versions of a
        // fact; it must not lift recent but unrelated memories above them.
        let relevance = ((card.final_score - lo) / spread).clamp(0.0, 1.0);
        card.final_score += weight * spread * recency * relevance * relevance;
    }
}

/// Explains why a retrieved fact is no longer current: what replaced it,
/// when, and which memories state each value. `None` while it is current.
pub(crate) fn describe_stale_fact(
    version: &crate::storage::FactVersionRow,
) -> Option<crate::api::types::WhyStale> {
    if version.is_current {
        return None;
    }
    let superseded_by = version.superseded_by.clone()?;
    Some(crate::api::types::WhyStale {
        fact_key: version.fact_key.clone(),
        stale_value: version.object.clone(),
        current_value: version.current_object.clone(),
        superseded_by,
        superseded_at_ms: version.superseded_at_ms.or(version.valid_to_ms),
        valid_from_ms: version.valid_from_ms,
        valid_to_ms: version.valid_to_ms,
        evidence: version.evidence.clone(),
    })
}
