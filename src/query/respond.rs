use super::*;

fn build_proof_packet(
    tenant: &TenantStore,
    query_text: &str,
    plan: &QueryPlan,
    card: &EvidenceCard,
    proof_mode: &str,
    verify_evidence: bool,
    evidence_radius: u32,
) -> ProofPacket {
    let mut source_turns = Vec::new();
    if evidence_radius > 0 && !card.source_session_id.is_empty() {
        let center = card.source_turn_index as u32;
        if let Ok(turns) = tenant.get_turn_window(
            &card.entity_id,
            &card.source_session_id,
            center,
            evidence_radius,
        ) {
            source_turns = turns
                .into_iter()
                .map(|turn| ProofTurn {
                    turn_id: turn.turn_id,
                    session_id: turn.session_id,
                    turn_index: turn.turn_index,
                    speaker: turn.speaker,
                    text: turn.raw_text,
                })
                .collect();
        }
    }
    if source_turns.is_empty() {
        let turn_ids = vec![card.source_memory_id.clone()];
        if let Ok(turns) = tenant.get_ledger_turns_batch(&turn_ids) {
            source_turns = turns
                .into_values()
                .map(|turn| ProofTurn {
                    turn_id: turn.turn_id,
                    session_id: turn.session_id,
                    turn_index: turn.turn_index,
                    speaker: turn.speaker,
                    text: turn.raw_text,
                })
                .collect();
            source_turns.sort_by_key(|turn| turn.turn_index);
        }
    }
    if source_turns.is_empty() {
        source_turns.push(ProofTurn {
            turn_id: card.source_memory_id.clone(),
            session_id: card.source_session_id.clone(),
            turn_index: card.source_turn_index as u32,
            speaker: None,
            text: card.claim_text.clone(),
        });
    }

    let missing_facets = plan
        .coverage_facets
        .iter()
        .enumerate()
        .filter_map(|(idx, facet)| {
            if idx < 64 && (card.facet_mask & (1u64 << idx)) == 0 {
                Some(facet.text.clone())
            } else {
                None
            }
        })
        .take(8)
        .collect::<Vec<_>>();

    let entity_required = !plan.subject_entities.is_empty();
    let lexical_required = !plan.lexical_terms.is_empty();
    let temporal_required = !plan.temporal_terms.is_empty();
    let entity_ok = !entity_required || card.entity_hits > 0;
    let lexical_ok = !lexical_required || card.lexical_hits > 0;
    let temporal_ok = !temporal_required || card.temporal_hits > 0;
    let source_ok = !card.source_memory_id.is_empty() && !card.source_session_id.is_empty();
    let facet_ok = missing_facets.is_empty() || !plan.coverage_mode;
    let query_terms = crate::fts::tokenize_for_similarity(query_text);
    let proof_text = source_turns
        .iter()
        .map(|turn| turn.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase();
    let lexical_overlap = query_terms
        .iter()
        .filter(|term| term.len() > 3 && proof_text.contains(term.as_str()))
        .count();
    let lexical_trace_ok = query_terms.is_empty() || lexical_overlap > 0;

    let mut checks = vec![
        ProofCheck {
            name: "source_backed".to_string(),
            passed: source_ok,
            detail: card.source_memory_id.clone(),
        },
        ProofCheck {
            name: "entity_support".to_string(),
            passed: entity_ok,
            detail: format!("{} entity hit(s)", card.entity_hits),
        },
        ProofCheck {
            name: "lexical_support".to_string(),
            passed: lexical_ok && lexical_trace_ok,
            detail: format!(
                "{} lexical hit(s), {} proof overlap(s)",
                card.lexical_hits, lexical_overlap
            ),
        },
        ProofCheck {
            name: "temporal_support".to_string(),
            passed: temporal_ok,
            detail: format!("{} temporal hit(s)", card.temporal_hits),
        },
        ProofCheck {
            name: "facet_coverage".to_string(),
            passed: facet_ok,
            detail: format!("{} missing facet(s)", missing_facets.len()),
        },
    ];

    let verified = if verify_evidence {
        checks.iter().all(|check| check.passed)
    } else {
        checks.iter().filter(|check| check.name != "facet_coverage").all(|check| check.passed)
    };
    if !verify_evidence {
        checks.push(ProofCheck {
            name: "verification_mode".to_string(),
            passed: true,
            detail: "lightweight proof pack only".to_string(),
        });
    }

    let support_score = (card.entity_hits.min(3) as f32 * 0.10)
        + (card.lexical_hits.min(5) as f32 * 0.055)
        + (card.temporal_hits.min(2) as f32 * 0.075)
        + (card.facet_mask.count_ones().min(5) as f32 * 0.045)
        + if source_ok { 0.20 } else { 0.0 }
        + if verified { 0.15 } else { 0.0 };
    let confidence = support_score.clamp(0.05, 0.99);

    ProofPacket {
        proof_mode: proof_mode.to_string(),
        verified,
        confidence,
        source_memory_id: card.source_memory_id.clone(),
        source_session_id: card.source_session_id.clone(),
        source_turn_index: card.source_turn_index,
        supporting_card_ids: card.card_id.clone().into_iter().collect(),
        supporting_event_ids: Vec::new(),
        entities_hit: card.entity_hits,
        lexical_hits: card.lexical_hits,
        temporal_hits: card.temporal_hits,
        missing_facets,
        checks,
        source_turns,
    }
}

pub(crate) fn score_build_response(
    s: &mut QueryPipelineState,
    mut evidence_cards: Vec<EvidenceCard>,
) -> EngineResult<Vec<QueryResult>> {
    let stage_start = Instant::now();

    // Pre-synthesized Phase 1: direct fact lookup.
    // When the planner inferred a `fact_key` (e.g., "relationship_status",
    // "purchase", "favorite_team") and we have an entity scope, attempt a
    // deterministic lookup against the fact_versions table and inject the
    // answer as a high-priority synthetic EvidenceCard so the reader LLM
    // receives the fact verbatim at the top of its context.
    if let (Some(ref fact_key), Some(ref entity_id)) =
        (s.plan.fact_key.as_ref(), s.payload.entity_id.as_ref())
    {
        let fact_value = if s.state.config.features.enabled(Feature::Facts) {
            s.tenant.get_current_fact_value(entity_id, fact_key)
        } else {
            Ok(None)
        };
        if let Ok(Some(fact_value)) = fact_value {
            let synthetic_score = 1.0e9_f32;
            let now_ms = s.now_ms;
            let synthetic_id = MemoryId::new(entity_id.as_str(), "synthetic", 0)
                .derived(Tag::SyntheticQuery)
                .derived(Tag::Named("fact".to_string()))
                .as_str()
                .to_string();
            evidence_cards.push(EvidenceCard {
                claim_text: format!("{}: {}", fact_key.replace('_', " "), fact_value),
                source_memory_id: synthetic_id.clone(),
                source_session_id: String::new(),
                card_id: Some(synthetic_id),
                semantic_rank: None,
                semantic_score: synthetic_score,
                bm25_rank: None,
                bm25_score: 0.0,
                session_router_rank: None,
                session_router_score: 0.0,
                card_score: 0.0,
                reranker_score: synthetic_score,
                entity_hits: 0,
                lexical_hits: 0,
                temporal_hits: 0,
                facet_mask: 0,
                graph_score: 0.0,
                child_score: 0.0,
                is_latest: true,
                card_type: "PreSynthesizedFact".to_string(),
                final_score: synthetic_score,
                inference_notes: None,
                internal_kind: MemoryKind::Fact,
                created_at_ms: now_ms,
                entity_id: entity_id.to_string(),
                source_turn_index: 0,
            });
            tracing::debug!(
                entity_id = %entity_id,
                fact_key = %fact_key,
                "pre-synthesized fact lookup injected"
            );
        }
    }

    evidence_cards.sort_by(|a, b| {
        b.final_score
            .partial_cmp(&a.final_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.source_memory_id.cmp(&b.source_memory_id))
    });

    // Ambiguity packet: when the top two candidates are nearly tied (e.g.
    // "James" vs "John"), surface the close runner-up to the reader LLM via
    // the top card's `inference_notes` so the model can disambiguate or
    // ask the user. The threshold is loaded from `ranking_config.json` and
    // propagated into `s.weights.ambiguity_delta_threshold` at construction.
    if evidence_cards.len() >= 2 {
        let top_score = evidence_cards[0].final_score;
        let second_score = evidence_cards[1].final_score;
        let delta = (top_score - second_score).abs();
        if delta < s.weights.ambiguity_delta_threshold
            && !MemoryId::parse(&evidence_cards[0].source_memory_id)
                .ok()
                .is_some_and(|id| id.tags().contains(&Tag::SyntheticQuery))
        {
            let note = format!(
                "AmbiguityPacket: top-2 candidates are within {:.3} of each other ({} vs {}); consider asking the user to disambiguate.",
                delta,
                evidence_cards[0].source_memory_id,
                evidence_cards[1].source_memory_id
            );
            if let Some(card) = evidence_cards.get_mut(0) {
                if let Some(notes) = card.inference_notes.as_mut() {
                    notes.push(note);
                } else {
                    card.inference_notes = Some(vec![note]);
                }
            }
        }
    }

    // Pre-synthesized fact cards are extra context, not retrieved memories,
    // so they must not take slots from the requested `limit`.
    let synthetic_cards = evidence_cards
        .iter()
        .filter(|c| {
            MemoryId::parse(&c.source_memory_id)
                .ok()
                .is_some_and(|id| id.tags().contains(&Tag::SyntheticQuery))
        })
        .count();
    let selected = select_candidates_with_session_head(
        evidence_cards,
        s.limit + synthetic_cards,
        &s.plan,
        s.plan.prefer_distilled,
        s.plan.prefer_episodic,
    );

    let mut source_keys = Vec::new();
    for card in &selected {
        source_keys.push((card.created_at_ms, card.source_memory_id.clone()));
    }
    let hydrate_obs_start = Instant::now();
    let read_failed = |stage: &'static str| {
        move |err: anyhow::Error| EngineError::Other(anyhow::anyhow!("{stage}: {err}"))
    };
    let source_observations =
        s.tenant.get_observations_batch(&source_keys).map_err(read_failed("observations"))?;

    let mut fact_memory_ids = Vec::new();
    let mut card_ids = Vec::new();
    for card in &selected {
        if card.internal_kind == crate::storage::MemoryKind::Fact {
            fact_memory_ids.push(card.source_memory_id.clone());
        }
        if let Some(ref cid) = card.card_id {
            card_ids.push(cid.clone());
        }
    }

    let factver_start = Instant::now();
    let fact_versions = s
        .tenant
        .fact_versions_for_memories(&fact_memory_ids)
        .map_err(read_failed("fact_versions"))?;
    s.diag.factver_us = factver_start.elapsed().as_micros() as u64;
    let cards_start = Instant::now();
    let memory_cards =
        s.tenant.get_memory_cards_batch(&card_ids).map_err(read_failed("memory_cards"))?;
    s.diag.build_cards_us = cards_start.elapsed().as_micros() as u64;

    (s.diag.hydrate_obs_ms, s.diag.hydrate_obs_us) = elapsed_ms_and_us(hydrate_obs_start);

    let proof_us = std::sync::atomic::AtomicU64::new(0);
    let mut queries: Vec<QueryResult> = selected
        .into_iter()
        .map(|card| {
            let text = if let Some(source_obs) = source_observations.get(&card.source_memory_id) {
                source_obs.textual_content.clone()
            } else {
                card.claim_text.clone()
            };
            let evidence = if s.include_evidence && s.proof_mode != "off" {
                let proof_start = Instant::now();
                let packet = Some(build_proof_packet(
                    &s.tenant,
                    &s.query_text,
                    &s.plan,
                    &card,
                    &s.proof_mode,
                    s.verify_evidence,
                    s.evidence_radius,
                ));
                proof_us.fetch_add(
                    proof_start.elapsed().as_micros() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
                packet
            } else {
                None
            };
            let mut fact_key = None;
            let mut superseded_by = None;
            let mut why_stale = None;
            // Facts are registered against derived records, so a turn is
            // matched through them (see `fact_versions_for_memories`).
            if let Some(version) = fact_versions.get(&card.source_memory_id) {
                fact_key = Some(version.fact_key.clone());
                superseded_by = version.superseded_by.clone();
                why_stale = describe_stale_fact(version);
            }

            let mut stability_score = None;
            if let Some(ref cid) = card.card_id {
                if let Some(mc) = memory_cards.get(cid) {
                    if let Some(ref lc) = mc.lifecycle {
                        stability_score = Some(lc.stability_score);
                    }
                }
            }

            let origin = MemoryId::parse(&card.source_memory_id)
                .ok()
                .filter(|id| id.tags().contains(&Tag::SyntheticQuery))
                .map_or(ResultOrigin::Stored, |_| ResultOrigin::SynthesizedFact);
            QueryResult {
                memory_id: card.source_memory_id.clone(),
                entity_id: card.entity_id,
                session_id: card.source_session_id,
                turn_index: card.source_turn_index,
                created_at_ms: card.created_at_ms,
                similarity: card.final_score,
                textual_content: text,
                evidence,
                inference_notes: None,
                fact_key,
                conflict_flag: Some(!card.is_latest),
                superseded_by,
                why_stale,
                stability_score,
                origin,
            }
        })
        .collect();
    s.diag.proof_us = proof_us.load(std::sync::atomic::Ordering::Relaxed);
    let confidence_start = Instant::now();
    let evidence_conf =
        compute_evidence_confidence(&queries, &s.query_text, s.state.intent_classifier.as_deref());
    s.diag.confidence_us = confidence_start.elapsed().as_micros() as u64;
    s.diag.evidence_confidence_bp = (evidence_conf * 10_000.0) as u64;
    s.diag.abstain_recommended = evidence_conf < 0.24 && !queries.is_empty();
    (s.diag.session_ms, s.diag.session_us) = elapsed_ms_and_us(stage_start);
    (s.diag.total_ms, s.diag.total_us) = elapsed_ms_and_us(s.total_start);

    // Pre-synthesized Phase 2: memory card as answer.
    // If the top-ranked result is backed by a latest, high-confidence memory
    // card, surface its claim_text as a synthetic answer row at position 0
    // so the reader LLM receives the distilled claim verbatim.
    if let Some(top) = queries.first() {
        if top.origin.is_stored() {
            if let Ok(Some(card)) = s.tenant.get_memory_card_by_source(&top.memory_id) {
                if card.is_latest && card.confidence >= 0.70 {
                    // Dated like the memory it restates, so recency ordering holds.
                    let source_created_at_ms = top.created_at_ms;
                    let synthetic = QueryResult {
                        memory_id: MemoryId::new(&card.entity_id, "synthetic", 0)
                            .derived(Tag::SyntheticQuery)
                            .derived(Tag::Named("card".to_string()))
                            .as_str()
                            .to_string(),
                        entity_id: card.entity_id.clone(),
                        session_id: card.source_session_id.clone(),
                        turn_index: top.turn_index,
                        created_at_ms: source_created_at_ms,
                        similarity: 1.0,
                        textual_content: format!("{}: {}", card.subject, card.object),
                        evidence: None,
                        inference_notes: Some(vec![format!(
                            "Pre-synthesized from memory card {} (confidence {:.2})",
                            card.card_id, card.confidence
                        )]),
                        fact_key: None,
                        conflict_flag: Some(false),
                        superseded_by: None,
                        why_stale: None,
                        stability_score: None,
                        origin: ResultOrigin::SynthesizedCard,
                    };
                    queries.insert(0, synthetic);
                }
            }
        }
    }

    Ok(queries)
}
