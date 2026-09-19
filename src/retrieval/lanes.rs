//! Which retrieval lanes run, for baselines and lane ablations.
//!
//! `TELLODB_LANES` is an allowlist, because the comparisons a paper needs are
//! stated that way: "BM25 only", "vector only", "hybrid without reranking".
//! The default `all` keeps every lane on.
//!
//! ```text
//! TELLODB_LANES=fts            BM25 only
//! TELLODB_LANES=vector         dense retrieval only
//! TELLODB_LANES=vector,fts     RRF hybrid, nothing else
//! ```
//!
//! This gates retrieval only. It is deliberately separate from
//! [`crate::features`], which turns off structures that are *built at ingest*:
//! a lane can be switched per query without re-ingesting anything.

use anyhow::{bail, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lane {
    /// Dense vector search (ANN or flat scan).
    Vector,
    /// BM25 over the FTS index.
    Fts,
    /// Memory-card retrieval.
    Cards,
    /// Cross-encoder reranking.
    Rerank,
    /// Knowledge-graph expansion of the top fused candidates.
    Graph,
    /// Session routing, which narrows the candidate sessions first.
    Route,
}

impl Lane {
    pub const ALL: [Lane; 6] =
        [Lane::Vector, Lane::Fts, Lane::Cards, Lane::Rerank, Lane::Graph, Lane::Route];

    pub fn name(self) -> &'static str {
        match self {
            Lane::Vector => "vector",
            Lane::Fts => "fts",
            Lane::Cards => "cards",
            Lane::Rerank => "rerank",
            Lane::Graph => "graph",
            Lane::Route => "route",
        }
    }

    fn bit(self) -> u32 {
        1 << (self as u32)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lanes {
    enabled: u32,
}

impl Default for Lanes {
    fn default() -> Self {
        Self { enabled: Lane::ALL.iter().fold(0, |acc, lane| acc | lane.bit()) }
    }
}

impl Lanes {
    /// Parses an allowlist. `all` (or empty) enables everything; an unknown
    /// name is an error, so a typo cannot quietly run the full pipeline and
    /// be reported as a baseline.
    pub fn parse(spec: &str) -> Result<Self> {
        let spec = spec.trim();
        if spec.is_empty() || spec.eq_ignore_ascii_case("all") {
            return Ok(Self::default());
        }
        let mut enabled = 0;
        for name in spec.split(',').map(str::trim).filter(|n| !n.is_empty()) {
            let name = name.to_ascii_lowercase();
            let Some(lane) = Lane::ALL.iter().find(|l| l.name() == name) else {
                let known: Vec<&str> = Lane::ALL.iter().map(|l| l.name()).collect();
                bail!("unknown TELLODB_LANES entry '{name}'; known: all,{}", known.join(","));
            };
            enabled |= lane.bit();
        }
        if enabled == 0 {
            bail!("TELLODB_LANES disables every lane, so no query can return anything");
        }
        Ok(Self { enabled })
    }

    pub fn enabled(self, lane: Lane) -> bool {
        self.enabled & lane.bit() != 0
    }

    /// The active lanes, for `/version` and run records.
    pub fn enabled_names(self) -> Vec<&'static str> {
        Lane::ALL.iter().filter(|l| self.enabled(**l)).map(|l| l.name()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_and_all_enable_every_lane() {
        for spec in ["", "all", "ALL", "  "] {
            let l = Lanes::parse(spec).unwrap();
            assert!(Lane::ALL.iter().all(|lane| l.enabled(*lane)), "{spec:?}");
        }
    }

    #[test]
    fn allowlist_enables_only_what_it_names() {
        let l = Lanes::parse("fts").unwrap();
        assert!(l.enabled(Lane::Fts));
        assert!(!l.enabled(Lane::Vector) && !l.enabled(Lane::Rerank));
        assert_eq!(l.enabled_names(), vec!["fts"]);

        let hybrid = Lanes::parse("vector, fts").unwrap();
        assert_eq!(hybrid.enabled_names(), vec!["vector", "fts"]);
    }

    #[test]
    fn typos_and_empty_selections_are_errors() {
        assert!(Lanes::parse("ftz").is_err());
        assert!(Lanes::parse("vector,bogus").is_err());
        // `,` parses to nothing selected, which would silently return zero
        // results for every query.
        assert!(Lanes::parse(",").is_err());
    }

    #[test]
    fn every_lane_has_a_distinct_bit() {
        let all = Lanes::parse(&Lane::ALL.iter().map(|l| l.name()).collect::<Vec<_>>().join(","))
            .unwrap();
        assert_eq!(all, Lanes::default());
        assert_eq!(all.enabled_names().len(), Lane::ALL.len());
    }
}
