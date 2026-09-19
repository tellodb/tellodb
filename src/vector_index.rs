//! Per-entity vector segments.
//!
//! SQLite (`vector_lookup.embedding`) is the source of truth; this index is a
//! cache of it, loaded one entity at a time on first use. Each entity's
//! segment is searched on its own, so an entity-scoped query never scans or
//! filters other entities' vectors and entities never contend on one lock.
//!
//! - Segments with at most `TELLODB_FLAT_THRESHOLD` vectors (default 20,000)
//!   are searched exactly by a linear scan.
//! - Larger segments get their own usearch HNSW graph.
//!
//! `TELLODB_VECTOR_QUANT` sets how vectors are held in memory:
//! `f32` (exact), `f16`, `i8` (per-vector scale) or `binary` (sign bits).
//! Quantized searches take the top `k × TELLODB_RESCORE_FACTOR` (default 4)
//! candidates and rescore them with the f32 vectors from the source.
//!
//! Distances returned are cosine distances (`1 - cosine similarity`).

use anyhow::{Context, Result};
use half::{f16, slice::HalfFloatSliceExt};
use parking_lot::{RwLock, RwLockUpgradableReadGuard};
use std::collections::HashMap;
use std::sync::Arc;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

fn swap_row<T: Copy>(data: &mut Vec<T>, row: usize, last: usize, width: usize) {
    if row != last {
        data.copy_within(last * width..(last + 1) * width, row * width);
    }
    data.truncate(last * width);
}

/// Where segment vectors come from. `record_*` hooks let a source that is not
/// already durable (the in-memory source used in tests and benchmarks)
/// observe writes; the SQLite source ignores them because ingest has already
/// committed the rows.
pub trait VectorSource: Send + Sync {
    /// Every `(vector_id, vector)` stored for `entity_id`.
    fn entity_vectors(&self, entity_id: &str) -> Result<Vec<(u64, Vec<f32>)>>;
    /// Entities with at least one stored vector.
    fn entities(&self) -> Result<Vec<String>>;
    /// Stored vectors by id (missing ids are omitted).
    fn vectors_by_id(&self, ids: &[u64]) -> Result<HashMap<u64, Vec<f32>>>;

    fn record_insert(&self, _entity_id: &str, _items: &[(u64, Vec<f32>)]) {}
    fn record_remove(&self, _entity_id: &str, _id: u64) {}
    fn record_clear(&self, _entity_id: Option<&str>) {}
}

/// A `VectorSource` that only lives in memory.
#[derive(Default)]
pub struct MemoryVectorSource {
    entities: RwLock<HashMap<String, HashMap<u64, Vec<f32>>>>,
}

impl VectorSource for MemoryVectorSource {
    fn entity_vectors(&self, entity_id: &str) -> Result<Vec<(u64, Vec<f32>)>> {
        let entities = self.entities.read();
        let mut rows: Vec<(u64, Vec<f32>)> = entities
            .get(entity_id)
            .map(|m| m.iter().map(|(id, v)| (*id, v.clone())).collect())
            .unwrap_or_default();
        rows.sort_by_key(|(id, _)| *id);
        Ok(rows)
    }

    fn entities(&self) -> Result<Vec<String>> {
        let mut names: Vec<String> = self
            .entities
            .read()
            .iter()
            .filter(|(_, m)| !m.is_empty())
            .map(|(e, _)| e.clone())
            .collect();
        names.sort();
        Ok(names)
    }

    fn vectors_by_id(&self, ids: &[u64]) -> Result<HashMap<u64, Vec<f32>>> {
        let entities = self.entities.read();
        Ok(ids
            .iter()
            .filter_map(|id| entities.values().find_map(|m| m.get(id)).map(|v| (*id, v.clone())))
            .collect())
    }

    fn record_insert(&self, entity_id: &str, items: &[(u64, Vec<f32>)]) {
        let mut entities = self.entities.write();
        for (id, _) in items {
            for (name, vectors) in entities.iter_mut() {
                if name != entity_id {
                    vectors.remove(id);
                }
            }
        }
        let vectors = entities.entry(entity_id.to_string()).or_default();
        for (id, vector) in items {
            vectors.insert(*id, vector.clone());
        }
    }

    fn record_remove(&self, entity_id: &str, id: u64) {
        if let Some(vectors) = self.entities.write().get_mut(entity_id) {
            vectors.remove(&id);
        }
    }

    fn record_clear(&self, entity_id: Option<&str>) {
        let mut entities = self.entities.write();
        match entity_id {
            Some(e) => {
                entities.remove(e);
            }
            None => entities.clear(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quantization {
    F32,
    F16,
    I8,
    Binary,
}

impl Quantization {
    pub fn parse(name: &str) -> Result<Self> {
        Ok(match name.trim().to_ascii_lowercase().as_str() {
            "" | "f32" => Quantization::F32,
            "f16" => Quantization::F16,
            "i8" => Quantization::I8,
            "binary" | "b1" => Quantization::Binary,
            other => anyhow::bail!("unknown TELLODB_VECTOR_QUANT '{other}' (f32, f16, i8, binary)"),
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Quantization::F32 => "f32",
            Quantization::F16 => "f16",
            Quantization::I8 => "i8",
            Quantization::Binary => "binary",
        }
    }

    /// In-memory bytes per vector in a flat segment.
    pub fn bytes_per_vector(self, dimensions: usize) -> usize {
        match self {
            Quantization::F32 => 4 * dimensions,
            Quantization::F16 => 2 * dimensions,
            Quantization::I8 => dimensions + 4,
            Quantization::Binary => 8 * dimensions.div_ceil(64),
        }
    }

    fn hnsw_scalar(self) -> ScalarKind {
        match self {
            Quantization::F32 => ScalarKind::F32,
            Quantization::F16 => ScalarKind::F16,
            // usearch's binary kind needs a Hamming metric; i8 plus rescoring
            // keeps cosine semantics.
            Quantization::I8 | Quantization::Binary => ScalarKind::I8,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VectorConfig {
    pub dimensions: usize,
    pub quantization: Quantization,
    /// Segments larger than this use HNSW instead of a flat scan.
    pub flat_threshold: usize,
    /// Quantized searches rescore `k × rescore_factor` candidates with f32.
    pub rescore_factor: usize,
    pub connectivity: usize,
    pub expansion_add: usize,
    pub expansion_search: usize,
}

impl VectorConfig {
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions,
            quantization: Quantization::F32,
            flat_threshold: 20_000,
            rescore_factor: 4,
            connectivity: 16,
            expansion_add: 128,
            expansion_search: 256,
        }
    }

    /// Reads `TELLODB_VECTOR_QUANT`, `TELLODB_FLAT_THRESHOLD`,
    /// `TELLODB_RESCORE_FACTOR` and `TELLODB_HNSW_{CONNECTIVITY,EF_ADD,EF_SEARCH}`.
    pub(crate) fn from_values(
        dimensions: usize,
        quantization: Option<&str>,
        flat_threshold: Option<usize>,
        rescore_factor: Option<usize>,
        connectivity: Option<usize>,
        expansion_add: Option<usize>,
        expansion_search: Option<usize>,
    ) -> Result<Self> {
        let defaults = Self::new(dimensions);
        Ok(Self {
            dimensions,
            quantization: Quantization::parse(quantization.unwrap_or_default())?,
            flat_threshold: flat_threshold.unwrap_or(defaults.flat_threshold),
            rescore_factor: rescore_factor
                .filter(|&factor| factor >= 1)
                .unwrap_or(defaults.rescore_factor),
            connectivity: connectivity.unwrap_or(defaults.connectivity),
            expansion_add: expansion_add.unwrap_or(defaults.expansion_add),
            expansion_search: expansion_search.unwrap_or(defaults.expansion_search),
        })
    }
}

fn normalized(vector: &[f32]) -> Vec<f32> {
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        vector.iter().map(|x| x / norm).collect()
    } else {
        vector.to_vec()
    }
}

/// Dot product written so the compiler vectorizes it (eight independent
/// accumulators; a single running float sum cannot be reordered).
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let chunks_a = a.chunks_exact(8);
    let chunks_b = b.chunks_exact(8);
    let tail: f32 = chunks_a.remainder().iter().zip(chunks_b.remainder()).map(|(x, y)| x * y).sum();
    for (ca, cb) in chunks_a.zip(chunks_b) {
        for i in 0..8 {
            acc[i] += ca[i] * cb[i];
        }
    }
    acc.iter().sum::<f32>() + tail
}

fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    a.iter().zip(b).map(|(x, y)| i32::from(*x) * i32::from(*y)).sum()
}

fn quantize_i8(v: &[f32]) -> (Vec<i8>, f32) {
    let max = v.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    if max <= f32::EPSILON {
        return (vec![0; v.len()], 0.0);
    }
    let scale = max / 127.0;
    (v.iter().map(|x| (x / scale).round().clamp(-127.0, 127.0) as i8).collect(), scale)
}

fn sign_bits(v: &[f32]) -> Vec<u64> {
    let mut words = vec![0u64; v.len().div_ceil(64)];
    for (i, x) in v.iter().enumerate() {
        if *x > 0.0 {
            words[i / 64] |= 1 << (i % 64);
        }
    }
    words
}

/// Flat storage in the configured precision; row `i` belongs to `ids[i]`.
enum Codes {
    F32(Vec<f32>),
    F16(Vec<f16>),
    I8 { codes: Vec<i8>, scales: Vec<f32> },
    Binary(Vec<u64>),
}

struct FlatSegment {
    dims: usize,
    ids: Vec<u64>,
    positions: HashMap<u64, usize>,
    codes: Codes,
}

impl FlatSegment {
    fn new(dims: usize, quantization: Quantization) -> Self {
        let codes = match quantization {
            Quantization::F32 => Codes::F32(Vec::new()),
            Quantization::F16 => Codes::F16(Vec::new()),
            Quantization::I8 => Codes::I8 { codes: Vec::new(), scales: Vec::new() },
            Quantization::Binary => Codes::Binary(Vec::new()),
        };
        Self { dims, ids: Vec::new(), positions: HashMap::new(), codes }
    }

    fn words(&self) -> usize {
        self.dims.div_ceil(64)
    }

    fn upsert(&mut self, id: u64, vector: &[f32]) {
        let v = normalized(vector);
        let words = self.words();
        let dims = self.dims;
        let row = if let Some(&row) = self.positions.get(&id) {
            row
        } else {
            let row = self.ids.len();
            self.ids.push(id);
            self.positions.insert(id, row);
            match &mut self.codes {
                Codes::F32(data) => data.resize(data.len() + dims, 0.0),
                Codes::F16(data) => data.resize(data.len() + dims, f16::ZERO),
                Codes::I8 { codes, scales } => {
                    codes.resize(codes.len() + dims, 0);
                    scales.push(0.0);
                }
                Codes::Binary(data) => data.resize(data.len() + words, 0),
            }
            row
        };
        match &mut self.codes {
            Codes::F32(data) => data[row * dims..(row + 1) * dims].copy_from_slice(&v),
            Codes::F16(data) => {
                for (slot, x) in data[row * dims..(row + 1) * dims].iter_mut().zip(&v) {
                    *slot = f16::from_f32(*x);
                }
            }
            Codes::I8 { codes, scales } => {
                let (q, scale) = quantize_i8(&v);
                codes[row * dims..(row + 1) * dims].copy_from_slice(&q);
                scales[row] = scale;
            }
            Codes::Binary(data) => {
                data[row * words..(row + 1) * words].copy_from_slice(&sign_bits(&v));
            }
        }
    }

    fn remove(&mut self, id: u64) -> bool {
        let Some(row) = self.positions.remove(&id) else {
            return false;
        };
        let last = self.ids.len() - 1;
        let (dims, words) = (self.dims, self.words());
        match &mut self.codes {
            Codes::F32(data) => swap_row(data, row, last, dims),
            Codes::F16(data) => swap_row(data, row, last, dims),
            Codes::I8 { codes, scales } => {
                swap_row(codes, row, last, dims);
                swap_row(scales, row, last, 1);
            }
            Codes::Binary(data) => swap_row(data, row, last, words),
        }
        self.ids.swap_remove(row);
        if row != last {
            self.positions.insert(self.ids[row], row);
        }
        true
    }

    /// Top `k` rows by (estimated) similarity, best first.
    fn scan(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        if self.ids.is_empty() || k == 0 {
            return Vec::new();
        }
        let dims = self.dims;
        let mut scored: Vec<(f32, u32)> = match &self.codes {
            Codes::F32(data) => data
                .chunks_exact(dims)
                .enumerate()
                .map(|(row, v)| (dot_f32(query, v), row as u32))
                .collect(),
            Codes::F16(data) => {
                // Bulk conversion uses the CPU's half-precision instructions
                // where available; per-element `to_f32` was ~8x slower.
                let mut row_f32 = vec![0.0f32; dims];
                data.chunks_exact(dims)
                    .enumerate()
                    .map(|(row, v)| {
                        v.convert_to_f32_slice(&mut row_f32);
                        (dot_f32(query, &row_f32), row as u32)
                    })
                    .collect()
            }
            Codes::I8 { codes, scales } => {
                let (q, q_scale) = quantize_i8(query);
                codes
                    .chunks_exact(dims)
                    .zip(scales)
                    .enumerate()
                    .map(|(row, (v, s))| (dot_i8(&q, v) as f32 * s * q_scale, row as u32))
                    .collect()
            }
            Codes::Binary(data) => {
                let q = sign_bits(query);
                let words = self.words();
                data.chunks_exact(words)
                    .enumerate()
                    .map(|(row, v)| {
                        let differing: u32 =
                            v.iter().zip(&q).map(|(a, b)| (a ^ b).count_ones()).sum();
                        // Share of agreeing signs, mapped to [-1, 1].
                        (1.0 - 2.0 * differing as f32 / dims as f32, row as u32)
                    })
                    .collect()
            }
        };
        let k = k.min(scored.len());
        let by_score = |a: &(f32, u32), b: &(f32, u32)| {
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1))
        };
        if k < scored.len() {
            scored.select_nth_unstable_by(k - 1, by_score);
            scored.truncate(k);
        }
        scored.sort_unstable_by(by_score);
        scored.into_iter().map(|(sim, row)| (self.ids[row as usize], sim)).collect()
    }
}

struct HnswSegment {
    index: Index,
}

impl HnswSegment {
    fn build(config: &VectorConfig, rows: &[(u64, Vec<f32>)]) -> Result<Self> {
        let options = IndexOptions {
            dimensions: config.dimensions,
            metric: MetricKind::Cos,
            quantization: config.quantization.hnsw_scalar(),
            connectivity: config.connectivity,
            expansion_add: config.expansion_add,
            expansion_search: config.expansion_search,
            multi: false,
        };
        let index = Index::new(&options).context("failed to create usearch index")?;
        index.reserve(rows.len().max(1024) * 2).context("failed to reserve HNSW capacity")?;
        let segment = Self { index };
        for (id, vector) in rows {
            segment.upsert(*id, vector)?;
        }
        Ok(segment)
    }

    fn upsert(&self, id: u64, vector: &[f32]) -> Result<()> {
        if self.index.contains(id) {
            self.index.remove(id).context("failed to replace HNSW vector")?;
        }
        if self.index.size() + 1 > self.index.capacity() {
            let capacity = (self.index.capacity().max(1024)).saturating_mul(2);
            self.index.reserve(capacity).context("failed to grow HNSW capacity")?;
        }
        self.index.add(id, &normalized(vector)).context("failed to add HNSW vector")
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u64, f32)>> {
        if self.index.size() == 0 {
            return Ok(Vec::new());
        }
        let matches = self.index.search(query, k).context("HNSW search failed")?;
        Ok(matches.keys.into_iter().zip(matches.distances).map(|(id, d)| (id, 1.0 - d)).collect())
    }
}

enum Segment {
    Flat(FlatSegment),
    Hnsw(HnswSegment),
}

impl Segment {
    fn len(&self) -> usize {
        match self {
            Segment::Flat(flat) => flat.ids.len(),
            Segment::Hnsw(hnsw) => hnsw.index.size(),
        }
    }

    fn bytes(&self, quantization: Quantization, dims: usize) -> usize {
        match self {
            Segment::Flat(flat) => flat.ids.len() * (quantization.bytes_per_vector(dims) + 8),
            Segment::Hnsw(hnsw) => hnsw.index.memory_usage(),
        }
    }
}

/// One entity's segment; `None` until loaded from the source.
#[derive(Default)]
struct Slot {
    segment: RwLock<Option<Segment>>,
}

pub struct VectorIndex {
    config: VectorConfig,
    source: Arc<dyn VectorSource>,
    slots: RwLock<HashMap<String, Arc<Slot>>>,
}

impl VectorIndex {
    pub fn new(config: VectorConfig, source: Arc<dyn VectorSource>) -> Self {
        Self { config, source, slots: RwLock::new(HashMap::new()) }
    }

    /// An index whose vectors live only in memory.
    pub fn in_memory(config: VectorConfig) -> Self {
        Self::new(config, Arc::new(MemoryVectorSource::default()))
    }

    pub fn config(&self) -> &VectorConfig {
        &self.config
    }

    fn slot(&self, entity_id: &str) -> Arc<Slot> {
        if let Some(slot) = self.slots.read().get(entity_id) {
            return slot.clone();
        }
        self.slots.write().entry(entity_id.to_string()).or_default().clone()
    }

    fn build_segment(&self, rows: &[(u64, Vec<f32>)]) -> Result<Segment> {
        if rows.len() > self.config.flat_threshold {
            return Ok(Segment::Hnsw(HnswSegment::build(&self.config, rows)?));
        }
        let mut flat = FlatSegment::new(self.config.dimensions, self.config.quantization);
        for (id, vector) in rows {
            self.check_dimensions(vector)?;
            flat.upsert(*id, vector);
        }
        Ok(Segment::Flat(flat))
    }

    fn check_dimensions(&self, vector: &[f32]) -> Result<()> {
        if vector.len() != self.config.dimensions {
            anyhow::bail!(
                "vector has dimension {} (expected {})",
                vector.len(),
                self.config.dimensions
            );
        }
        Ok(())
    }

    /// Runs `f` on the entity's segment, loading it from the source first.
    fn with_segment<T>(&self, entity_id: &str, f: impl FnOnce(&Segment) -> Result<T>) -> Result<T> {
        let slot = self.slot(entity_id);
        let guard = slot.segment.upgradable_read();
        if let Some(segment) = guard.as_ref() {
            return f(segment);
        }
        let mut write = RwLockUpgradableReadGuard::upgrade(guard);
        if write.is_none() {
            let rows = self
                .source
                .entity_vectors(entity_id)
                .with_context(|| format!("loading vectors for entity {entity_id}"))?;
            *write = Some(self.build_segment(&rows)?);
        }
        let read = parking_lot::RwLockWriteGuard::downgrade(write);
        f(read.as_ref().expect("segment loaded above"))
    }

    pub fn len(&self, entity_id: Option<&str>) -> Result<usize> {
        match entity_id {
            Some(e) => self.with_segment(e, |s| Ok(s.len())),
            None => self
                .source
                .entities()?
                .iter()
                .try_fold(0, |n, e| Ok(n + self.with_segment(e, |s| Ok(s.len()))?)),
        }
    }

    /// Approximate in-memory bytes of the loaded segments.
    pub fn loaded_bytes(&self) -> usize {
        let slots: Vec<Arc<Slot>> = self.slots.read().values().cloned().collect();
        slots
            .iter()
            .filter_map(|slot| {
                slot.segment
                    .read()
                    .as_ref()
                    .map(|s| s.bytes(self.config.quantization, self.config.dimensions))
            })
            .sum()
    }

    pub fn insert(&self, entity_id: &str, id: u64, vector: &[f32]) -> Result<()> {
        self.insert_batch(entity_id, &[(id, vector.to_vec())])
    }

    /// Adds or replaces vectors. Call after the vectors are durable in the
    /// source: an entity that is not loaded yet picks them up when it loads.
    pub fn insert_batch(&self, entity_id: &str, items: &[(u64, Vec<f32>)]) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        for (_, vector) in items {
            self.check_dimensions(vector)?;
        }
        self.source.record_insert(entity_id, items);
        let slot = self.slot(entity_id);
        let mut guard = slot.segment.write();
        let crossed_threshold = match guard.as_mut() {
            None => false,
            Some(Segment::Flat(flat)) => {
                for (id, vector) in items {
                    flat.upsert(*id, vector);
                }
                flat.ids.len() > self.config.flat_threshold
            }
            Some(Segment::Hnsw(hnsw)) => {
                for (id, vector) in items {
                    hnsw.upsert(*id, vector)?;
                }
                false
            }
        };
        if crossed_threshold {
            // Rebuilt as HNSW from the source on next use.
            *guard = None;
        }
        Ok(())
    }

    /// Nearest vectors as `(vector_id, cosine distance)`, closest first.
    /// `entity_id: None` searches every entity and merges the results.
    pub fn search(
        &self,
        entity_id: Option<&str>,
        query: &[f32],
        limit: usize,
    ) -> Result<Vec<(u64, f32)>> {
        self.check_dimensions(query)?;
        if limit == 0 {
            return Ok(Vec::new());
        }
        let query = normalized(query);
        let exact = self.config.quantization == Quantization::F32;
        let candidates =
            if exact { limit } else { limit.saturating_mul(self.config.rescore_factor) };

        let entities = match entity_id {
            Some(e) => vec![e.to_string()],
            None => self.source.entities()?,
        };
        let mut hits: Vec<(u64, f32)> = Vec::new();
        for entity in &entities {
            hits.extend(self.with_segment(entity, |segment| match segment {
                Segment::Flat(flat) => Ok(flat.scan(&query, candidates)),
                Segment::Hnsw(hnsw) => hnsw.search(&query, candidates),
            })?);
        }

        if !exact && !hits.is_empty() {
            let ids: Vec<u64> = hits.iter().map(|(id, _)| *id).collect();
            let full = self.source.vectors_by_id(&ids)?;
            hits = hits
                .into_iter()
                .filter_map(|(id, _)| full.get(&id).map(|v| (id, dot_f32(&query, &normalized(v)))))
                .collect();
        }
        hits.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0))
        });
        hits.truncate(limit);
        Ok(hits.into_iter().map(|(id, sim)| (id, 1.0 - sim)).collect())
    }

    /// Removes a vector; returns how many were removed from loaded segments.
    pub fn remove(&self, entity_id: &str, id: u64) -> Result<usize> {
        self.source.record_remove(entity_id, id);
        let slot = self.slot(entity_id);
        let mut guard = slot.segment.write();
        Ok(match guard.as_mut() {
            None => 0,
            Some(Segment::Flat(flat)) => usize::from(flat.remove(id)),
            Some(Segment::Hnsw(hnsw)) => {
                hnsw.index.remove(id).context("failed to remove HNSW vector")?
            }
        })
    }

    /// Drops loaded segments (one entity, or all). Call after the source rows
    /// are gone; segments reload from the source on next use.
    pub fn clear(&self, entity_id: Option<&str>) -> Result<()> {
        self.source.record_clear(entity_id);
        let mut slots = self.slots.write();
        match entity_id {
            Some(e) => {
                slots.remove(e);
            }
            None => slots.clear(),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        }
        fn vector(&mut self, dims: usize) -> Vec<f32> {
            normalized(&(0..dims).map(|_| self.next()).collect::<Vec<_>>())
        }
    }

    fn config(dims: usize, quantization: Quantization) -> VectorConfig {
        VectorConfig { quantization, ..VectorConfig::new(dims) }
    }

    #[test]
    fn scoped_search_returns_only_that_entity() {
        let index = VectorIndex::in_memory(config(3, Quantization::F32));
        index.insert("a", 1, &[0.8, 0.2, 0.1]).unwrap();
        index.insert("a", 2, &[0.1, 0.9, 0.2]).unwrap();
        index.insert("b", 3, &[0.8, 0.2, 0.1]).unwrap();

        let hits = index.search(Some("a"), &[0.8, 0.2, 0.1], 5).unwrap();
        assert_eq!(hits.iter().map(|h| h.0).collect::<Vec<_>>(), vec![1, 2]);
        assert!(hits[0].1.abs() < 1e-5);
        assert_eq!(index.search(None, &[0.8, 0.2, 0.1], 5).unwrap().len(), 3);
        assert_eq!(index.len(Some("a")).unwrap(), 2);
        assert_eq!(index.len(None).unwrap(), 3);
    }

    #[test]
    fn upsert_remove_and_clear() {
        let index = VectorIndex::in_memory(config(3, Quantization::F32));
        index.insert("e", 1, &[1.0, 0.0, 0.0]).unwrap();
        index.insert("e", 2, &[0.0, 1.0, 0.0]).unwrap();
        index.insert("e", 1, &[0.0, 0.0, 1.0]).unwrap();
        assert_eq!(index.len(Some("e")).unwrap(), 2);
        assert_eq!(index.search(Some("e"), &[0.0, 0.0, 1.0], 1).unwrap()[0].0, 1);

        assert_eq!(index.remove("e", 1).unwrap(), 1);
        assert_eq!(index.search(Some("e"), &[0.0, 0.0, 1.0], 5).unwrap().len(), 1);
        index.clear(Some("e")).unwrap();
        assert_eq!(index.len(Some("e")).unwrap(), 0);
    }

    #[test]
    fn unloaded_entities_load_from_source() {
        let source = Arc::new(MemoryVectorSource::default());
        source.record_insert("e", &[(7, vec![0.0, 1.0, 0.0])]);
        let index = VectorIndex::new(config(3, Quantization::F32), source.clone());
        assert_eq!(index.search(Some("e"), &[0.0, 1.0, 0.0], 1).unwrap()[0].0, 7);
        // A write to a loaded segment is visible without reloading.
        index.insert("e", 8, &[1.0, 0.0, 0.0]).unwrap();
        assert_eq!(index.search(Some("e"), &[1.0, 0.0, 0.0], 1).unwrap()[0].0, 8);
    }

    #[test]
    fn dimension_mismatch_is_an_error() {
        let index = VectorIndex::in_memory(config(3, Quantization::F32));
        assert!(index.insert("e", 1, &[1.0, 0.0]).is_err());
        assert!(index.search(Some("e"), &[1.0], 1).is_err());
    }

    fn recall_at_10(quantization: Quantization, flat_threshold: usize) -> f32 {
        let dims = 64;
        let mut rng = Rng(42);
        let rows: Vec<(u64, Vec<f32>)> = (0..2_000).map(|i| (i, rng.vector(dims))).collect();
        let exact = VectorIndex::in_memory(config(dims, Quantization::F32));
        let index =
            VectorIndex::in_memory(VectorConfig { flat_threshold, ..config(dims, quantization) });
        exact.insert_batch("e", &rows).unwrap();
        index.insert_batch("e", &rows).unwrap();
        let mut found = 0;
        for _ in 0..50 {
            let q = rng.vector(dims);
            let truth: Vec<u64> =
                exact.search(Some("e"), &q, 10).unwrap().iter().map(|h| h.0).collect();
            let got = index.search(Some("e"), &q, 10).unwrap();
            found += got.iter().filter(|h| truth.contains(&h.0)).count();
        }
        found as f32 / 500.0
    }

    #[test]
    fn quantized_and_hnsw_search_keep_recall() {
        assert!((recall_at_10(Quantization::F32, usize::MAX) - 1.0).abs() < f32::EPSILON);
        for quantization in [Quantization::F16, Quantization::I8] {
            let recall = recall_at_10(quantization, usize::MAX);
            assert!(recall >= 0.98, "{} recall {recall}", quantization.name());
        }
        // Uniform random vectors are the worst case for sign bits (no cluster
        // structure); require clearly better than the 2% a random candidate
        // set of k×4 would reach. `examples/vector_bench.rs` reports real
        // numbers on clustered data.
        let binary = recall_at_10(Quantization::Binary, usize::MAX);
        assert!(binary >= 0.2, "binary recall {binary}");
        assert!(recall_at_10(Quantization::F32, 100) >= 0.9, "HNSW segment");
    }

    #[test]
    fn flat_segment_remove_keeps_rows_aligned() {
        for quantization in
            [Quantization::F32, Quantization::F16, Quantization::I8, Quantization::Binary]
        {
            let mut flat = FlatSegment::new(3, quantization);
            flat.upsert(1, &[1.0, 0.0, 0.0]);
            flat.upsert(2, &[0.0, 1.0, 0.0]);
            flat.upsert(3, &[0.0, 0.0, 1.0]);
            assert!(flat.remove(1));
            assert!(!flat.remove(1));
            let top = flat.scan(&[0.0, 0.0, 1.0], 1);
            assert_eq!(top[0].0, 3, "{}", quantization.name());
            let top = flat.scan(&[0.0, 1.0, 0.0], 1);
            assert_eq!(top[0].0, 2, "{}", quantization.name());
        }
    }

    #[test]
    fn crossing_the_flat_threshold_switches_to_hnsw() {
        let dims = 8;
        let mut rng = Rng(3);
        let index = VectorIndex::in_memory(VectorConfig {
            flat_threshold: 10,
            ..config(dims, Quantization::F32)
        });
        let rows: Vec<(u64, Vec<f32>)> = (0..5).map(|i| (i, rng.vector(dims))).collect();
        index.insert_batch("e", &rows).unwrap();
        assert_eq!(index.len(Some("e")).unwrap(), 5);
        let more: Vec<(u64, Vec<f32>)> = (5..30).map(|i| (i, rng.vector(dims))).collect();
        index.insert_batch("e", &more).unwrap();
        assert_eq!(index.len(Some("e")).unwrap(), 30);
        let q = more[3].1.clone();
        assert_eq!(index.search(Some("e"), &q, 1).unwrap()[0].0, 8);
    }
}
