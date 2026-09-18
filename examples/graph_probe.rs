//! Times two-hop graph neighbour expansion on an existing tenant database.
//!
//! `cargo run --profile fastrelease --example graph_probe -- path/to/tellodb.db`
use std::time::Instant;
use tellodb::storage::TenantStore;

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("usage: graph_probe <tellodb.db>");
    let store = TenantStore::new(std::path::Path::new(&path))?;
    let conn = store.get_conn()?;
    let seeds: Vec<String> = conn
        .prepare("SELECT DISTINCT memory_id FROM edges ORDER BY memory_id LIMIT 24")?
        .query_map([], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    drop(conn);
    for degree in [i64::MAX as usize / 2, 128] {
        let start = Instant::now();
        let (mut queries, mut edges) = (0, 0);
        for seed in &seeds {
            let level0 = store.get_edge_cluster_neighbors_typed(seed, None, 50, degree)?;
            queries += 1;
            for (mid, _, _) in &level0 {
                edges += store.get_edge_cluster_neighbors_typed(mid, None, 50, degree)?.len();
                queries += 1;
            }
        }
        println!(
            "max_degree={:<20} 24 seeds: {:>8.1} ms, {} queries, {} depth-2 edges",
            degree,
            start.elapsed().as_secs_f64() * 1e3,
            queries,
            edges
        );
    }
    Ok(())
}
