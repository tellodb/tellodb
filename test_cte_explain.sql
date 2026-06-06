CREATE TABLE memory_links (source_memory_id TEXT, target_memory_id TEXT, link_type TEXT);
CREATE INDEX idx_memory_links_target ON memory_links(target_memory_id);
CREATE INDEX idx_links_source ON memory_links(source_memory_id);
EXPLAIN QUERY PLAN
WITH RECURSIVE
  bfs(node, depth, path_weight) AS (
    SELECT 'A', 0, 1.0
    UNION ALL
    SELECT
      CASE WHEN ml.source_memory_id = bfs.node THEN ml.target_memory_id ELSE ml.source_memory_id END,
      bfs.depth + 1,
      bfs.path_weight * 0.6
    FROM bfs
    JOIN memory_links ml ON ml.source_memory_id = bfs.node OR ml.target_memory_id = bfs.node
    WHERE bfs.depth < 2
  )
SELECT node, SUM(path_weight) FROM bfs WHERE depth > 0 GROUP BY node;
