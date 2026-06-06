CREATE VIRTUAL TABLE fts_memories USING fts5(memory_id UNINDEXED, content);
EXPLAIN QUERY PLAN DELETE FROM fts_memories WHERE memory_id = 'test';
