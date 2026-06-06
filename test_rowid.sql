CREATE TABLE m (rowid INTEGER PRIMARY KEY, memory_id TEXT UNIQUE);
CREATE VIRTUAL TABLE f USING fts5(memory_id UNINDEXED, content);
INSERT INTO m(memory_id) VALUES ('a');
INSERT INTO f(memory_id, content) VALUES ('a', 'content a');

INSERT INTO m(memory_id) VALUES ('b');
INSERT INTO f(memory_id, content) VALUES ('b', 'content b');

DELETE FROM f WHERE memory_id = 'a';
DELETE FROM m WHERE memory_id = 'a';

INSERT INTO m(memory_id) VALUES ('c');
INSERT INTO f(memory_id, content) VALUES ('c', 'content c');

SELECT rowid, memory_id FROM m;
SELECT rowid, memory_id FROM f;
