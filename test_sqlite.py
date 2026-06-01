import sqlite3

conn = sqlite3.connect(':memory:')
c = conn.cursor()
c.execute('''CREATE TABLE memories (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL UNIQUE,
    content TEXT NOT NULL
)''')

c.execute("INSERT INTO memories (memory_id, content) VALUES ('mem-001', 'Old text')")
print("First insert rowid:", c.lastrowid)

c.execute("INSERT OR REPLACE INTO memories (memory_id, content) VALUES ('mem-001', 'New text')")
print("Second insert rowid:", c.lastrowid)
