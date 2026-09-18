/* C interface to tellodb. JSON in, JSON out. See crates/tellodb-ffi/src/lib.rs. */
#ifndef TELLODB_H
#define TELLODB_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TellodbHandle TellodbHandle;

/* Opens or creates a data directory. NULL on error (see tellodb_last_error). */
TellodbHandle *tellodb_open(const char *data_dir);

/* JSON array of memories -> JSON ingest report. NULL on error. */
char *tellodb_ingest(const TellodbHandle *handle, const char *memories_json);

/* JSON query -> JSON array of hits. NULL on error. */
char *tellodb_query(const TellodbHandle *handle, const char *query_json);

/* {"value": string|null}. NULL on error. */
char *tellodb_current_fact(const TellodbHandle *handle, const char *entity_id, const char *fact_key);

/* Last error on this thread, or NULL. Do not free. */
const char *tellodb_last_error(void);

/* Frees a string returned by tellodb_ingest/query/current_fact. */
void tellodb_string_free(char *s);

/* Flushes and closes the handle. */
void tellodb_close(TellodbHandle *handle);

#ifdef __cplusplus
}
#endif

#endif
