//! C ABI over [`tellodb::db::Db`]. Requests and responses are JSON strings,
//! so bindings only need to pass UTF-8 buffers.
//!
//! Every function returning `*mut c_char` returns either a JSON document or
//! NULL on error; call `tellodb_last_error` for the message and release every
//! non-NULL string with `tellodb_string_free`. Handles are safe to share
//! across threads.

use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};
use std::ptr;
use tellodb::db::{Db, Memory, Query};

pub struct TellodbHandle {
    db: Db,
}

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_error(message: impl std::fmt::Display) {
    let text = CString::new(message.to_string().replace('\0', " ")).unwrap_or_default();
    LAST_ERROR.with(|slot| *slot.borrow_mut() = Some(text));
}

/// # Safety
/// `ptr` must be NULL or a NUL-terminated string.
unsafe fn read_str<'a>(ptr: *const c_char, name: &str) -> anyhow::Result<&'a str> {
    anyhow::ensure!(!ptr.is_null(), "{name} is NULL");
    Ok(CStr::from_ptr(ptr).to_str()?)
}

fn into_c_string(result: anyhow::Result<String>) -> *mut c_char {
    match result.and_then(|s| Ok(CString::new(s)?)) {
        Ok(s) => s.into_raw(),
        Err(err) => {
            set_error(format!("{err:#}"));
            ptr::null_mut()
        }
    }
}

/// Opens (or creates) a data directory. Returns NULL on error.
///
/// # Safety
/// `data_dir` must be a NUL-terminated UTF-8 path.
#[no_mangle]
pub unsafe extern "C" fn tellodb_open(data_dir: *const c_char) -> *mut TellodbHandle {
    let opened = read_str(data_dir, "data_dir").and_then(Db::open);
    match opened {
        Ok(db) => Box::into_raw(Box::new(TellodbHandle { db })),
        Err(err) => {
            set_error(format!("{err:#}"));
            ptr::null_mut()
        }
    }
}

/// Stores memories from a JSON array of
/// `{"entity_id", "text", "session_id"?, "turn_index"?, "role"?, "timestamp_ms"?, "kind"?, "memory_id"?}`.
/// Returns an ingest report.
///
/// # Safety
/// `handle` must come from `tellodb_open`; `memories_json` must be NUL-terminated.
#[no_mangle]
pub unsafe extern "C" fn tellodb_ingest(
    handle: *const TellodbHandle,
    memories_json: *const c_char,
) -> *mut c_char {
    into_c_string((|| {
        let handle = handle.as_ref().ok_or_else(|| anyhow::anyhow!("handle is NULL"))?;
        let memories: Vec<Memory> =
            serde_json::from_str(read_str(memories_json, "memories_json")?)?;
        Ok(serde_json::to_string(&handle.db.ingest(memories)?)?)
    })())
}

/// Searches with `{"text", "entity_id"?, "limit"?, "as_of_ms"?, "reference_time_ms"?, "rerank"?}`.
/// Returns a JSON array of hits.
///
/// # Safety
/// `handle` must come from `tellodb_open`; `query_json` must be NUL-terminated.
#[no_mangle]
pub unsafe extern "C" fn tellodb_query(
    handle: *const TellodbHandle,
    query_json: *const c_char,
) -> *mut c_char {
    into_c_string((|| {
        let handle = handle.as_ref().ok_or_else(|| anyhow::anyhow!("handle is NULL"))?;
        let query: Query = serde_json::from_str(read_str(query_json, "query_json")?)?;
        Ok(serde_json::to_string(&handle.db.query(query)?)?)
    })())
}

/// Current value of a fact (`{"value": ...}`, `null` when unknown).
///
/// # Safety
/// `handle` must come from `tellodb_open`; strings must be NUL-terminated.
#[no_mangle]
pub unsafe extern "C" fn tellodb_current_fact(
    handle: *const TellodbHandle,
    entity_id: *const c_char,
    fact_key: *const c_char,
) -> *mut c_char {
    into_c_string((|| {
        let handle = handle.as_ref().ok_or_else(|| anyhow::anyhow!("handle is NULL"))?;
        let value = handle
            .db
            .current_fact(read_str(entity_id, "entity_id")?, read_str(fact_key, "fact_key")?)?;
        Ok(serde_json::json!({ "value": value }).to_string())
    })())
}

/// Message for the last error on this thread, or NULL. Owned by the library;
/// valid until the next call on this thread.
#[no_mangle]
pub extern "C" fn tellodb_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| slot.borrow().as_ref().map_or(ptr::null(), |s| s.as_ptr()))
}

/// Frees a string returned by this library.
///
/// # Safety
/// `s` must be NULL or a pointer returned by a `tellodb_*` function, freed once.
#[no_mangle]
pub unsafe extern "C" fn tellodb_string_free(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}

/// Closes the database (flushing its write-ahead log).
///
/// # Safety
/// `handle` must be NULL or come from `tellodb_open`, closed once.
#[no_mangle]
pub unsafe extern "C" fn tellodb_close(handle: *mut TellodbHandle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}
