#!/usr/bin/env bash
# E6 -- does an acknowledged write survive `kill -9`?
#
# The engine defaults to `synchronous = FULL`, which fsyncs the WAL on every
# commit, so a write the client saw a 2xx for should still be there after the
# power goes out. That is a claim about a pragma until something actually kills
# the process. TELLODB_DURABILITY=normal is the same harness with the fsync
# relaxed, which turns "you could lose up to a checkpoint interval" from an
# estimate into a measured number.
#
# WHAT THIS DOES AND DOES NOT PROVE. `kill -9` ends the process; it does not
# drop what the kernel has already buffered, and the OS goes on flushing those
# pages afterwards. So this measures durability across PROCESS DEATH, which is
# a real and reportable property, and it is the regime where
# `synchronous = NORMAL` is already safe. It is NOT power loss: that needs the
# page cache to die too (a VM reset, dm-flakey, or pulling the plug), and it is
# the only regime where FULL and NORMAL should diverge. Do not report a result
# from this harness as power-loss durability.
#
#   bash benchmarks/crash_injection.sh [trials] [writes-per-trial]
set -uo pipefail

REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
TRIALS="${1:-5}"
WRITES="${2:-40}"
PORT="${PORT:-3971}"
URL="http://127.0.0.1:${PORT}"
KEY="${TELLODB_API_KEY:-crash-harness-key}"
BIN="$REPO_ROOT/target/fastrelease/tellodb"

[[ -x "$BIN" ]] || { echo "build first: cargo build --profile fastrelease --bin tellodb" >&2; exit 1; }

start_engine() {
    local dir="$1" mode="$2" log="$3"
    # The previous trial's socket can still be in TIME_WAIT; without this the
    # next engine dies on bind and the trial is silently skipped.
    for _ in $(seq 1 30); do
        curl -sf "$URL/healthz" >/dev/null 2>&1 || break
        sleep 1
    done
    TELLODB_DATA_DIR="$dir" PORT="$PORT" TELLODB_API_KEY="$KEY" \
        TELLODB_DURABILITY="$mode" TELLODB_RERANK=off \
        "$BIN" serve >"$log" 2>&1 &
    ENGINE_PID=$!
    for _ in $(seq 1 120); do
        curl -sf "$URL/healthz" >/dev/null 2>&1 && return 0
        kill -0 "$ENGINE_PID" 2>/dev/null || { echo "engine died on startup" >&2; return 1; }
        sleep 1
    done
    return 1
}

# Counts what the store actually holds, read straight from the file once the
# engine is stopped. Going through the API would mean trusting the same
# process whose recovery is the thing under test.
stored_count() {
    # Only the turns the harness actually acknowledged. Ingest also writes
    # derived rows (chunks, companions, cards) into `memories` under the same
    # entity, and counting those would report more survivors than writes.
    sqlite3 "$1/tenants/default/tellodb.db" \
        "SELECT COUNT(*) FROM memories
          WHERE entity_id = 'crashtest' AND memory_id NOT LIKE '%::%::%::%';" 2>/dev/null \
        || echo unreadable
}

printf '%-8s %-6s %-12s %-10s %-10s %s\n' mode trial acknowledged survived lost integrity
for mode in full normal; do
    for trial in $(seq 1 "$TRIALS"); do
        dir="$(mktemp -d "${TMPDIR:-/tmp}/tellodb_crash.XXXXXX")"
        log="$dir/engine.log"
        start_engine "$dir" "$mode" "$log" || { echo "skip: engine would not start"; rm -rf "$dir"; continue; }

        acked=0
        for i in $(seq 1 "$WRITES"); do
            code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$URL/ingest" \
                -H "x-api-key: $KEY" -H 'content-type: application/json' \
                -d "{\"entity_id\":\"crashtest\",\"memory_id\":\"crashtest::s1::$i\",\"timestamp\":$((1700000000000+i)),\"session_id\":\"s1\",\"turn_index\":$i,\"role\":\"user\",\"textual_content\":\"durability probe number $i for the crash harness\",\"relations\":[]}")
            [[ "$code" == 2* ]] && acked=$((acked+1))
        done

        # No graceful shutdown, no checkpoint: this is the power cord.
        kill -9 "$ENGINE_PID" 2>/dev/null
        wait "$ENGINE_PID" 2>/dev/null

        # Reopening is itself the recovery test: the engine refuses a corrupt
        # or newer-schema file rather than limping on.
        if start_engine "$dir" "$mode" "$dir/engine2.log"; then
            reopened=ok
            kill "$ENGINE_PID" 2>/dev/null; wait "$ENGINE_PID" 2>/dev/null
        else
            reopened=FAILED
        fi
        survived=$(stored_count "$dir")
        integrity="$reopened/$(sqlite3 "$dir/tenants/default/tellodb.db" 'PRAGMA integrity_check;' 2>/dev/null | head -1)"

        printf '%-8s %-6s %-12s %-10s %-10s %s\n' \
            "$mode" "$trial" "$acked" "$survived" "$((acked-survived))" "$integrity"
        rm -rf "$dir"
    done
done
