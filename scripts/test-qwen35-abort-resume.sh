#!/usr/bin/env bash
# Regression test for fix/deltanet-truncation-resume-guard.
#
# Bug: an early-truncated turn (client cancels mid-decode) used to leave the
# NON-REVERSIBLE DeltaNet recurrent state and `conversation_tokens` dirty with
# UNCOMMITTED tokens — the AR decode-abort path returned without resetting
# (unlike the DFlash + AR prefill-abort paths). The next turn then resumed the
# prompt cache off that poisoned prior, drifting the recurrent state
# off-distribution -> whitespace garbage / premature EOS, worse on each retry
# (observed live on qwen3.6-35b-a3b: prior_tail decayed real-code -> "   \tt   "
# -> EOS across three retries).
#
# Fix (cancel+edit prefix resume): the AR decode-abort path now PRESERVES the
# committed prefix instead of full-resetting. On cancel it keeps only
# checkpoints at or before the cancelled request's prompt length — each
# attested with a prefix hash over the committed tokens it covers — restores
# DeltaNet/conversation to the latest such one, and drops every decode-region
# checkpoint. The next turn's LCP therefore cannot reach into the uncommitted
# decode region, and an edited prompt resumes from the common prefix instead
# of cold-prefilling. Prefill-phase aborts, the explicit daemon reset and the
# model-swap path still full-reset (fail closed).
#
# DETERMINISTIC ASSERTIONS: after a real mid-decode abort,
#   * the next turn enters with `[qwen-cache GEN-ENTRY] conv_tok=<preserved
#     prompt length>` (was: 0) and the cancel line shows
#     `[qwen-cache cancel] preserved to checkpoint pos=<same>`;
#   * no `[qwen-cache resume] ... prefix-hash mismatch` appears, and any
#     `rewound` line resumes at or before the cancelled prompt length;
#   * the following turns stay coherent (no whitespace attractor / empty
#     garbage), including an unedited resend of the cancelled prompt.
# WITHOUT the fix, conv_tok would carry Turn A's partial generation (poison)
# and a retry could `rewound` off it.
#
# Requires the daemon running with HIPFIRE_QWEN_CACHE_TRACE=1 (for GEN-ENTRY /
# cache logs) and no speculator armed (--spec off / dflash_mode off — the
# cancel-preserve path declines to full reset when a speculator is loaded).
# Launch e.g.:
#   HIPFIRE_QWEN_CACHE_TRACE=1 HIPFIRE_MODEL=qwen3.6-35b-a3b.mq4 \
#     bash scripts/serve-restart.sh 11435
set -uo pipefail
PORT=${1:-11435}
MODEL=${HIPFIRE_TEST_MODEL:-qwen3.6-35b-a3b.mq4}
LOG=${HIPFIRE_SERVE_LOG:-$HOME/.hipfire/serve.log}
EP="http://127.0.0.1:$PORT/v1/chat/completions"
# Turn A prompt: short (< HIPFIRE_CACHE_CKPT_INTERVAL) so the first prefill
# chunk leaves a checkpoint at the prompt length, but with thinking ON and a
# hard ask it is guaranteed to STILL BE DECODING when we close the socket at
# 5s (it cannot finish a multi-section analysis in 5s, and won't hit a natural
# stop). This is what makes the cancel land in decode, not after.
PROMPT="Reason step by step in exhaustive detail, then write a comprehensive multi-section analysis comparing B-trees, LSM-trees, hash indexes, and tries for high-ingest time-series workloads. Cover concurrency control, cache behavior, and write amplification thoroughly."
# Turn B: an EDIT of Turn A's prompt (append-only). The render shares the
# prompt-text prefix with the preserved conversation (lcp > 0), but the
# chat-template trailer diverges from it and a short prompt has no checkpoint
# at or before lcp, so this takes the fail-closed cold reset — the guard's
# point is that it must NOT resume off uncommitted decode tokens. This is the
# user's cancel -> edit -> resend workflow.
PROMPT_EDITED="$PROMPT Also cover fractal trees and cache-oblivious layouts."

command -v jq >/dev/null || { echo "need jq"; exit 2; }
curl -fsS -m 10 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 \
  || { echo "daemon not responding on :$PORT — start it with $MODEL loaded first"; exit 2; }

LINES0=$(wc -l < "$LOG" 2>/dev/null || echo 0)

echo "=== Turn A: stream with thinking ON, close the socket mid-decode (curl -m 5) ==="
# Thinking ON + a hard prompt guarantees Turn A is still in <think> decode at 5s.
# curl -m 5 closes the socket -> native serve detects the disconnect ->
# sends {type:abort} -> AR decode-abort fires. `|| true`: -m exits 28.
curl -sS -N -m 5 "$EP" -H 'content-type: application/json' \
  -d "$(jq -cn --arg m "$MODEL" --arg p "$PROMPT" '{model:$m,messages:[{role:"user",content:$p}],max_tokens:4000,temperature:0,stream:true}')" \
  >/tmp/abort_turnA.sse 2>/dev/null || true
BYTES=$(wc -c </tmp/abort_turnA.sse 2>/dev/null || echo 0)
echo "  Turn A socket closed at ~5s ($BYTES bytes streamed)"
[ "$BYTES" -gt 0 ] || { echo "FAIL: Turn A produced no output — never reached decode"; exit 1; }
sleep 3   # let the daemon's stdin reader process the abort + state reset

echo "=== Turn B: edited prompt (cancel -> edit -> resend) ==="
RESB=$(curl -fsS -m 120 "$EP" -H 'content-type: application/json' \
  -d "$(jq -cn --arg m "$MODEL" --arg p "$PROMPT_EDITED" '{model:$m,messages:[{role:"user",content:$p}],max_tokens:48,temperature:0,stream:false,chat_template_kwargs:{enable_thinking:false}}')")
CONTENT=$(echo "$RESB" | jq -r '.choices[0].message.content // ""')
echo "  Turn B content: $(printf '%s' "$CONTENT" | head -c 160)"

NEW=$(tail -n +$((LINES0+1)) "$LOG" 2>/dev/null)

# (1) the cancel must have propagated to a real decode-abort: the cancel path
#     prints its preserve line (the abort path's only evidence in the serve log
#     -- the fail-closed reset it falls back to is silent by design).
echo "$NEW" | grep -qE '\[qwen-cache cancel\] preserved to checkpoint pos=' \
  && echo "  OK: cancel propagated to a decode-abort (cancel path fired)" \
  || { echo "FAIL: no decode-abort fired (Turn A finished before cancel, or disconnect not detected)"; echo "--- new log ---"; echo "$NEW" | grep -E 'qwen-cache|abort|cancel'; exit 1; }

# (2) the cancel must have PRESERVED the committed prefix (not full-reset):
#     Turn B enters with conv_tok = the preserved prompt length (> 0), the
#     cancel line's checkpoint pos is at or before it, and conv_tok never
#     exceeds the committed length (no uncommitted decode tokens carried).
CANCEL_LINE=$(echo "$NEW" | grep -E '\[qwen-cache cancel\] preserved to checkpoint pos=' | tail -1)
CANCEL_POS=$(echo "$CANCEL_LINE" | grep -oE 'pos=[0-9]+' | grep -oE '[0-9]+$')
COMMITTED=$(echo "$CANCEL_LINE" | grep -oE 'committed=[0-9]+' | grep -oE '[0-9]+$')
LAST_CONV=$(echo "$NEW" | grep -oE '\[qwen-cache GEN-ENTRY\] conv_tok=[0-9]+' | tail -1 | grep -oE '[0-9]+$')
echo "  cancel preserved pos=${CANCEL_POS:-?} committed=${COMMITTED:-?}; Turn B entered with conv_tok=${LAST_CONV:-?}"
if [ -z "$CANCEL_POS" ]; then
  echo "FAIL: no '[qwen-cache cancel] preserved ...' line — decode-abort full-reset (prefix not preserved)"; exit 1
fi
[ "${LAST_CONV:-0}" != "0" ] \
  && echo "  OK: decode-abort preserved the committed prefix (conv_tok=${LAST_CONV})" \
  || { echo "FAIL: conv_tok=${LAST_CONV:-?} after abort — prefix not preserved (regression)"; exit 1; }
[ "${CANCEL_POS}" -le "${LAST_CONV:-0}" ] \
  && echo "  OK: preserved checkpoint pos=${CANCEL_POS} <= conv_tok=${LAST_CONV}" \
  || { echo "FAIL: preserved pos=${CANCEL_POS} > conv_tok=${LAST_CONV:-?}"; exit 1; }
[ "${LAST_CONV:-0}" -le "${COMMITTED:-0}" ] \
  && echo "  OK: conv_tok=${LAST_CONV} <= committed=${COMMITTED} (no uncommitted decode tokens)" \
  || { echo "FAIL: conv_tok=${LAST_CONV:-?} > committed=${COMMITTED:-?} — uncommitted decode tokens carried (poison)"; exit 1; }

# (3) no uncommitted-token poison: no hash mismatch, and any rewound line
#     resumes at or before the cancelled prompt length. A cold reset (no
#     rewound line) is the fail-closed outcome for a non-extension render and
#     is equally poison-free.
echo "$NEW" | grep -q 'prefix-hash mismatch' \
  && { echo "FAIL: prefix-hash mismatch on the Turn B path — resume off an unattested checkpoint"; exit 1; } \
  || echo "  OK: no prefix-hash mismatch on the Turn B path"
REWOUND_POS=$(echo "$NEW" | grep -E '\[qwen-cache resume\] rewound to checkpoint pos=' | tail -1 | grep -oE 'pos=[0-9]+' | grep -oE '[0-9]+$')
if [ -n "$REWOUND_POS" ]; then
  [ "$REWOUND_POS" -le "${COMMITTED:-0}" ] \
    && echo "  OK: rewound to pos=${REWOUND_POS} <= committed=${COMMITTED}" \
    || { echo "FAIL: rewound to pos=${REWOUND_POS} > committed=${COMMITTED:-?} — resumed off uncommitted decode tokens"; exit 1; }
else
  echo "  OK: no rewound line (live-continuation or cold reset — fail-closed)"
fi

# (4) coherence sanity on Turn B (catches whitespace-attractor / empty garbage)
LEN=${#CONTENT}; NONSPACE=$(printf '%s' "$CONTENT" | tr -d '[:space:]' | wc -c)
UNIQ=$(printf '%s' "$CONTENT" | fold -w1 | sort -u | wc -l)
echo "  Turn B coherence: len=$LEN nonspace=$NONSPACE uniq_chars=$UNIQ"
{ [ "$NONSPACE" -ge 5 ] && [ "$UNIQ" -ge 6 ]; } \
  && echo "  OK: Turn B coherent after cancel+edit" \
  || { echo "FAIL: Turn B degenerate/empty after a cancel"; exit 1; }

# (5) Turn C: resend Turn A's EXACT prompt unedited — the original poison
#     shape. Must be coherent and must NOT resume from a checkpoint past
#     Turn A's prompt length.
echo "=== Turn C: resend Turn A's exact prompt unedited ==="
RESC=$(curl -fsS -m 120 "$EP" -H 'content-type: application/json' \
  -d "$(jq -cn --arg m "$MODEL" --arg p "$PROMPT" '{model:$m,messages:[{role:"user",content:$p}],max_tokens:48,temperature:0,stream:false,chat_template_kwargs:{enable_thinking:false}}')")
CCONTENT=$(echo "$RESC" | jq -r '.choices[0].message.content // ""')
echo "  Turn C content: $(printf '%s' "$CCONTENT" | head -c 160)"
LINES1=$(wc -l < "$LOG" 2>/dev/null || echo 0)
NEW2=$(tail -n +$((LINES1+1)) "$LOG" 2>/dev/null)
echo "$NEW2" | grep -q 'prefix-hash mismatch' \
  && { echo "FAIL: prefix-hash mismatch on the Turn C path"; exit 1; }
CREWOUND=$(echo "$NEW2" | grep -E '\[qwen-cache resume\] rewound to checkpoint pos=' | tail -1 | grep -oE 'pos=[0-9]+' | grep -oE '[0-9]+$')
CLEN=${#CCONTENT}; CNONSPACE=$(printf '%s' "$CCONTENT" | tr -d '[:space:]' | wc -c)
CUNIQ=$(printf '%s' "$CCONTENT" | fold -w1 | sort -u | wc -l)
echo "  Turn C coherence: len=$CLEN nonspace=$CNONSPACE uniq_chars=$CUNIQ"
{ [ "$CNONSPACE" -ge 5 ] && [ "$CUNIQ" -ge 6 ]; } \
  && echo "  OK: Turn C coherent (unedited resend did not poison)" \
  || { echo "FAIL: Turn C degenerate/empty after an unedited resend"; exit 1; }
if [ -n "$CREWOUND" ]; then
  [ "$CREWOUND" -le "${COMMITTED:-0}" ] \
    && echo "  OK: Turn C rewound to pos=${CREWOUND} <= committed=${COMMITTED}" \
    || { echo "FAIL: Turn C resumed from pos=${CREWOUND} > committed=${COMMITTED:-?} (poison shape)"; exit 1; }
else
  echo "  OK: Turn C did not resume from a checkpoint (no rewound line)"
fi

echo "PASS: cancel preserved the committed prefix; no uncommitted-token poison"
