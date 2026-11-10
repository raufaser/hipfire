# Cancel + edit prefix resume on arch 5 (qwen35) — plan and build record

Date: 2026-11-10. Status: IMPLEMENTED on `feature/qwen35-cancel-prefix-resume`
(cut from `origin/beta`; features land on beta per CONTRIBUTING.md).
Related: `scripts/test-qwen35-abort-resume.sh` (guard, updated);
`docs/VALIDATION.md` (acceptance route).

Outcome: a decode-phase cancel now preserves the hash-attested committed
prefix instead of full-resetting, and an edited resend reuses the common
prefix (live-continuation or validated rewind). The guard passes, and fails
with the abort sites reverted. 75k-token sim: cancel keeps 74240/75647,
the resend continues live with cached 74240/75667; greedy output is
byte-identical to a fresh cold process; `serve_harness` battery + chain
pass clean.

The appendix below is the recipe the build followed. Where the build
differs, the "As built" section lists the deviations; the recipe text
itself is left unchanged as the historical record.

---

# Task: make `hipfire serve` resume from the common prefix after cancel + prompt edit (qwen35 / arch 5)

## User's workflow and problem

The user runs a long `hipfire serve` session (single GPU, gfx1100) behind llama-swap, with a
Qwen3.8-27B model loaded as **arch 5 (`qwen3_5`)**. Their chat client supports correcting a
mistake: a few seconds after sending a prompt they realize it is wrong, they **cancel the
in-flight generation, edit the prompt, and resend**.

**Problem:** every cancel + edit forces a **full session re-prefill**. On a 27B model with a
large context this is a multi-second (or longer) stall before the new generation starts.

**Goal:** cancel + edit + resend must reuse the already-processed common prefix, so only the
edited suffix is prefilled — the way llama.cpp and beellama.cpp behave. The user is not looking
for a workflow change; cancel + edit is the intended workflow.

## Repro environment (user's actual setup)

llama-swap launches:

```
hipfire serve --model "<model-dir>/Swift-1.5-Qwen3.8-27b.mq4v2l-pro" \
              --spec off --kv-k fwht4 --kv-v lloyd4 --idle-timeout 0
```

- Model: `<model-dir>/Swift-1.5-Qwen3.8-27b.mq4v2l-pro`, **arch 5 (`qwen3_5`)**, single GPU,
  gfx1100, AR path (`--spec off`), KV `fwht4`/`lloyd4`.
- Arch confirmed three ways: the file header is `HFQM` with arch byte `0x05` and
  `"architecture":"qwen3_5"`; the daemon load summary prints `(qwen3_5)`; the registry entry
  `qwen3.8:27b-mq4-pro` is `arch_id: 5`.
- Config: `~/.hipfire/config.toml` has no session-cache override.
- llama-swap `config.yaml`, entry `Swift-1.5-hipfire` (llama-swap manages the daemon).
- The daemon is currently running on :5804 (llama-swap manages it). `curl
  http://127.0.0.1:5804/v1/models` returns the `load_summary` line with `(qwen3_5)`.

**Do not confuse this with arch 16 / Qwen4.** Only `qwen3.8:flash-next*` is arch 16. In
particular `HIPFIRE_QWEN4_TURN_SNAPSHOTS` is a **Qwen4-only** switch (read only by
`Qwen4Bundle`, `crates/hipfire-arch-qwen4/src/bundle.rs:904`) and does nothing on arch 5. Do
not propose it.

## Reference behavior: what llama.cpp / beellama.cpp do

Local checkouts: `~/Development/llama.cpp` and `~/Development/beellama.cpp`.

**llama.cpp (attention-only):**

- Client disconnect → `server_response_reader::stop()` posts `SERVER_TASK_TYPE_CANCEL`
  (`tools/server/server-queue.cpp:603-622`).
- The cancel handler calls `slot.release()` (`tools/server/server-context.cpp:2901-2910`).
- `release()` (`tools/server/server-context.cpp:586-608`) sets the slot IDLE but **keeps
  `prompt.tokens` and the KV cache** for a normal slot (only child/stateless slots are cleared).
- The next request picks the slot by longest-common-prefix:
  `server_prompt_plan_reuse` (`tools/server/server-task.h:841`) computes
  `lexical_tokens = prompt.tokens.get_common_prefix(requested)`; prefill sets
  `n_past = get_common_prefix(input_tokens)` (`server-context.cpp:4698`) and rolls the KV back
  to that boundary with `llama_memory_seq_rm_plan` / `seq_rm`
  (`server-context.cpp:4870-4903`).
- Reuse is exact up to the first differing token, even mid-message. No checkpoint is needed
  because attention KV is append-only and suffix-removable.

**beellama.cpp (recurrent / KVarN fork — the right analog for hipfire's DeltaNet):**

- Same `release()`-keeps-state behavior (`server-context.cpp:811-838`).
- Recurrent state cannot be suffix-trimmed, so it captures **durable checkpoints during
  prefill**: `create_checkpoint` (`server-context.cpp:3152`) is called **before** `llama_decode`
  (`server-context.cpp:5340`), so each checkpoint covers exactly the prefix processed so far.
  Boundaries are stable: chunk multiples, user-message starts/ends, KVarN group seams, periodic
  anchors, near the prompt end (`server-context.cpp:5280-5341`).
- On the next request, `server_prompt_plan_reuse` (`server-task.h:862-871`) picks the longest
  checkpoint with `n_tokens <= LCP` and an aligned boundary; `prompt_load` restores it
  transactionally (`restore_checkpoint_transaction`, `server-context.cpp:4924`) and reprocesses
  only the suffix. If no checkpoint covers the LCP, it forces a full re-process
  (`server-context.cpp:4938-4948`).

**Common shape:** cancel does not discard state; the next request reuses the longest common
prefix against retained checkpoints. The partial generation is trimmed, not replayed.

## hipfire's current behavior (code-path diagnosis)

The mechanism that exists on arch 5 is the **DeltaNet checkpoint ring**, not the Qwen4 session
cache:

- `HIPFIRE_CACHE_CKPT_RESUME` — default **on** (`crates/hipfire-generate/src/ar.rs:791`).
- `HIPFIRE_CACHE_CKPT_INTERVAL` — default 2048 tokens (`ar.rs:797`).
- `HIPFIRE_CACHE_CKPT_MAX` — default 8 snapshots (`ar.rs:804`).
- Checkpoints are taken during prefill (`ar.rs:4952-4961`) **and during decode**
  (`ar.rs:5485-5495`) via `speculative::take_dn_checkpoint`
  (`crates/hipfire-arch-qwen35/src/speculative.rs:13058`).
- On a divergent render, the resume logic (`ar.rs:4368-4414`) restores the latest checkpoint at
  or before the LCP, rewinds `seq_pos` and the KV write head, truncates `conversation_tokens`, and
  replays only the suffix. Gated by `ckpt_resume_enabled() && evict_safe && arch_key == "qwen35"`.

**Why cancel + edit cold-prefills:** the cancel path throws the ring and the prior away.

- `check_abort` (`ar.rs:2153`) → `production_fail_closed_rollback` (`common.rs:191`) →
  `m.seq_pos = 0; m.conversation_tokens.clear()` (`common.rs:202-203`).
- `fail_closed_reset_target_and_spec` (`common.rs:1399`) →
  `free_checkpoints(&mut m.prefill_checkpoints, gpu)` (`common.rs:1494`).
- The next turn's `lcp` is computed against `m.conversation_tokens` (`ar.rs:4194-4199`); with the
  prior empty, `lcp = 0` and there is no ring to restore → full cold prefill.

**The safety invariant** (comment at `ar.rs:4335-4346`):

> this restore-checkpoint-at-rpos + replay rendered[rpos..] is exact iff the checkpoint at rpos
> reflects the committed prefix rendered[..rpos]. That holds because (a) rpos <= lcp =>
> rendered[..rpos] == conversation_tokens[..rpos] ..., and (b) ALL abort paths now full-reset, so
> a retained checkpoint can never carry UNCOMMITTED tokens — the poison that used to drift the
> non-reversible DeltaNet state into garbage. If you ever remove an abort-reset (or let
> conversation_tokens diverge from the forwarded stream), this resume becomes unsound: re-validate
> with a per-checkpoint prefix hash (llama.cpp's tokens_hash contract) or cold-recompute.

The original bug is reproduced and guarded by `scripts/test-qwen35-abort-resume.sh`: after a
mid-decode abort, the next turn must enter with `conv_tok=0` and must not
`[qwen-cache resume] rewound`. The observed failure mode was whitespace garbage / premature EOS
from resuming off uncommitted decode tokens.

## Fix direction

Match llama.cpp/beellama.cpp: **on cancel, keep the committed-prefix state; on the next request,
reuse the longest common prefix.** Minimum viable shape for arch 5:

1. **A distinct cancel path** that preserves the committed boundary instead of the full reset.
   Keep it separate from the explicit daemon reset / model swap path, which must keep today's full
   reset (the model-swap cache-invalidation smoke in AGENTS.md §2B requires it).
2. **Truncate `conversation_tokens` to the cancelled request's prompt length** instead of clearing
   it. This is what stops the LCP from reaching into the uncommitted decode region — the exact
   poison from the original bug. The prompt length is `rendered`/`prompt_tokens` in `generate_ar`;
   threading it to the cancel site in `common.rs` is the first implementation question.
3. **Keep the prefill checkpoints** (positions ≤ prompt length) instead of freeing the ring; drop
   the decode checkpoints. The resume logic's
   `rposition(|(p, _)| *p <= lcp && *p < rendered.len())` then selects only committed-prefix
   checkpoints.
4. **Add the per-checkpoint prefix hash** the comment asks for (llama.cpp's `tokens_hash`
   contract) and validate it before restore. Without this, the change is exactly the unsound case
   the comment names; with it, the resume is robust even if ring recycling or truncation is
   imperfect.
5. **Update `scripts/test-qwen35-abort-resume.sh`**: its `conv_tok=0` assertion becomes
   `conv_tok == prompt_length` (or `< prior_len`), and the "no resume after cancel" assertion
   becomes "resume only from a hash-validated checkpoint". The test's purpose (no
   uncommitted-token poison) is preserved; its assertion changes.

**Coverage caveat:** the ring is a sliding window of ~`INTERVAL × MAX` ≈ 16K tokens. The resume
helps when the edit is near the tail (the user's "few seconds after sending" case), not when
editing an early message in a very long context. `HIPFIRE_CACHE_CKPT_MAX` / `_INTERVAL` are the
tunables, at VRAM cost.

**Exactness caveat:** DeltaNet state after a prefill chunk vs. a decode step is not guaranteed
byte-identical (the codebase labels such state session-exact, not cold-exact). Acceptance should
compare greedy output against a fresh process and confirm the argmax is robust, or compare against
a reference split at the same index.

## Constraints / non-negotiables

- The fail-closed reset exists for correctness. Do not weaken it; add a provably-valid fast path
  and fall back otherwise.
- Do not break `scripts/test-qwen35-abort-resume.sh`'s purpose; if its assertion must change,
  update it deliberately and say so.
- Model-swap cache invalidation (AGENTS.md §2B) must keep working: stale pointer-keyed caches
  must not survive a swap.
- This is a user-facing state-lifecycle change → validation route is
  `python3 scripts/serve_harness.py battery|chain --model ...` per `docs/VALIDATION.md` (retired
  coherence-gate scripts are not acceptance). Eyeball decoded output.
- Any perf claim needs fresh processes, byte-identical prompts (`benchmarks/prompts/`), prompt
  md5, model md5 and binary md5 per AGENTS.md §5.
- Run `scripts/pre-commit-ci.sh origin/<base>` before committing.

## Acceptance

1. `HIPFIRE_QWEN_CACHE_TRACE=1` shows a meaningful `lcp`/`reused` after cancel+edit, with a
   `[qwen-cache resume] rewound to checkpoint pos=...` line instead of a cold prefill.
2. Greedy output after cancel+edit equals the output of the same edited prompt run in a fresh
   process (state correctness, not just speed).
3. Normal (non-cancel) turns still take the live-continuation zero-prefill path.
4. An unedited resend after cancel does not resume off uncommitted decode tokens (the original
   poison); the prefix-hash validation rejects it.
5. `serve_harness` battery + chain pass, and the prefill-time win is reported with the fixture
   metadata above.
6. `scripts/test-qwen35-abort-resume.sh` passes in its updated form.

## Files

- `crates/hipfire-generate/src/ar.rs` — cancel check 2153; resume logic 4368-4414; safety comment
  4335-4346; checkpoint capture 4952, 5485; `ckpt_resume_enabled` 791.
- `crates/hipfire-generate/src/common.rs` — `production_fail_closed_rollback` 191;
  `fail_closed_reset_target_and_spec` 1399; `free_checkpoints` call 1494.
- `crates/hipfire-arch-qwen35/src/speculative.rs` — `take_dn_checkpoint` 13058; `DeltaNetSnapshot`
  (add the prefix hash here).
- `scripts/test-qwen35-abort-resume.sh` — the guard to update.
- `docs/env-vars.md` (~1047-1049) and `docs/CONFIG.md` (~940) — `HIPFIRE_CACHE_CKPT_*` docs.
- `docs/VALIDATION.md` — the validation route.

## As built — deviations from the recipe

- The prefix hash is SHA-256, first 8 bytes little-endian
  (`dn_prefix_hash_two` / `dn_prefix_hash`), not the FNV-1a the Step 1
  sketch shows. It matches the existing
  `hipfire_runtime::checkpoint_pool::prefix_fingerprint`, whose doc records
  why the repo abandoned FNV-1a for prefix fingerprints; `sha2` was already
  a dependency of `hipfire-arch-qwen35`.
- The cancel path validates the hash too (the Step 3 sketch only checks
  `pos <= committed_len`): it walks back to the newest index with `pos <=
  committed_len`, a non-zero hash, and `hash ==
  dn_prefix_hash(conversation_tokens[..pos])`, and full-resets if none
  qualifies.
- Six `take_dn_checkpoint` call sites, not five: `vmm_conv.rs:148` also
  passes `None`. `m.asst_turn_cache.clear()` kept per Step 3 (only
  completed turns ever insert, so clearing is conservative).
- The decode-checkpoint assert is `debug_assert!(eviction.is_some() ||
  seq_pos == conversation_tokens.len())`, because under eviction `seq_pos`
  is physical. `#[allow(clippy::too_many_arguments)]` on
  `take_dn_checkpoint`, matching repo practice.
- Guard Turn B (short prompt, append edit) takes the fail-closed cold
  reset: the chat-template trailer diverges from the preserved render and
  no checkpoint sits at or before lcp. So the script accepts
  live-continuation, a hash-validated rewind at or before the committed
  length, or a cold reset — and rejects any hash mismatch or any rewind
  past the committed length. Turn C (exact resend) checks coherence plus
  no rewind past it.

## Measured acceptance results (gfx1100, HIP 7.16)

Model `Swift-1.5-Qwen3.8-27b.mq4v2l-pro` (arch 5, md5 `1c988e37…`,
16,458,017,792 B), `--spec off --kv-k fwht4 --kv-v lloyd4`,
`HIPFIRE_QWEN_CACHE_TRACE=1`, explicit `--max-seq` (see headroom note).
Daemon md5 `73742f04…`, CLI `2cffc0fe…` — validation builds; they carried
a local-only Windows link shim for beta's `clock_gettime` break that is
not in the diff (upstream fix belongs in its own PR).

- Guard `test-qwen35-abort-resume.sh`: PASS. With the two decode-abort
  sites reverted to `production_fail_closed_rollback` it FAILS
  (`conv_tok=0`, no preserve line) — the guard still guards.
- 75k sim (`cancel_100k_*`, md5 `aeaacc1f…` / `072226d3…`): cancel keeps
  `pos=74240 (committed=75647)`; the edited resend logs `lcp=74240`,
  continues live, `cached_tokens=74240/75667`, coherent output.
- Greedy equality at 75k: cancel+edit output byte-identical to a fresh
  cold process.
- `serve_harness battery` + `chain` (greedy): 0 runaway, 0 empty,
  0 attractor, 0 retrieval miss.
- Headroom caveat, found by measurement: the context-full guard counts
  the resident prefix plus the full edited prompt, so reuse needs
  `max_seq` headroom for both (~2x prompt at 75k) and cold-resets
  without it. Documented in `docs/CONFIG.md`. Beta auto-fit caps this
  model at 4k context (the fork fits 261k) — a pre-existing gap, which
  is why the long runs set `--max-seq` explicitly.

---

# Appendix: implementation steps (for a less-capable agent)

Read this whole appendix before touching code. Each step ends with a check.
Do the steps in order. If any check fails, stop — do not work around it.

## Step 0 — reproduce the cold-prefill baseline (no code changes)

1. Restart the daemon with the trace env var (the running :5804 daemon was
   started without it, so current logs prove nothing):
   `HIPFIRE_QWEN_CACHE_TRACE=1 hipfire serve --model <arch5 model> --spec off ...`
2. Send a long prompt, cancel mid-decode (close the socket / Ctrl-C the client),
   then resend the edited prompt.
3. Expected (today's behavior): the next turn logs `[qwen-cache GEN-ENTRY]
   conv_tok=0`, `lcp=0`, no `[qwen-cache resume] rewound` line, and a full
   prefill. Save this log — it is the before/after comparison.
4. Check: `scripts/test-qwen35-abort-resume.sh` passes on the unmodified tree.

## Step 1 — add the prefix hash to `DeltaNetSnapshot`

File: `crates/hipfire-arch-qwen35/src/speculative.rs` (struct at ~1400,
impl at ~1422, constructors `new_for` ~1589, `save_from` ~1869,
`restore_to` ~1953, `free_gpu` ~1988).

1. Add a field `pub prefix_hash: u64` to `DeltaNetSnapshot`, initialized to `0`
   in `new_for` (and any other constructor that builds the struct literally —
   compile errors will list them; set each to `0`).
2. Add next to `take_dn_checkpoint` (~13058):
   ```rust
   pub fn dn_prefix_hash_two(a: &[u32], b: &[u32]) -> u64 {
       let mut h: u64 = 0xcbf2_9ce4_8422_2325; // FNV-1a 64
       for t in a.iter().chain(b.iter()) {
           for byte in t.to_le_bytes() {
               h ^= byte as u64;
               h = h.wrapping_mul(0x0100_0000_01b3);
           }
       }
       h
   }
   ```
   (`dn_prefix_hash(tokens) = dn_prefix_hash_two(tokens, &[])` if a
   one-slice form is wanted.)
3. Change `take_dn_checkpoint` signature: add a trailing param
   `prefix: Option<(&[u32], &[u32])>` (two slices so callers never allocate;
   the hash covers `a` chained with `b`). Inside, after the early returns,
   set `snap.prefix_hash = prefix.map(|(a, b)| dn_prefix_hash_two(a, b)).unwrap_or(0)`
   just before `cks.push((pos, snap))`. `None` means "legacy caller" and stores
   `0` — behavior for those paths is unchanged (see Step 2).
4. Check: `cargo check -p hipfire-arch-qwen35` fails only at the
   `take_dn_checkpoint` call sites ( Steps 2 fixes them).

## Step 2 — update the 5 `take_dn_checkpoint` call sites

- `crates/hipfire-generate/src/ar.rs:4953` (prefill loop). Pass
  `Some((&m.conversation_tokens, &new_tokens[..done]))` where
  `done = m.seq_pos.saturating_sub(base)` and `base` is
  `m.conversation_tokens.len()` captured before the `while start < ...` loop
  (conversation is only extended after the loop at ~5011, so `base` is fixed
  and `done` is exactly the tokens forwarded so far this turn). This covers
  cold prefill (`base == 0`), resumed prefill, and live extension uniformly.
  Borrow: `m.conversation_tokens`, `new_tokens`, `m.seq_pos`,
  `m.prefill_checkpoints` (via `&mut`), and the `dn`/`kv` bundle borrows are
  disjoint fields — the existing call already composes this way; keep the
  same statement order.
- `ar.rs:5486` (decode loop). Pass `Some((&m.conversation_tokens, &[]))`
  (here conversation IS the forwarded stream; `m.seq_pos` should equal its
  length — add `debug_assert_eq!(m.seq_pos, m.conversation_tokens.len())`).
- `crates/hipfire-arch-qwen35/src/mtp_speculator.rs:474`,
  `speculative.rs:13519`, `speculative.rs:13622` (DFlash/spec paths): pass
  `None`. Zero behavior change on those paths; wiring real hashes there is
  explicitly out of scope.
- Check: `cargo check` passes workspace-wide.

## Step 3 — add the cancel-preserve function (new code, one place)

Put it in `ar.rs` near `ckpt_resume_enabled` (~791), NOT in `common.rs`
(`common.rs` cannot call back into `ar.rs` — circular dependency; the env-var
helpers live in `ar.rs`).

```rust
/// Cancel-with-prefix-preserve for qwen35 (arch 5/6) AR decode aborts.
/// Keeps only COMMITTED prefix state; every other case falls back to
/// `production_fail_closed_rollback` (full reset). `committed_len` is the
/// prompt length of the cancelled request (NOT including decoded tokens).
pub fn cancel_preserve_qwen35_prefix(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    committed_len: usize,
) -> RollbackEpilogue {
    // Gate: same conditions the resume path (~4347-4379) requires, PLUS
    // no-speculator and no-adaptive-KV (both are state the checkpoint
    // restore does not cover). Anything else -> full reset (fail closed).
    let arch_ok = m.state.as_ref().map_or(false, |s| s.as_ref().arch_key() == "qwen35");
    let evict_safe = m.pp <= 1 && m.eviction.is_none()
        && m.qwen35().map_or(false, |b| b.kv_cache.compact_offset == 0);
    let eligible = ckpt_resume_enabled() && arch_ok && evict_safe
        && m.speculator.is_none() && m.kv_adaptive.is_none();
    // Latest checkpoint at or before the COMMITTED length (never decode region).
    let idx = if eligible {
        m.prefill_checkpoints.iter().rposition(|(p, _)| *p <= committed_len)
    } else { None };
    let Some(idx) = idx else {
        return crate::common::production_fail_closed_rollback(m, gpu, None, None);
    };
    let rpos = m.prefill_checkpoints[idx].0;
    // Drop every checkpoint past the committed length (decode poison):
    // drain from the back with free_gpu per snapshot (never bare truncate).
    while m.prefill_checkpoints.len() > idx + 1 {
        if let Some((_, snap)) = m.prefill_checkpoints.pop() { snap.free_gpu(gpu); }
    }
    // Restore DN to the kept checkpoint; any failure -> full reset.
    let ok = /* downcast m.state to Qwen35Bundle, ck.restore_to(&mut b.dn_state, gpu).is_ok() */;
    if !ok {
        return crate::common::production_fail_closed_rollback(m, gpu, None, None);
    }
    m.seq_pos = rpos;
    m.conversation_tokens.truncate(rpos);
    m.asst_turn_cache.clear(); // prompt-render cache only; rebuilds from history
    crate::common::free_checkpoints(&mut m.dflash_checkpoints, gpu);
    crate::common::fail_closed_invalidate_graphs_and_replay(gpu);
    // Attested sync, same pattern as production_fail_closed_rollback_live:
    // fail_closed_epilogue_after_sync + fail_closed_device_sync, then
    // gpu.replay.begin_replay_observation_window() on rolled_back.
    ...
    eprintln!("[qwen-cache cancel] preserved to checkpoint pos={rpos} (committed={committed_len})");
    epilogue
}
```

Notes for the implementer:
- The downcast/restore lines are copied verbatim from the resume block
  (`ar.rs:4385-4395`); the epilogue/sync lines from
  `common.rs:236-245`. Do not invent new sync logic.
- `m.asst_turn_cache.clear()` keeps today's behavior (the aborted turn never
  committed a turn to it). Do NOT clear `m.conversation_tokens` beyond the
  truncate, do NOT zero DN (it now holds exactly `conversation_tokens`).
- Do NOT touch `fail_closed_reset_target_and_spec` or the model-swap/reset
  paths — they keep the full reset (AGENTS.md §2B smoke depends on it).
- Prefill-phase aborts stay full reset (partial prefill advanced DN
  irreversibly; there is no committed boundary to preserve to).

## Step 4 — switch ONLY the decode-abort sites to the new function

- `ar.rs:5286` (qwen35 dense AR decode abort — the user's path). Replace
  `production_fail_closed_rollback(m, gpu, None, None)` with
  `cancel_preserve_qwen35_prefix(m, gpu, ngram_scope_start)`.
  `ngram_scope_start` (~5039, still in scope) IS the prompt length:
  it was captured as `m.conversation_tokens.len()` right after the post-prefill
  extend (~5011) and before any decode token. Keep the surrounding
  `let _ = (kv, dn, weights, config, scratch);` drop line and the
  `emit_spec_cancel_after_rollback` line unchanged.
- `ar.rs:2153` (`generate_ar_with_forward` decode abort). Replace with
  `cancel_preserve_qwen35_prefix(m, gpu, prompt_tokens.len())`
  (`prompt_tokens` is the function parameter; the gate inside declines to full
  reset for every non-qwen35 route, so this is a safe no-op elsewhere).
- Do NOT change: `ar.rs:4740`, `ar.rs:4872` (+ the `prefill_aborted` rollback
  at ~4970) — prefill aborts stay full reset. `ar.rs:6341` — the llama-route
  decode loop, stays full reset (different arch, no qwen35 ring).
- Check: `cargo check` + `cargo clippy` clean on touched crates.

## Step 5 — validate the hash at restore time

In the resume block (`ar.rs:~4380-4414`), after `let rpos = ...` and before
`restore_to`, insert:

```rust
let stored = m.prefill_checkpoints[idx].1.prefix_hash;
if stored != 0 && stored != speculative::dn_prefix_hash_two(&rendered[..rpos], &[]) {
    eprintln!("[qwen-cache resume] REJECTED checkpoint pos={rpos}: prefix-hash mismatch — cold reset");
    // fall through to the existing cold-reset path (the `None` arm):
    // easiest is to wrap: `let resumed = if hash_ok { ...existing... } else { None }`
}
```

`stored == 0` (legacy `None` callers from Step 2) skips validation and keeps
today's behavior. `rpos <= lcp` already holds at this point, so
`rendered[..rpos] == conversation_tokens[..rpos]` — hashing `rendered` checks
exactly what is about to be committed to.

## Step 6 — update the safety comment and docs

- Rewrite the `SAFETY INVARIANT` comment (`ar.rs:4335-4346`) to describe the
  NEW invariant: (a) checkpoints carry a prefix hash over the committed tokens
  they cover; (b) decode-phase cancel preserves only checkpoints at or before
  the committed prompt length and rewinds DN/conversation to the latest one;
  (c) prefill aborts and model swap still full-reset; (d) restore validates
  the hash and cold-resets on mismatch. Keep the pointer to the guard script.
- `docs/env-vars.md` (~1047-1049) + `docs/CONFIG.md` (~940): document that
  `HIPFIRE_CACHE_CKPT_*` checkpoints now also serve cancel+edit prefix reuse
  on arch 5 and that each snapshot carries a prefix hash. One short paragraph
  each; do not rewrite the pages.

## Step 7 — update `scripts/test-qwen35-abort-resume.sh` (deliberately)

The script's PURPOSE (no uncommitted-token poison) is unchanged; its
assertions must change because preserved state is now the correct outcome:

1. Turn A prompt is short (< interval), so the first prefill chunk leaves a
   checkpoint at the prompt length: after cancel, Turn B must enter with
   `conv_tok > 0` (was: `== 0`). Assert `conv_tok != 0` AND Turn B's log shows
   `[qwen-cache cancel] preserved to checkpoint pos=...` with
   `pos <= conv_tok`.
2. Replace assertion (3) ("never `[qwen-cache resume] rewound`") with: any
   `rewound` line after the cancel must carry a hash-validated checkpoint —
   assert the log does NOT contain `prefix-hash mismatch` on the Turn B path,
   and that Turn B took either live-continuation or a `rewound` with
   `rpos <=` Turn A's committed length.
3. Keep assertion (4) (Turn B coherence) verbatim, and add Turn C: resend Turn
   A's EXACT prompt unedited → must be coherent and must not resume from a
   checkpoint past Turn A's prompt length (the original poison shape).
4. Check: script passes on the new tree AND fails if you temporarily revert
   Step 4 (guard still guards).

## Step 8 — acceptance (same as the handover, in this order)

1. `HIPFIRE_QWEN_CACHE_TRACE=1`: cancel+edit shows `lcp > 0`, a
   `[qwen-cache cancel] preserved...` line, then `[qwen-cache resume] rewound
   to checkpoint pos=...` and a short suffix prefill instead of a cold one.
2. Greedy equality: run the edited prompt in a FRESH process; token IDs must
   match the cancel+edit run exactly (state correctness, not just speed).
   DeltaNet prefill-chunk vs decode-step state is session-exact, not
   cold-exact — if IDs differ by argmax-level noise, compare against a
   reference split at the same index before concluding.
3. Normal turns unchanged (live-continuation zero-prefill still fires).
4. `python3 scripts/serve_harness.py battery --model ...` then `chain`
   (per `docs/VALIDATION.md`; retired coherence-gate scripts are NOT
   acceptance). Eyeball the decoded output in both — a tight stddev or oddly
   high τ is a warning sign, not a win.
5. Report with fixture metadata per AGENTS.md §5: fresh processes,
   byte-identical prompts (`benchmarks/prompts/` + md5), model md5, binary
   md5s, GPU/ROCm versions.
6. `scripts/pre-commit-ci.sh origin/<base>` before committing.

