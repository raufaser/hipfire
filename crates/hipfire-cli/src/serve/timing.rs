// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Opt-in llama.cpp-style per-request timing lines for the serve console
//! (`HIPFIRE_TIMING=1`). Prefill progress per completed chunk, the decode
//! rate every 3 s, and a final prompt/eval/total summary per request.
//!
//! Lines are returned to the caller (which writes them to stderr) so tests
//! need no terminal. All rates use the gateway measurement basis, the same
//! one `metrics::observe_done` reports: the interval includes admission
//! queueing, not just engine time.

use std::time::{Duration, Instant};

/// Decode lines are redrawn at most this often.
const DECODE_INTERVAL: Duration = Duration::from_secs(3);

pub(crate) struct TimingPrinter {
    enabled: bool,
    id: String,
    started: Instant,
    interval: Duration,
    prompt_total: Option<u64>,
    tokens: u64,
    first_token: Option<Instant>,
    /// (instant, token count) of the last decode line, for the trailing
    /// window rate.
    last_decode: Option<(Instant, u64)>,
    finished: bool,
}

impl TimingPrinter {
    pub(crate) fn new(enabled: bool, id: &str) -> Self {
        Self::with_interval(enabled, id, DECODE_INTERVAL)
    }

    /// Test seam: a shorter decode redraw interval.
    pub(crate) fn with_interval(enabled: bool, id: &str, interval: Duration) -> Self {
        Self {
            enabled,
            id: id.to_owned(),
            started: Instant::now(),
            interval,
            prompt_total: None,
            tokens: 0,
            first_token: None,
            last_decode: None,
            finished: false,
        }
    }

    /// One prefill chunk completed (`done` of `total` prompt tokens).
    pub(crate) fn prefill(&mut self, done: u64, total: u64) -> Option<String> {
        if !self.enabled {
            return None;
        }
        self.prompt_total = Some(total);
        let t = self.started.elapsed().as_secs_f64();
        if t <= 0.0 {
            return None;
        }
        Some(format!(
            "{} prompt processing, n_tokens = {done}, progress = {:.2}, t = {t:.2} s / {:.0} tokens per second",
            self.id,
            done as f64 / total.max(1) as f64,
            done as f64 / t
        ))
    }

    /// One generated token (a `token` or `reasoning` event).
    pub(crate) fn token(&mut self) -> Option<String> {
        if !self.enabled {
            return None;
        }
        self.tokens += 1;
        let now = Instant::now();
        let first = *self.first_token.get_or_insert(now);
        // The first line waits a full interval from the first token, so `tg`
        // never divides a sub-clock-resolution span (and the first line
        // appears ~3 s into generation, like llama.cpp's).
        if now.duration_since(first) < self.interval {
            return None;
        }
        let last = self.last_decode;
        if last.is_some_and(|(at, _)| now.duration_since(at) < self.interval) {
            return None;
        }
        let tg = self.tokens as f64 / now.duration_since(first).as_secs_f64().max(f64::EPSILON);
        // shortcut: tg_3s is the rate since the previous line (a ~3 s
        // segment), not a sliding 3 s window — the 3 s redraw makes the two
        // coincide in practice; a shorter trailing segment uses its real age.
        let tg3 = match last {
            Some((at, count)) => {
                (self.tokens.saturating_sub(count)) as f64
                    / now.duration_since(at).as_secs_f64().max(f64::EPSILON)
            }
            None => tg,
        };
        self.last_decode = Some((now, self.tokens));
        Some(format!(
            "{} n_gen = {}, tg = {:.2} t/s, tg_3s = {:.2} t/s",
            self.id, self.tokens, tg, tg3
        ))
    }

    /// Final lines: the prompt/eval/total summary from the `done` payload and
    /// the gateway attempt clock. `first_token_at` is the attempt-elapsed
    /// duration to the first generated token (`None` = no token produced);
    /// `total` the whole attempt.
    pub(crate) fn finish(
        &mut self,
        done: &serde_json::Value,
        first_token_at: Option<Duration>,
        total: Duration,
    ) -> Vec<String> {
        if !self.enabled || self.finished {
            return Vec::new();
        }
        self.finished = true;
        let mut out = Vec::new();
        let id = self.id.as_str();
        let u64f = |key: &str| done.get(key).and_then(serde_json::Value::as_u64);
        // Prefer the request's own counts; fall back to what we observed.
        let tokens = u64f("tokens").unwrap_or(self.tokens);
        let prompt_tokens = u64f("prompt_tokens").or(self.prompt_total).unwrap_or(0);
        let total_ms = total.as_secs_f64() * 1000.0;
        if let Some(ttft) = first_token_at {
            let ttft_ms = ttft.as_secs_f64() * 1000.0;
            if prompt_tokens > 0 {
                out.push(format!(
                    "{id} prompt eval time = {ttft_ms:.2} ms / {prompt_tokens} tokens ({:.2} ms per token, {:.2} tokens per second)",
                    ttft_ms / prompt_tokens as f64,
                    prompt_tokens as f64 / ttft.as_secs_f64().max(f64::EPSILON)
                ));
            }
            if tokens > 0 {
                let decode_ms = total.as_secs_f64() - ttft.as_secs_f64();
                if decode_ms > 0.0 {
                    out.push(format!(
                        "{id}        eval time = {decode_ms:.2} ms / {tokens} tokens ({:.2} ms per token, {:.2} tokens per second)",
                        decode_ms * 1000.0 / tokens as f64,
                        tokens as f64 / decode_ms
                    ));
                }
            }
        }
        out.push(format!(
            "{id}       total time = {total_ms:.2} ms / {} tokens",
            prompt_tokens + tokens
        ));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn done(tokens: u64, prompt_tokens: u64) -> serde_json::Value {
        serde_json::json!({ "type": "done", "tokens": tokens, "prompt_tokens": prompt_tokens })
    }

    #[test]
    fn prefill_line_reports_tokens_progress_and_rate() {
        let mut p = TimingPrinter::new(true, "r1");
        // Past the clock's resolution so `elapsed() > 0`.
        std::thread::sleep(Duration::from_millis(20));
        let line = p.prefill(10240, 62886).expect("first chunk draws");
        assert!(
            line.starts_with("r1 prompt processing, n_tokens = 10240, progress = "),
            "{line}"
        );
        assert!(line.contains("tokens per second"), "{line}");
    }

    #[test]
    fn decode_lines_are_throttled_to_the_interval() {
        let mut p = TimingPrinter::with_interval(true, "r1", Duration::from_millis(5));
        // Inside the first interval from the first token: no line.
        for _ in 0..10 {
            assert!(p.token().is_none(), "inside the interval");
        }
        std::thread::sleep(Duration::from_millis(10));
        let first = p.token().expect("due after the interval");
        assert!(first.starts_with("r1 n_gen = 11, tg = "), "{first}");
        assert!(first.contains("tg_3s = "), "{first}");
        // A burst right after is throttled.
        for _ in 0..50 {
            assert!(p.token().is_none(), "inside the 3 s interval");
        }
    }

    #[test]
    fn summary_carries_prompt_eval_and_total() {
        let mut p = TimingPrinter::new(true, "r1");
        p.prefill(62886, 62886);
        for _ in 0..1905 {
            p.token();
        }
        let lines = p.finish(
            &done(1905, 62886),
            Some(Duration::from_millis(28199)),
            Duration::from_millis(73317),
        );
        let text = lines.join("\n");
        assert!(
            text.contains("prompt eval time = 28199.00 ms / 62886 tokens"),
            "{text}"
        );
        assert!(text.contains("eval time = "), "{text}");
        assert!(
            text.contains("total time = 73317.00 ms / 64791 tokens"),
            "{text}"
        );
    }

    #[test]
    fn finish_without_tokens_skips_the_eval_line_and_finishes_once() {
        let mut p = TimingPrinter::new(true, "r1");
        p.prefill(10, 10);
        let lines = p.finish(&done(0, 10), None, Duration::from_millis(5));
        let text = lines.join("\n");
        assert!(!text.contains("eval time"), "{text}");
        assert!(text.contains("total time = 5.00 ms / 10 tokens"), "{text}");
        assert!(
            p.finish(&done(0, 10), None, Duration::from_millis(5))
                .is_empty(),
            "second finish is a no-op"
        );
    }

    #[test]
    fn disabled_printer_never_writes() {
        let mut p = TimingPrinter::new(false, "r1");
        assert!(p.prefill(1, 2).is_none());
        assert!(p.token().is_none());
        assert!(p
            .finish(&done(1, 1), None, Duration::from_millis(1))
            .is_empty());
    }
}
