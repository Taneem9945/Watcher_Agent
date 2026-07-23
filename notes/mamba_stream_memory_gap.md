# Mamba Stream Memory Gap

## Current State

The repository currently uses Mamba as a **window classifier**.

The flow is:

1. Raw UNSW-NB15 CSV is loaded.
2. Data is preprocessed into numeric features.
3. Rows are grouped into fixed windows.
4. Each window is passed through the Mamba classifier.
5. The model outputs a benign or attack prediction.

So right now, Mamba is acting as the detector backbone for supervised classification.

## Intended Stream-Memory State

The long-term design is more like a **stream memory watcher**.

That future flow would look like:

1. Incoming network event.
2. Mamba updates its memory or state over time.
3. The system preserves context across windows.
4. Suspicious patterns can be detected as they build.
5. A context builder turns that signal into an analyst-ready packet.
6. The LLM explains the situation and recommends action.

In that version, Mamba is not just classifying one window. It is helping remember what has been happening across the stream.

## Exact Gap

What is missing today:

- No live event stream.
- No persistent hidden state passed from one window to the next.
- No `tick()` or incremental update API.
- No alert object built from evolving memory.
- No direct handoff from Mamba memory into the context builder.

What exists today instead:

- Batch windows.
- Independent predictions per window.
- A separate LLM reasoning path over a compact summary.

## Where the Context Builder Fits

The context builder was meant to sit **after** the classifier.

It would combine:

- prediction
- attack probability
- risk level
- window statistics
- previous context

That makes it a bridge from machine signal to human-readable explanation.

## Summary

Current pipeline:

`raw CSV -> preprocessing -> windowing -> Mamba classifier -> prediction`

Target pipeline:

`event stream -> Mamba memory -> context builder -> LLM watcher -> analyst action`

The current repo has the classifier part working. The true stream-memory watcher is still the next architectural step.
