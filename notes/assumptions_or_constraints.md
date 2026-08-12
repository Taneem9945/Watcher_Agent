# Assumptions And Constraints

These notes capture the current working assumptions for the mixed, timestamped Watcher Agent prototype.

## Points Covered

- The active prototype should use P025 mixed-source data only.
- The prototype assumes different sources have already been normalized into a shared event shape.
- Duplicate evidence is not considered a major issue for the prototype.
- Duplicate rows should be handled during normalization where possible.
- Timestamp alignment is assumed to exist for the current test dataset.
- Different event granularities are acceptable for the first prototype.
- Data volume is not considered a blocker for initial development.
- Mamba labeling is not part of the current direction.
- Formal evaluation is out of scope for the immediate prototype.
- Trust and explainability are not primary constraints right now.
- Current context selection is assumed to be sufficient for the prototype.
- Fixed 60-second windows are acceptable for now, but must be revisited later.
- Stateful operation is currently handled through app-level stream memory.
- App-level stream memory is not the same as true recurrent Mamba hidden-state carryover.
- The current Mamba encoder proves the data path, not learned security intelligence.
- Mamba must eventually receive a meaningful learning objective or compatible checkpoint before its embeddings can be treated as reliable security evidence.
- Ollama assessments should be generated fresh for each replay run, not reused from cache.

## Active Data Source

- The active web prototype should use P025 mixed-source packets only.
- The current selected P025 sources are `zeek_conn`, `zeek_http`, and `host_auth`.
- Older UNSW-related code may still exist in the repository for earlier experiments, but it should not be used by the active web replay path.
- The web replay should reject non-P025 packet shapes instead of silently running older data.

## Shared Event Shape

- The key assumption is that mixed data sources can be converted into one shared event shape before reaching the Watcher Agent.
- The shared shape does not mean every event has every field populated.
- It means every event follows the same broad schema, with missing fields made explicit.
- This lets different records, such as Zeek connections, HTTP activity, host auth logs, and future alerts, move through the same pipeline.

## Duplicate Evidence

- Duplicate evidence is not treated as a major issue for the prototype.
- Repeated evidence across sources may still be useful because it can reinforce that multiple sensors observed related activity.
- Exact duplicate rows or repeated records should be handled during normalization where practical.
- Duplicate handling does not need to be perfect before the first end-to-end prototype works.

## Timestamp Alignment

- Timestamp alignment is assumed to be available for the current test dataset.
- This means events from different sources can be grouped into the same time windows.
- The prototype does not currently solve hard timestamp problems such as clock drift, missing timezone metadata, or delayed log arrival.
- Those issues should be revisited later for a real deployment.

## Event Granularity

- Different event granularities are acceptable for the prototype.
- A Zeek connection, an HTTP request, a host auth message, an alert, or a packet-derived event may represent different levels of detail.
- For now, each becomes timestamped evidence in the normalized stream.
- The LLM is expected to use source meaning and readable evidence to understand that not all rows mean the same kind of event.

## Volume

- Data volume is not considered a blocker for the first prototype.
- The prototype can use selected sources, bounded windows, and limited replay slices.
- Mamba is a good candidate for large sequence processing, but the rest of the system still has practical limits.
- Ollama calls are much slower than Mamba inference, so live LLM assessment can become the bottleneck.

## Mamba Labeling

- Mamba labeling is not part of the current direction.
- The current direction treats Mamba as a sequence encoder and signal producer.
- The LLM is responsible for assessment and recommended action.
- This means the prototype should avoid using Mamba classifier fields such as `prediction`, `prediction_label`, `attack_probability`, or `risk_level`.

## Evaluation Scope

- Formal evaluation is out of scope for the immediate prototype.
- The first goal is to prove the end-to-end path.
- Later evaluation should measure whether the system detects meaningful patterns, produces consistent assessments, avoids hallucination, and handles missing or mixed-source evidence correctly.

## Trust And Explainability

- Trust and explainability are not primary constraints right now.
- The current focus is pipeline construction and LLM reasoning behavior.
- Even so, the prototype should avoid presenting untrained Mamba encoder diagnostics as trustworthy security conclusions.
- Debug information can remain visible in evidence/debug views as long as it is clearly not treated as a verdict.

## Context Selection

- Current context selection is assumed to be good enough for the prototype.
- The current context includes window metadata, source counts, stable identifiers, missing-data information, readable evidence, relationship hints, stream memory, and Mamba encoder diagnostics.
- This assumption should be revisited if the LLM gives shallow, inconsistent, or unsupported assessments.

## Windowing

- Windowing design is simplified for the prototype because the P025 data already includes timestamps.
- The current prototype uses fixed 60-second windows.
- A 60-second window is a practical starting point, not a final architecture decision.
- Later work should test whether shorter, longer, or overlapping windows produce better reasoning.

## Stateful Operation

- Stateful operation is considered handled for the current prototype through app-level stream memory.
- The current stream memory tracks recent windows, previous window ID, embedding history, embedding trends, and previous Mamba representation summaries.
- This is useful for giving the LLM recent context.
- It is not true recurrent Mamba hidden-state carryover.
- True Mamba state carryover may matter later if the system becomes a continuous live stream instead of a replayed-window prototype.

## Mamba Encoder Meaning

- The current P025 Mamba encoder pass proves that mixed-data tensors can move through Mamba and produce representation signals.
- It does not prove security intelligence yet.
- Even if Mamba is used only as an encoder, its representation still needs to be made meaningful later.
- Future options include a compatible checkpoint, supervised training, self-supervised next-event prediction, reconstruction/anomaly learning, contrastive learning, or baseline behavior modeling.
- This is a definite future requirement before treating Mamba embeddings as reliable security evidence.

## Fresh Ollama Runs

- Ollama assessments should be generated fresh for each replay run.
- The active web replay should not reuse cached LLM assessments.
- Saved assessment files can be used as audit logs, but not as cache for deciding what to show during a new run.
- This keeps each run unique and makes the live demo behavior honest.
