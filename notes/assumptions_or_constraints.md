# Assumptions And Constraints

These notes capture the current assumptions for moving from a single flow dataset toward a mixed, timestamped Watcher Agent prototype.

- The key question is: after different data sources are converted into one shared event shape, what challenges and implications still remain for the pipeline?

- Duplicate evidence is not treated as a major issue for the prototype. The assumption is that repeated evidence across sources can still be useful context for the Watcher Agent.

- Duplicate rows or repeated records will be handled during the shared-shape normalization stage where possible.

- Timestamp alignment is assumed to be available for testing because the current mixed dataset already includes timestamped records across sources.

- Different event granularity is not treated as a blocking issue for the prototype. Packet data, flow summaries, alerts, and host logs can all become timestamped evidence records, with the LLM helping interpret their meaning in context.

- Data volume is not treated as a blocker for the first prototype. The assumption is that we can start with bounded windows, selected sources, or limited replay slices if the full volume becomes too heavy.

- Mamba labeling is not part of the current direction. For now, Mamba is treated as a sequence encoder/signal producer rather than a supervised attack classifier.

- Formal evaluation is out of scope for the immediate prototype. The first goal is to make the end-to-end pipeline work before building a rigorous scoring framework.

- Trust and explainability are not being treated as primary constraints right now. The current focus is pipeline construction and LLM reasoning behavior.

- Context selection is assumed to be good enough for the prototype. The current context builder is treated as sufficient unless later testing shows that the LLM needs different or richer evidence.

- Windowing design is considered simplified for the prototype because the mixed dataset already provides timestamps. The current assumption is to use fixed 60-second windows and revisit the window size later.

## Important Considerations

- Stateful operation is considered handled for the current prototype through the app-level stream memory implementation.

- The current stream memory tracks recent windows, last signal, last delta, rolling trends, embedding L2 norm history, previous Mamba representation summaries, window count, and last window ID.

- This is app-level stateful operation, not true recurrent Mamba hidden-state carryover. That distinction is acknowledged, but it is not treated as a blocker for the prototype.
