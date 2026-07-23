# Stage A Live Replay

- Proposed: postpone real event normalization and packet capture.
- Proposed: replay existing UNSW rows one by one as if they are live flow events.
- Proposed flow: preprocessed row stream -> rolling window buffer -> Mamba signal -> StreamMemory -> context builder -> mandatory Ollama assessment.
- Added `RollingWindowBuffer` to collect individual rows into model-sized windows.
- Added `scripts/replay_rows_as_stream.py` to run the row-by-row replay pipeline.
- Added JSONL output so each completed window is saved as one record.
- Updated the replay path so every completed window gets an Ollama LLM assessment.
- Added tests for buffer sliding and window labels.
- Still not done: real Zeek/Suricata/NetFlow ingestion and event normalization.
