# Watcher-SSM Minimal

This project is a minimal prototype for an LLM-based Watcher Agent for cybersecurity network-stream monitoring.

The current version vendors the official `state-spaces/mamba` repo locally and uses the Mamba-1 block where possible, with a fallback path for environments that cannot load the compiled pieces. Ollama reasoning is part of the watcher pipeline.

## Current Prototype

CSV network data -> preprocessing -> windowing -> Mamba-1 style classifier -> stream memory -> context builder -> Ollama assessment

## Future System

Network stream -> S6/Mamba stream memory -> context builder -> LLM Watcher Agent -> assessment and recommended action

## Why S6/Mamba?

Suspicious network behavior often appears across multiple events, not one isolated data point. S6/Mamba can process long sequences efficiently and preserve temporal context.

## Setup

```bash
pip install -r requirements.txt
```

## Download Kaggle Dataset

```bash
python scripts/download_kaggle_dataset.py --dataset mrwellsdavid/unsw-nb15 --output data/raw/network_data.csv
```

This expects Kaggle credentials to be configured for `kagglehub`.

## Train

```bash
python scripts/train.py --config config.yaml
```

## Evaluate

```bash
python scripts/evaluate.py --config config.yaml --checkpoint checkpoints/best_mamba_watcher.pt
```

## Export LLM-Ready Contexts

```bash
python scripts/export_contexts.py --config config.yaml --checkpoint checkpoints/best_mamba_watcher.pt
```

## Compare Raw vs Model Input

```bash
python scripts/compare_raw_vs_s6.py --config config.yaml --rows 5 --windows 3
```

This writes:
- `results/raw_vs_s6_preview.json`
- `results/raw_vs_s6_side_by_side.csv`
- `results/raw_vs_s6_report.html`
- `results/raw_vs_s6_window_plot.png`

The report shows the original UNSW-NB15 rows, the processed feature vectors, and a few example windows.

## Ollama Reasoning Pipeline

```bash
python scripts/reason_with_ollama.py --config config.yaml --model llama3.1 --output results/ollama_assessments.json
```

This sends preprocessed window packets to a local Ollama server at `http://localhost:11434` and expects JSON-only assessments back.
It also saves the exact raw Ollama reply to `results/ollama_raw_responses.json`.

## Stream Pipeline Runner

```bash
python scripts/run_stream_pipeline.py --config config.yaml --checkpoint checkpoints/best_mamba_watcher.pt --output results/stream_pipeline.json
```

This runs the full watcher flow in one place:
- network windows
- Mamba signal
- stream memory update
- context packet build
- mandatory Ollama assessment

## Stage A Live Replay

```bash
python scripts/replay_rows_as_stream.py --config config.yaml --checkpoint checkpoints/best_mamba_watcher.pt --max-windows 5 --output results/live_replay.jsonl
```

This replays preprocessed dataset rows one by one as if they are live flow events. A rolling buffer builds model-sized windows before Mamba runs, then Ollama produces the local LLM watcher assessment.

## Web Simulator

```bash
cd web
npm.cmd install
node node_modules/vite/bin/vite.js build
cd ..
python scripts/web_simulator_server.py --config config.yaml --checkpoint checkpoints/best_mamba_watcher.pt --port 8000 --web-root web/dist
```

This starts a local React dashboard backed by a Python WebSocket server. The browser streams window updates, stream-memory state, and Ollama assessments live.

## Future Work

- Add real-time streaming
- Add packet-level or Zeek/Suricata log ingestion
- Add S7 as an alternative state-space backbone
- Add analyst-facing alert explanations
