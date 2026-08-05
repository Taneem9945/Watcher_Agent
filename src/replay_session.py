from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .context_builder import build_context_packet
from .data_loader import load_network_dataset
from .llm.ollama_client import OllamaClient, OllamaError
from .llm.ollama_watcher import OllamaWatcher
from .mamba_model import MambaWatcher
from .mamba_signal import infer_window_signal
from .preprocessing import preprocess_dataframe
from .stream_buffer import RollingWindowBuffer
from .stream_memory import StreamMemory
from .utils import load_config


def _load_model(config: dict[str, Any], checkpoint_path: str, device: torch.device, input_dim: int) -> MambaWatcher:
    model_cfg = config["model"]
    model = MambaWatcher(
        input_dim=input_dim,
        d_model=int(model_cfg["d_model"]),
        d_state=int(model_cfg["d_state"]),
        d_conv=int(model_cfg["d_conv"]),
        expand=int(model_cfg["expand"]),
        num_classes=int(model_cfg["num_classes"]),
        dropout=float(model_cfg["dropout"]),
        pooling=model_cfg.get("pooling", "last"),
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@dataclass
class ReplaySession:
    config_path: str = "config.yaml"
    checkpoint_path: str = "checkpoints/best_mamba_watcher.pt"
    start_row: int = 0
    max_windows: int | None = None
    ollama_model: str = "llama3.1"
    ollama_base_url: str = "http://localhost:11434"

    def __post_init__(self) -> None:
        self.config = load_config(self.config_path)
        self.data_cfg = self.config["data"]
        device_name = self.config["project"].get("device", "cpu")
        self.device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")

        self.df = load_network_dataset(
            self.data_cfg["raw_path"],
            label_col=self.data_cfg["label_col"],
            timestamp_col=self.data_cfg.get("timestamp_col"),
        )
        drop_columns = list(self.data_cfg.get("drop_columns", []))
        if self.data_cfg.get("timestamp_col") and self.data_cfg["timestamp_col"] in self.df.columns:
            drop_columns.append(self.data_cfg["timestamp_col"])

        self.X, self.y, self.feature_names, _ = preprocess_dataframe(
            self.df,
            label_col=self.data_cfg["label_col"],
            categorical_columns=self.data_cfg.get("categorical_columns", []),
            drop_columns=drop_columns,
            binary_label=bool(self.data_cfg.get("binary_label", True)),
            attack_label_values=self.data_cfg.get("attack_label_values", []),
        )
        window_cfg = self.config["windowing"]
        self.buffer = RollingWindowBuffer(
            window_size=int(window_cfg["window_size"]),
            step_size=int(window_cfg["step_size"]),
            label_strategy=window_cfg.get("label_strategy", "any_attack"),
        )
        self.model = _load_model(self.config, self.checkpoint_path, self.device, len(self.feature_names))
        self.watcher = OllamaWatcher(
            client=OllamaClient(base_url=self.ollama_base_url, model_name=self.ollama_model)
        )
        self.stream_memory = StreamMemory()
        self.prev_context_packet: dict[str, Any] | None = None
        self.prev_signal_packet: dict[str, Any] | None = None
        self.row_index = int(self.start_row)
        self.events_seen = 0
        self.windows_emitted = 0
        self.finished = False

    def reset(self) -> None:
        self.buffer = RollingWindowBuffer(
            window_size=self.buffer.window_size,
            step_size=self.buffer.step_size,
            label_strategy=self.buffer.label_strategy,
        )
        self.stream_memory = StreamMemory()
        self.prev_context_packet = None
        self.prev_signal_packet = None
        self.row_index = int(self.start_row)
        self.events_seen = 0
        self.windows_emitted = 0
        self.finished = False

    def status(self) -> dict[str, Any]:
        return {
            "row_index": int(self.row_index),
            "events_seen": int(self.events_seen),
            "windows_emitted": int(self.windows_emitted),
            "buffer_pending": int(self.buffer.pending_count()),
            "finished": bool(self.finished),
            "max_windows": self.max_windows,
            "total_rows": int(len(self.X)),
        }

    def has_more_rows(self) -> bool:
        return self.row_index < len(self.X)

    def _emit_window(self, buffered_window) -> dict[str, Any]:
        signal_packet = infer_window_signal(
            model=self.model,
            X_window=buffered_window.X_window,
            window_id=self.windows_emitted,
            true_label=buffered_window.y_window,
            device=self.device,
            previous_signal=self.prev_signal_packet,
        )
        signal_row = signal_packet["signal_rows"][0]
        stream_memory_before = self.stream_memory.snapshot()
        stream_memory_after = self.stream_memory.update(signal_packet)
        context_packet = build_context_packet(
            window_id=self.windows_emitted,
            X_window=buffered_window.X_window,
            feature_names=self.feature_names,
            prediction=int(signal_row["prediction"]),
            attack_probability=float(signal_row["attack_probability"]),
            true_label=buffered_window.y_window,
            previous_context=self.prev_context_packet,
            model_signal=signal_row,
            previous_model_signal=(self.prev_signal_packet["signal_rows"][0] if self.prev_signal_packet else None),
            stream_memory=stream_memory_after,
        )
        try:
            result = self.watcher.assess_context_packet(context_packet, prompt_mode="encoder")
        except OllamaError as exc:
            raise RuntimeError(str(exc)) from exc

        record = {
            "window_id": self.windows_emitted,
            "events_seen": int(self.events_seen),
            "source_row_start": buffered_window.start_event_index,
            "source_row_end": buffered_window.end_event_index,
            "pending_buffer_rows": self.buffer.pending_count(),
            "true_label": buffered_window.y_window,
            "mamba_signal_packet": signal_packet,
            "stream_memory_before": stream_memory_before,
            "stream_memory_after": stream_memory_after,
            "context_packet": context_packet,
            "llm_input_packet": result.get("llm_input_packet", {}),
            "llm_assessment": {k: v for k, v in result.items() if k != "raw_response_text"},
            "raw_response_text": result.get("raw_response_text", ""),
        }
        self.prev_context_packet = context_packet
        self.prev_signal_packet = signal_packet
        self.windows_emitted += 1
        if self.max_windows is not None and self.windows_emitted >= self.max_windows:
            self.finished = True
        return record

    async def step_row(self) -> list[dict[str, Any]]:
        if self.finished or not self.has_more_rows():
            self.finished = True
            return []

        self.buffer.add(self.X[self.row_index], int(self.y[self.row_index]), event_index=self.row_index)
        self.row_index += 1
        self.events_seen += 1

        emitted: list[dict[str, Any]] = []
        while self.buffer.ready() and not self.finished:
            buffered = self.buffer.pop_window()
            emitted.append(await asyncio.to_thread(self._emit_window, buffered))
            if self.finished:
                break
        if not self.has_more_rows() and not self.buffer.ready():
            self.finished = True
        return emitted
