from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.llm.ollama_client import OllamaClient
from src.mixed_data.llm_assessment import assess_p025_packet
from src.mixed_data.llm_packets import assert_no_forbidden_keys
from src.utils import ensure_dir


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _assessment_for_display(assessment: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in assessment.items()
        if key not in {"llm_input_packet", "raw_response_text"}
    }


def _assert_p025_packet(packet: dict[str, Any]) -> None:
    packet_type = packet.get("packet_type")
    if packet_type != "p025_mixed_window_llm_evidence":
        raise ValueError(
            "The web replay only accepts P025 mixed-source packets. "
            f"Expected packet_type='p025_mixed_window_llm_evidence', got {packet_type!r}."
        )


@dataclass
class P025WebReplaySession:
    packets_path: str = "results/p025_llm_packets.jsonl"
    assessments_path: str = "results/p025_web_ollama_assessments.jsonl"
    start_window: int = 0
    max_windows: int | None = None
    ollama_model: str = "llama3.1"
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout: int = 240

    def __post_init__(self) -> None:
        self.packets_file = Path(self.packets_path)
        if not self.packets_file.exists():
            raise FileNotFoundError(f"P025 LLM packets file does not exist: {self.packets_file}")
        self.packets = _read_jsonl(self.packets_file)
        for packet in self.packets:
            _assert_p025_packet(packet)
            assert_no_forbidden_keys(packet)

        self.assessments_file = Path(self.assessments_path)
        self.client = OllamaClient(
            base_url=self.ollama_base_url,
            model_name=self.ollama_model,
            timeout=self.ollama_timeout,
        )
        self.reset()

    def reset(self) -> None:
        self.run_id = str(uuid.uuid4())
        self.window_index = int(self.start_window)
        self.windows_emitted = 0
        self.events_seen = 0
        self.finished = False

    def status(self) -> dict[str, Any]:
        return {
            "row_index": int(self.window_index),
            "events_seen": int(self.events_seen),
            "windows_emitted": int(self.windows_emitted),
            "buffer_pending": 0,
            "finished": bool(self.finished),
            "max_windows": self.max_windows,
            "total_rows": int(len(self.packets)),
            "total_windows": int(len(self.packets)),
            "stream_mode": "p025_encoder_llm",
            "run_id": self.run_id,
        }

    def _limit_reached(self) -> bool:
        return self.max_windows is not None and self.windows_emitted >= self.max_windows

    def has_more_windows(self) -> bool:
        return self.window_index < len(self.packets)

    def _assessment_for_packet(self, packet: dict[str, Any]) -> dict[str, Any]:
        assessment = assess_p025_packet(self.client, packet)
        assessment_with_run = {**assessment, "run_id": self.run_id}
        ensure_dir(self.assessments_file.parent)
        with self.assessments_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(assessment_with_run) + "\n")
        return assessment

    def _build_record(self, packet: dict[str, Any], assessment: dict[str, Any]) -> dict[str, Any]:
        window = packet.get("window_metadata", {})
        record = {
            "record_type": "p025_encoder_llm_window",
            "run_id": self.run_id,
            "window_id": int(window.get("window_id", self.window_index)),
            "events_seen": int(self.events_seen),
            "pending_buffer_rows": 0,
            "window": window,
            "llm_packet": packet,
            "stream_memory": packet.get("stream_memory", {}),
            "llm_assessment": _assessment_for_display(assessment),
            "assessment_source": "ollama_live",
        }
        assert_no_forbidden_keys(record)
        return record

    async def step_row(self) -> list[dict[str, Any]]:
        if self.finished or not self.has_more_windows() or self._limit_reached():
            self.finished = True
            return []

        packet = self.packets[self.window_index]
        event_count = int(packet.get("window_metadata", {}).get("event_count") or 0)
        assessment = await asyncio.to_thread(self._assessment_for_packet, packet)
        self.events_seen += event_count
        record = self._build_record(packet, assessment)
        self.window_index += 1
        self.windows_emitted += 1
        if not self.has_more_windows() or self._limit_reached():
            self.finished = True
        return [record]
