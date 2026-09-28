"""
Hardware Telemetry & Experiment Tracker
=======================================
Context manager for recording experiment telemetry, hardware metrics,
peak VRAM utilization, and execution results into machine-readable JSONL
and human-readable Markdown formats.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import torch


@dataclass
class ExperimentRecord:
    """Dataclass holding structured telemetry and metadata for a single run."""

    timestamp: str
    experiment_id: str
    commit: str
    device: str
    arch: str
    cuda_version: str
    peak_vram_mb: float
    duration_sec: float
    status: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_markdown_row(self) -> str:
        """Format the experiment record as a row in a Markdown table."""
        # Sanitize pipe symbols and line breaks to preserve table structure
        sanitized_notes = self.notes.replace("|", "\\|").replace("\n", " ").strip()
        formatted_metrics = ", ".join(f"{k}: {v}" for k, v in self.metrics.items())
        formatted_metrics = formatted_metrics.replace("|", "\\|").replace("\n", " ").strip()

        if formatted_metrics and sanitized_notes:
            notes_col = f"{formatted_metrics} | {sanitized_notes}"
        elif formatted_metrics:
            notes_col = formatted_metrics
        elif sanitized_notes:
            notes_col = sanitized_notes
        else:
            notes_col = "-"

        device_col = f"{self.device} ({self.arch})"
        vram_col = f"{self.peak_vram_mb:.2f} MB"
        duration_col = f"{self.duration_sec:.3f}s"

        return (
            f"| {self.timestamp} | {self.experiment_id} | {self.commit} | "
            f"{device_col} | {vram_col} | {duration_col} | {self.status} | {notes_col} |"
        )


class ExperimentTracker:
    """
    Automated laboratory notebook context manager.

    Measures elapsed time, peak GPU VRAM allocation, hardware specs,
    and logs outcomes to both docs/experiments.jsonl and docs/EXPERIMENT_LOG.md.
    """

    def __init__(
        self,
        experiment_id: str,
        notes: str = "",
        log_dir: Optional[str | Path] = None,
    ) -> None:
        self.experiment_id: str = experiment_id
        self.notes: str = notes
        self.metrics: Dict[str, Any] = {}
        self.status: str = "SUCCESS"
        self.start_time: float = 0.0
        self.duration_sec: float = 0.0
        self.peak_vram_mb: float = 0.0

        if log_dir is not None:
            self.log_dir = Path(log_dir)
        else:
            # Default to repo_root/docs
            self.log_dir = Path(__file__).resolve().parent.parent / "docs"

    def log_metric(self, key: str, value: Any) -> None:
        """Record a custom experiment metric (e.g. loss, accuracy, throughput)."""
        self.metrics[key] = value

    def set_status(self, status: str) -> None:
        """Override the outcome status of the experiment run."""
        self.status = status

    def _get_git_commit(self) -> str:
        """Fetch the current short Git commit hash."""
        try:
            res = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
                cwd=self.log_dir.parent,
            )
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout.strip()
        except Exception:
            pass
        return "unknown"

    def _get_hardware_info(self) -> tuple[str, str, str]:
        """Detect and return (device_name, arch_tag, cuda_version)."""
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            arch = f"sm_{cap[0]}{cap[1]}"
            cuda_ver = torch.version.cuda or "N/A"
        else:
            device_name = "CPU"
            arch = "N/A"
            cuda_ver = "N/A"
        return device_name, arch, cuda_ver

    def __enter__(self) -> ExperimentTracker:
        """Prepare hardware telemetry and start the precision timer."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        self.start_time = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool:
        """Capture elapsed metrics, query hardware, and persist logs."""
        self.duration_sec = time.perf_counter() - self.start_time

        if exc_type is not None:
            if self.status == "SUCCESS":
                self.status = f"FAILED: {exc_val}"

        if torch.cuda.is_available():
            self.peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
        else:
            self.peak_vram_mb = 0.0

        device_name, arch, cuda_version = self._get_hardware_info()
        commit_hash = self._get_git_commit()
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        record = ExperimentRecord(
            timestamp=timestamp_str,
            experiment_id=self.experiment_id,
            commit=commit_hash,
            device=device_name,
            arch=arch,
            cuda_version=cuda_version,
            peak_vram_mb=round(self.peak_vram_mb, 2),
            duration_sec=round(self.duration_sec, 4),
            status=self.status,
            metrics=self.metrics,
            notes=self.notes,
        )

        self._save_record(record)

        # Do not suppress exceptions
        return False

    def _save_record(self, record: ExperimentRecord) -> None:
        """Persist record to JSONL and Markdown log files."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 1. Machine-readable JSON-Lines log
        jsonl_path = self.log_dir / "experiments.jsonl"
        with open(jsonl_path, "a", encoding="utf-8") as f_jsonl:
            f_jsonl.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

        # 2. Human-readable Markdown log table
        md_path = self.log_dir / "EXPERIMENT_LOG.md"
        header = (
            "# Automated Experiment Research Log\n\n"
            "| Timestamp | ID | Commit | Device (Arch) | Peak VRAM | Duration | Status | Metrics / Notes |\n"
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
        )

        needs_header = not md_path.exists() or md_path.stat().st_size == 0

        with open(md_path, "a", encoding="utf-8") as f_md:
            if needs_header:
                f_md.write(header)
            f_md.write(record.to_markdown_row() + "\n")
