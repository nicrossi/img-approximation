from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping
import datetime as _dt


def write_metrics(
        cfg: Mapping[str, Any],
        output_dir: Path,
        elapsed_time: float,
        extra: Mapping[str, Any] | None = None,
) -> None:
    """Write a metrics.json file with configuration and run info."""
    metrics: dict[str, Any] = {
        "parameters": cfg,
        "elapsed_time": f"{elapsed_time:.2f}",
        "completed_at": _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=-3))).isoformat(),
    }
    if extra:
        metrics.update(extra)

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))