from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping
import datetime as _dt
from dataclasses import dataclass, field

@dataclass
class GAMetrics:
    """Simple genetic algorithm metrics writer"""
    mean_fitnesses: list[float] = field(default_factory=list)
    max_fitnesses: list[float] = field(default_factory=list)
    min_fitnesses: list[float] = field(default_factory=list)
    std_fitnesses: list[float] = field(default_factory=list)


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

def plot_metrics(metrics: GAMetrics, output_dir: Path) -> None:
    """Plot fitness metrics over generations and save to output_dir."""
    import matplotlib.pyplot as plt

    generations = range(len(metrics.mean_fitnesses))

    plt.figure(figsize=(10, 6))
    plt.plot(generations, metrics.mean_fitnesses, label='Mean Fitness', linewidth =2.5)
    plt.plot(generations, metrics.max_fitnesses, label='Max Fitness', linewidth =2.5)
    plt.plot(generations, metrics.min_fitnesses, label='Min Fitness', linewidth =2.5)
    lower = [m - s for m, s in zip(metrics.mean_fitnesses, metrics.std_fitnesses)]
    upper = [m + s for m, s in zip(metrics.mean_fitnesses, metrics.std_fitnesses)]
    plt.fill_between(generations, lower, upper, color='gray', alpha=0.6, label='Std Dev')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness over Generations')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'fitness_plot.png')
    plt.close()