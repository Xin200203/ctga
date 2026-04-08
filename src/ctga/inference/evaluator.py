"""Evaluation entry points and metric aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MetricMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float) -> None:
        self.total += value
        self.count += 1

    @property
    def average(self) -> float:
        return self.total / max(self.count, 1)


@dataclass
class Evaluator:
    metrics: dict[str, MetricMeter] = field(default_factory=dict)

    def update(self, **kwargs: float) -> None:
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = MetricMeter()
            self.metrics[key].update(float(value))

    def summarize(self) -> dict[str, float]:
        return {key: meter.average for key, meter in self.metrics.items()}
