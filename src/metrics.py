import numpy as np
from typing import Callable, Any, Iterable, Dict
from dataclasses import dataclass


@dataclass(frozen=True)
class SequenceClassificationMetric:
    name: str
    fun: Callable[[np.ndarray, np.ndarray], Any]

    def __call__(self, *args, **kwargs):
        return self.fun(*args, **kwargs)


@dataclass(frozen=True)
class SequenceClassificationMetrics:
    metrics: Iterable[SequenceClassificationMetric]

    def __call__(self, logits: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        predictions = np.argmax(logits, axis=-1)
        result = {}
        for metric in self.metrics:
            result[metric.name] = metric(predictions, labels)
        return result
