import numpy as np
from typing import Callable, Any, Iterable, Dict
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


@dataclass(frozen=True)
class Metric:
    name: str
    fun: Callable[[np.ndarray, np.ndarray], Any]

    def __call__(self, *args, **kwargs):
        return self.fun(*args, **kwargs)


@dataclass(frozen=True)
class SequenceClassificationMetrics:
    metrics: Iterable[Metric]

    def __call__(self, logits: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        predictions = np.argmax(logits, axis=-1)
        result = {}
        for metric in self.metrics:
            result[metric.name] = metric(labels, predictions)
        return result


accuracy = Metric(
    name="accuracy",
    fun=accuracy_score
)

precision = Metric(
    name="precision",
    fun=precision_score
)

recall = Metric(
    name="recall",
    fun=recall_score
)

f1 = Metric(
    name="f1",
    fun=f1_score
)
