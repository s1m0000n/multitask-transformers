from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union, Set
import numpy as np
import matplotlib.pyplot as plt

from src.utils import validate_isinstance


class MetricsLog:
    def __init__(self, validate: bool = True) -> None:
        self.data: Dict[str, List[Dict[str, Any]]] = {}
        self.meta = {}
        self.validate = validate

    def add(self, task: str, metrics: Dict[str, Any]) -> 'MetricsLog':
        validate_isinstance(metrics, dict, "metrics", validate=self.validate)
        if validate_isinstance(task, str, "task", validate=self.validate) not in self.data:
            self.data[task] = [metrics, ]
        else:
            self.data[task].append(metrics)
        return self

    def add_meta(self, metric_name: str, value: Any) -> 'MetricsLog':
        if validate_isinstance(metric_name, str, "metric_name", validate=self.validate) not in self.meta:
            self.meta[metric_name] = [value, ]
        else:
            self.meta[metric_name].append(value)
        return self

    def get(self, task: str, metrics: Optional[Union[str, List[str], Tuple[str], Set[str]]] = None) -> Union[np.ndarray, Dict[str, np.ndarray], List[Dict[str, Any]]]:
        task_data = self.data[validate_isinstance(task, str, "task")]
        if metrics is None:
            return task_data
        elif isinstance(metrics, str):
            return np.array([d[metrics] for d in task_data])
        elif isinstance(metrics, (list, tuple, set)):
            result = {}
            for metric in metrics:
                result[metric] = self.get(task, metric)
            return result
        else:
            raise TypeError("Wrong type for 'metrics', must be an instance of str | (list | tuple | set)[str] | None")

    def get_meta(self, metric_names: Union[str, List[str], Tuple[str], Set[str]]) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if isinstance(metric_names, str):
            return np.array(self.meta[metric_names])
        elif isinstance(metric_names, (list, tuple, set)):
            result = {}
            for metric in metric_names:
                result[metric] = self.get_meta(metric)
            return result
        else:
             raise TypeError("Wrong type for 'metrics', must be an instance of str | (list | tuple | set)[str]")

    def plot(self, task_metrics: Dict[str, str]) -> None:
        raise NotImplemented("Implement metrics plotting")

    def plot_meta(self, metric_names: Union[str, List[str], Tuple[str], Set[str]]) -> None:
        raise NotImplemented("Implement meta-metrics plotting")

