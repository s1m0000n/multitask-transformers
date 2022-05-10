from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Set, Iterable
import numpy as np§
import matplotlib.pyplot as plt

from src.utils import validate_isinstance
from dataclasses import dataclass


@dataclass(frozen=True)
class MetricConfig:
    metric: str
    task: Optional[str] = None
    make_legend_label: Callable[[str, str], str] = lambda task, metric: f"{task} / {metric}"
    line: Optional[str] = "-"
    marker: Optional[str] = None
    color: Optional[str] = None

    @property
    def label(self) -> str:
        task = "meta" if self.task is None else self.task
        return self.make_legend_label(self.task, self.metric)

    def copy(self, **params):
        return MetricConfig(**dict(self.__dict__, **params))


@dataclass(frozen=True)
class PlotConfig:
    title: str = "Unnamed plot"
    show_legend: bool = True
    xlabel: str = "Steps"
    ylabel: str = "Value"
    show: bool = False

    def copy(self, **params):
        return PlotConfig(**dict(self.__dict__, **params))


class MultitaskMetricsLog:
    def __init__(self, validate: bool = True) -> None:
        self.data: Dict[str, List[Dict[str, Any]]] = {}
        self.meta = {}
        self.validate = validate

    def add(self, task: str, metrics: Dict[str, Any]) -> 'MultitaskMetricsLog':
        if validate_isinstance(task, str, "task", validate=self.validate) not in self.data:
            self.data[task] = [metrics, ]
        else:
            self.data[task].append(metrics)
        return self

    def add_meta(self, metric_name: str, value: Any) -> 'MultitaskMetricsLog':
        if validate_isinstance(metric_name, str, "metric_name", validate=self.validate) not in self.meta:
            self.meta[metric_name] = [value, ]
        else:
            self.meta[metric_name].append(value)
        return self

    @property
    def tasks(self) -> Iterable[str]:
        return self.data.keys()

    def get(
            self,
            task: str,
            metrics: Optional[Union[str, List[str], Tuple[str], Set[str]]] = None
    ) -> Union[np.ndarray, Dict[str, np.ndarray], List[Dict[str, Any]]]:
        task_data = self.data[validate_isinstance(task, str, "task")]
        if metrics is None:
            return task_data
        if isinstance(metrics, str):
            return np.array([d[metrics] for d in task_data])
        if isinstance(metrics, (list, tuple, set)):
            result = {}
            for metric in metrics:
                result[metric] = self.get(task, metric)
            return result
        raise TypeError("Wrong type for 'metrics', must be an instance of str | (list | tuple | set)[str] | None")

    def get_meta(
            self,
            metrics: Union[str, List[str], Tuple[str], Set[str]]
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if isinstance(metrics, str):
            return np.array(self.meta[metrics])
        if isinstance(metrics, (list, tuple, set)):
            result = {}
            for metric in metrics:
                result[metric] = self.get_meta(metric)
            return result
        raise TypeError("Wrong type for 'metrics', must be an instance of str | (list | tuple | set)[str]")

    def metrics(self, task: Optional[str] = None) -> Iterable[str]:
        if task is None:
            return self.meta.keys()
        task_data = self.get(task)
        if len(task_data) < 0:
            raise IndexError("Task log is empty, so no logs available")
        return self.get(task)[0].keys()

    def plot(
        self,
        *metric_configs: MetricConfig,
        plot_config: Optional[PlotConfig] = None
    ) -> None:
        plot_config = PlotConfig() if plot_config is None else plot_config
        if len(metric_configs) == 0:
            configs = [MetricConfig(metric, task) for task in self.tasks for metric in self.metrics(task)]
            configs += [MetricConfig(metric) for metric in self.metrics()]
        for cfg in metric_configs:
            y = self.get_meta(cfg.metric) if cfg.task is None else self.get(cfg.task, cfg.metric)
            plt.plot(
                y,
                marker=cfg.marker if cfg.marker else "",
                linestyle=cfg.line if cfg.line else "",
                color=cfg.color,
                label=cfg.label
            )
        plt.xlabel(plot_config.xlabel)
        plt.ylabel(plot_config.ylabel)
        plt.title(plot_config.title)
        if plot_config.show_legend:
            plt.legend()
        if plot_config.show:
            plt.show()
