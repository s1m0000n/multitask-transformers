from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Set, Union, Optional

import torch.nn as nn
from transformers.file_utils import ModelOutput

from .models import MultitaskModel, HFHead
from .data import Data
from .utils import validate_isinstance


@dataclass
class Task:
    name: str
    head: Union[HFHead, nn.Module]
    data: Data
    compute_metrics: Optional[Callable[[Any], Dict[str, Any]]] = None

    def __post_init__(self) -> None:
        validate_isinstance(self.name, str, "name")
        validate_isinstance(self.head, [HFHead, nn.Module], "head")
        validate_isinstance(self.data, Data, "data")
        validate_isinstance(self.compute_metrics, Callable, "compute_metrics", optional=True)


class Tasks:
    def __init__(self, tasks: Iterable[Any], model_path: Optional[str] = None) -> None:
        self.tasks = {}
        for task in validate_isinstance(tasks, Iterable, "tasks"):
            if not isinstance(task, Task):
                to_task = getattr(task, "to_task", None)
                if to_task is None:
                    raise AttributeError("Tasks passed to 'tasks' must be either instances of 'Task', "
                                         "or implement 'to_task(model_path: str)' method. Got element "
                                         "which failed for both condition checks")
                if model_path is None:
                    raise TypeError("'model_path' must be specified, if tasks need to be configured with .to_task()")
                task = to_task(validate_isinstance(model_path, str, "model_path"))
                if not isinstance(task, Task):
                    raise TypeError("'to_task' must return a 'Task' instance")
            validate_isinstance(task, Task, "task")
            self.tasks[task.name] = task

    def __getitem__(self, item: str) -> Task:
        return self.tasks[item]

    def keys(self):
        return self.tasks.keys()

    def values(self):
        return self.tasks.values()

    def items(self):
        return self.tasks.items()

    @property
    def names(self) -> Set[str]:
        return set(self.keys())

    @property
    def heads(self) -> Dict[str, Union[HFHead, nn.Module]]:
        return {name: task.head for name, task in self.items()}

    @property
    def data(self) -> Dict[str, Data]:
        return {name: self[name].data for name in self.names}

    def model(self, encoder_path: str) -> MultitaskModel:
        return MultitaskModel(encoder_path, self.heads)


