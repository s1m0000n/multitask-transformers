# TODO: Docstrings for everything
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datasets import DatasetDict
from transformers import AutoModelForSequenceClassification
from typing import Optional, Callable, Iterable, Type, Union, Dict, Set

from .data import Data, MultitaskDataLoader, MultitaskBatchSampler
from .metrics import SequenceClassificationMetrics
from .models import Head, MultitaskModel
from .preprocessing import Preprocessor
from .tokenizers import TokenizerConfig


@dataclass(frozen=True)
class Task:
    name: str
    head: Head
    data: Data
    metrics: Optional[Callable] = None

    def __post_init__(self):
        if not isinstance(self.name, str):
            raise TypeError("Wrong type for 'name', must be an instance of 'str'")
        if not isinstance(self.head, Head):
            raise TypeError("Wrong type for 'head', must be an instance of 'Head'")
        if not isinstance(self.data, Data):
            raise TypeError("Wrong type for 'data', must be an instance of 'Data'")
        if self.metrics is not None and not callable(self.metrics):
            raise TypeError("Wrong type for 'metrics', must be a callable, _call__() method not implemented")


class CastableToTask(ABC):
    """
    Abstract class for specific _tasks, which can be casted to Task with implemented method
    'make_task(model_path: str) -> Task'
    """

    @abstractmethod
    def make_task(self, model_path: str) -> Task:
        pass


@dataclass(frozen=True)
class SequenceClassificationTask(CastableToTask):
    name: str
    dataset_dict: DatasetDict
    num_labels: int = 2
    metrics: Optional[SequenceClassificationMetrics] = None
    preprocessor: Optional[Preprocessor] = None
    tokenizer_config: Optional[TokenizerConfig] = None

    def __post_init__(self) -> None:
        if self.metrics is not None and not isinstance(self.metrics, SequenceClassificationMetrics):
            raise TypeError("Wrong type for 'metrics', must be either equal None, "
                            "or be an instance of 'SequenceClassificationMetrics'")

    def make_task(self, model_path: str) -> Task:
        tokenizer_config = TokenizerConfig() if self.tokenizer_config is None else self.tokenizer_config
        return Task(
            name=self.name,
            head=Head(
                class_=AutoModelForSequenceClassification,
                config_params={"num_labels": self.num_labels},
            ),
            data=Data(
                data=self.dataset_dict,
                configured_tokenizer=tokenizer_config.make_configured_tokenizer(model_path),
                preprocessor=self.preprocessor
            ),
            metrics=self.metrics
        )


class Tasks:
    def __init__(self, model_path: str, tasks: Iterable[Union[Task, Type[CastableToTask]]]) -> None:
        if not isinstance(model_path, str):
            raise TypeError("Wrong type for 'model_path', must be an instance of 'str'")
        self.model_path = model_path
        self._model = None
        self._tasks = {}
        for task in tasks:
            if isinstance(task, Task):
                self._tasks[Task.name] = Task
            else:
                cast_fun = getattr(task, "make_task", None)
                if cast_fun is None:
                    raise AttributeError("Expected _tasks to contain instances of Task or classes castable to Task, "
                                         "implementing method 'make_task(model_path: str) -> Task', "
                                         "got class, which is not Task, which not implements method 'make_task'")
                casted_task = cast_fun(self.model_path)
                if not isinstance(casted_task, Task):
                    raise TypeError(f"Wrong type for result of using 'cast_task' on castable task - "
                                    f"instance of '{type(task)}', expected 'Task', but got '{type(casted_task)}'")
                self._tasks[casted_task.name] = casted_task

    def __getitem__(self, task_name: str) -> Task:
        if task_name not in self._tasks:
            raise KeyError(f"Task named {task_name} is not found")
        return self._tasks[task_name]

    @property
    def names(self) -> Set[str]:
        return set(self._tasks.keys())

    @property
    def tasks(self) -> Dict[str, Task]:
        return self._tasks

    @property
    def heads(self) -> Dict[str, Head]:
        return {name: task.head for name, task in self.tasks.items()}

    @property
    def data(self) -> Dict[str, Data]:
        return {name: task.data for name, task in self.tasks.items()}

    @property
    def model(self) -> MultitaskModel:
        if self._model is None:
            self._model = MultitaskModel.create(self.model_path, self.heads)
        return self._model

    def keys(self) -> Set[str]:
        return self.names

    def values(self):
        return set(self._tasks.values())

    def items(self) -> Dict[str, Task]:
        return self.tasks

    def make_model(self) -> MultitaskModel:
        return self.model

    def make_data_loader(
            self,
            part: str,
            batch_size: Optional[int] = None,
            task_batch_sizes: Optional[Dict[str, int]] = None,
            shuffle_task_data: Union[bool, Dict[str, bool]] = True,
            shuffle_batches: bool = True,
            columns: Iterable[str] = ("input_ids", "attention_mask", "labels")
    ) -> MultitaskDataLoader:
        return MultitaskDataLoader(
            task_datasets=self.data,
            part=part,
            batch_size=batch_size,
            task_batch_sizes=task_batch_sizes,
            shuffle_task_data=shuffle_task_data,
            shuffle_batches=shuffle_batches,
            columns=columns
        )

    def make_batch_sampler(
            self,
            part: str,
            batch_size: Optional[int] = None,
            task_batch_sizes: Optional[Dict[str, int]] = None,
            shuffle_task_data: Union[bool, Dict[str, bool]] = True,
            shuffle_batches: bool = True,
            columns: Iterable[str] = ("input_ids", "attention_mask", "labels")
    ) -> MultitaskBatchSampler:
        return MultitaskBatchSampler(
            tasks=self,
            part=part,
            batch_size=batch_size,
            task_batch_sizes=task_batch_sizes,
            shuffle_task_data=shuffle_task_data,
            shuffle_batches=shuffle_batches,
            columns=columns
        )