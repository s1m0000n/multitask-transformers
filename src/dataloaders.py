"""
Data loaders, batch samplers and other iterables with support for multitask learning
"""
import math
from typing import Callable, Dict, Optional, List, Any, Generator
from dataclasses import dataclass, field

import numpy as np
from datasets.arrow_dataset import Dataset

from .data import Data
from .utils import slice_into_chunks, validate_isinstance


@dataclass
class DataLoader:
    """
    Just regular implementation of data loader with collation

    :param data: Dataset with PyTorch feature dicts
    :param collate_fun: Data collator callable
    :param batch_size: Integer batch size
    :param shuffle: If true => data is shuffled before slicing into batches & collation
    """
    data: Dataset
    collate_fun: Callable[[List[Dict[str, Any]]], Dict[str, Any]]
    batch_size: int
    shuffle: bool = True

    batch_data: Generator = field(init=False)

    def __len__(self) -> int:
        return math.ceil(len(self.data) / self.batch_size)

    def __iter__(self) -> 'DataLoader':
        data = self.data.shuffle() if self.shuffle else self.data
        self.batch_data = slice_into_chunks(self.batch_size, data)
        return self

    def __next__(self):
        # If there is more => next just retrieves a new batch
        # Else => StopIteration() is propagated further
        return self.collate_fun(next(self.batch_data))


@dataclass(frozen=True)
class TaskBatch:
    name: str
    data: Dict[str, Any]


@dataclass
class MultitaskBatchSampler:
    """
    Data loader alternative for multitask learning, yielding everything necessary
    for running the data through the base model & task head with loss & metrics

    Meant to be used for training on mixed batches from all tasks, optimizing on a single
    task per batch (computing single loss, and updating encoder + tasks's head)

    :param tasks: Some aggregated tasks

    :param part: General part name for unfilled in 'parts'
    :param parts: Mapping task name -> part name

    :param batch_size: General batch size for unfilled in 'batch_sizes'
    :param batch_sizes: Mapping task name -> batch size

    :param shuffle_data: General shuffling data for unfilled in 'shuffle_task_data'
    :param shuffle_task_data: Mapping task name -> shuffle data?

    :param shuffle_batches: Determines whether batches are shuffled (batches of all tasks in set)
    :param columns: Columns to be prepared for training with pytorch (* => torch.Tensor)
    """
    datasets: Dict[str, Data]
    part: Optional[str] = None
    parts: Optional[Dict[str, str]] = None
    batch_size: Optional[int] = None
    batch_sizes: Optional[Dict[str, int]] = None
    shuffle_data: bool = True
    shuffle_task_data: Optional[Dict[str, bool]] = None

    data_loaders: Dict[str, DataLoader] = field(init=False)
    num_batches_per_task: Dict[str, int] = field(init=False)
    sampler: Callable[[], str] = field(init=False)

    def __post_init__(self) -> None:
        if self.batch_sizes is None:
            self.batch_sizes = {}
        if self.shuffle_task_data is None:
            self.shuffle_task_data = {}
        if self.parts is None:
            self.parts = {}

        # Validating & figuring out all params

        for task in self.datasets:
            # -> Batch sizes
            if task not in self.batch_sizes:
                if self.batch_size is None:
                    raise ValueError("Expected to have either batch size for all tasks in "
                                     "'batch_sizes' and 'batch_size': Optional[Any] or"
                                     "skipped some tasks in 'batch_sizes' with default in 'batch_size' or"
                                     "'batch_sizes' = None, 'batch_size': int "
                                     "for all tasks to have the same batch size")
                self.batch_sizes[task] = validate_isinstance(self.batch_size, int, "batch_size")
            else:
                validate_isinstance(
                    self.batch_sizes[task], int, f"self.batch_sizes[\"{task}\"]")

            # -> Parts
            if task not in self.parts:
                if self.part is None:
                    raise ValueError("TODO error text")
                self.parts[task] = validate_isinstance(self.part, str, "part")
            else:
                validate_isinstance(
                    self.parts[task], str, f"parts[\"{task}\"]")

            # -> Shuffling
            if task not in self.shuffle_task_data:
                if self.shuffle_data is None:
                    raise ValueError("TODO text")
                self.shuffle_task_data[task] = validate_isinstance(self.shuffle_data, bool, "suffle_data")
            else:
                validate_isinstance(
                    self.shuffle_task_data[task], bool, f"shuffle_task_data[\"{task}\"]")

        self.data_loaders = {}
        self.num_batches_per_task = {}

        for name, data in self.datasets.items():
            features = data.features(self.parts[name])
            self.data_loaders[name] = DataLoader(
                data=features,
                collate_fun=data.collator,
                batch_size=self.batch_sizes[name],
                shuffle=self.shuffle_task_data[name],
            )
            self.num_batches_per_task[name] = len(self.data_loaders[name])

    def __len__(self) -> int:
        """
        Number of batches to be sampled
        """
        return sum(self.num_batches_per_task.values())

    def __iter__(self) -> 'MultitaskBatchSampler':
        # Initializing sampling stuff
        total_num = len(self)
        task_names = []
        task_probs = []
        for name, num_batches in self.num_batches_per_task.items():
            task_names.append(name)
            task_probs.append(num_batches / total_num)
        task_probs = np.array(task_probs)
        num_tasks = len(task_probs)
        self.sampler = lambda: task_names[np.random.choice(num_tasks, p=task_probs)]

        # Initializing data loaders
        self.data_loaders = {name: iter(dl) for name, dl in self.data_loaders.items()}

        return self

    def __next__(self):
        stopped_names = set()
        all_names = set(self.datasets.keys())
        while stopped_names != all_names:
            name = self.sampler()
            try:
                data = next(self.data_loaders[name])
                return TaskBatch(
                    name=name,
                    data=data,
                )
            except StopIteration:
                stopped_names.add(name)
        raise StopIteration()


@dataclass
class MultitaskMetabatchSampler:
    datasets: Dict[str, Data]
    part: Optional[str] = None
    parts: Optional[Dict[str, str]] = None
    batch_size: Optional[int] = None
    batch_sizes: Optional[Dict[str, int]] = None
    shuffle_data: bool = True
    shuffle_task_data: Optional[Dict[str, bool]] = None
    stop_strategy: str = "longest"

    data_loaders: Dict[str, DataLoader] = field(init=False)
    num_batches_per_task: Dict[str, int] = field(init=False)
    target_task: str = field(init=False)

    def __post_init__(self) -> None:
        if self.batch_sizes is None:
            self.batch_sizes = {}
        if self.shuffle_task_data is None:
            self.shuffle_task_data = {}
        if self.parts is None:
            self.parts = {}

        # Validating & figuring out all params

        for task in self.datasets:
            # -> Batch sizes
            if task not in self.batch_sizes:
                if self.batch_size is None:
                    raise ValueError("Expected to have either batch size for all tasks in "
                                     "'batch_sizes' and 'batch_size': Optional[Any] or"
                                     "skipped some tasks in 'batch_sizes' with default in 'batch_size' or"
                                     "'batch_sizes' = None, 'batch_size': int "
                                     "for all tasks to have the same batch size")
                self.batch_sizes[task] = validate_isinstance(self.batch_size, int, "batch_size")
            else:
                validate_isinstance(
                    self.batch_sizes[task], int, f"self.batch_sizes[\"{task}\"]")

            # -> Parts
            if task not in self.parts:
                if self.part is None:
                    raise ValueError("TODO error text")
                self.parts[task] = validate_isinstance(self.part, str, "part")
            else:
                validate_isinstance(
                    self.parts[task], str, f"parts[\"{task}\"]")

            # -> Shuffling
            if task not in self.shuffle_task_data:
                if self.shuffle_data is None:
                    raise ValueError("TODO text")
                self.shuffle_task_data[task] = validate_isinstance(self.shuffle_data, bool, "suffle_data")
            else:
                validate_isinstance(
                    self.shuffle_task_data[task], bool, f"shuffle_task_data[\"{task}\"]")

        self.data_loaders = {}
        self.num_batches_per_task = {}

        for name, data in self.datasets.items():
            features = data.features(self.parts[name])
            self.data_loaders[name] = DataLoader(
                data=features,
                collate_fun=data.collator,
                batch_size=self.batch_sizes[name],
                shuffle=self.shuffle_task_data[name],
            )
            self.num_batches_per_task[name] = len(self.data_loaders[name])

        if self.stop_strategy in ("shortest", "longest"):
            names = list(self.data_loaders.keys())
            lens = np.array(list(map(len, self.data_loaders.values())))
            idx = (np.argmin if self.stop_strategy == "shortest" else np.argmax)(lens)
            self.target_task = names[idx]
        elif self.stop_strategy in self.datasets:
            self.target_task = self.stop_strategy
        else:
            raise KeyError("Unexpected stopping strategy name 'stop_strategy', "
                           "must be either \"shortest\", or \"longest\", or a task name ")

    def __len__(self):
        return self.num_batches_per_task[self.target_task]

    def __iter__(self) -> 'MultitaskMetabatchSampler':
        self.data_loaders = {name: iter(dl) for name, dl in self.data_loaders.items()}
        return self

    def __next__(self):
        batch = {}
        for name, loader in self.data_loaders.items():
            try:
                batch[name] = next(loader)
            except StopIteration:
                if name == self.target_task:
                    raise StopIteration()
                else:
                    self.data_loaders[name] = iter(self.data_loaders[name])
                    batch[name] = next(loader)
        return batch
