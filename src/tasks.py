"""
Containers and methods for task
"""
import numpy as np
from transformers.tokenization_utils import TextInput, TextInputPair, \
    PreTokenizedInput, PreTokenizedInputPair, EncodedInput, EncodedInputPair
from datasets import Dataset, DatasetDict, load_metric
from pydantic import BaseModel, validator
from transformers import EvalPrediction, BatchEncoding, PretrainedConfig, AutoConfig, \
    AutoModelForSequenceClassification, AutoTokenizer
from typing import Any, Type, TypeVar, Dict, Callable, List, Union, Iterable, Optional, Tuple

from .features import ClassificationConverter
from .models import MultitaskModel
from .utils import dmap, itercat

from abc import ABC, abstractmethod

T = TypeVar('T')

Batch = Dict[str, Union[
    List[TextInput], List[TextInputPair],
    List[PreTokenizedInput], List[PreTokenizedInputPair],
    List[EncodedInput], List[EncodedInputPair],
]]

ComputeMetrics = Callable[[EvalPrediction], Dict[str, float]]


class ConfiguredTask(BaseModel):
    """
    Task with config initialized from pretrained model
    """
    name: str
    cls: Any
    config: Any
    data: DatasetDict
    converter: Callable[[Batch], BatchEncoding]
    compute_metrics: Optional[ComputeMetrics] = None

    @validator('cls')
    def cls_validate(cls, value: T) -> T:
        from_pretrained = getattr(value, "from_pretrained", None)
        if (not from_pretrained) or (not callable(from_pretrained)):
            raise TypeError("Wrong type for 'cls', must have a callable method from_pretrained")
        return value

    @validator('config')
    def config_validate(cls, value: T) -> T:
        if issubclass(type(value), PretrainedConfig):
            return value
        raise TypeError("Wrong type for 'config', must be subclass of transformers.PretrainedConfig")


class IncompleteTask(ABC):
    """
    Abstract class for incomplete tasks, which are further configured
    with the implemented in each subclass .configure() method
    """

    @abstractmethod
    def configure(self, model_name: str) -> ConfiguredTask:
        pass


class Task(IncompleteTask):
    """
    Basic class for representing tasks with undefined shared encoder model with config as dict
    """

    def __init__(self, name: str, cls: Any, data: DatasetDict,
                 converter: Callable[[Batch], BatchEncoding],
                 config: Dict[str, Any] = None,
                 compute_metrics: Optional[ComputeMetrics] = None) -> None:
        if config is None:
            config = {}
        self.name = name
        from_pretrained = getattr(cls, "from_pretrained", None)
        if (not from_pretrained) or (not callable(from_pretrained)):
            raise TypeError("Wrong type for 'cls', must have a callable method from_pretrained")
        self.cls = cls
        self.config = config
        self.data = data
        self.converter = converter
        self.compute_metrics = compute_metrics

    def configure(self, model_path: str) -> ConfiguredTask:
        """
        Casts Task to ConfiguredTask by using encoder model_path to create a transformers config
        :param model_path: Model path / name in Hugging Face Transformers module
        :return: Ready to use ConfiguredTask
        """
        return ConfiguredTask(
            name=self.name,
            cls=self.cls,
            config=AutoConfig.from_pretrained(model_path, **self.config),
            data=self.data,
            converter=self.converter,
            compute_metrics=self.compute_metrics
        )


class ClassificationMetrics:
    def __init__(self, metrics: Iterable[str] = ("accuracy", "precision", "recall", "f1")):
        self.metrics = [load_metric(metric) for metric in metrics]

    def __call__(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        result = {}
        for metric in self.metrics:
            result.update(metric.compute(predictions, labels))
        return result


class ClassificationTask(IncompleteTask):
    """
    Class for solving basic classification tasks with preconfigured converter to features & model head:
        (Encoder model) -> nn.Dropout -> nn.Linear(hidden_size, num_labels) -> cases {
            num_labels == 1 => MSELoss                                 # Reduces to regression

            num_labels > 1 and labels: int | float => CrossEntropyLoss # Single label classification

            else => BCEWithLogitsLoss                                  # Multi label classification
        }
    """

    def __init__(self, name: str, data: DatasetDict,
                 num_labels: int = 2,
                 input_field: str = "input",
                 label_field: str = "label",
                 preprocessor: Optional[Callable] = None,
                 config: Optional[Dict[str, Any]] = None,
                 tokenizer_args: Optional[Tuple] = None,
                 tokenizer_kwargs: Optional[Dict[str, Any]] = None,
                 compute_metrics: Optional[ComputeMetrics] = ClassificationMetrics()) -> None:
        self.name = name
        self.data = data if preprocessor is None else dmap(preprocessor, data)
        self.config = config if config else {}
        if "num_labels" not in self.config:
            self.config["num_labels"] = num_labels
        self.input_field = input_field
        self.label_field = label_field
        self.tokenizer_args = tokenizer_args if tokenizer_args else ()
        self.tokenizer_kwargs = tokenizer_kwargs if tokenizer_kwargs else {
            "padding": True,
            "truncation": True,
        }
        self.compute_metrics = compute_metrics

    def configure(self, model_path: str) -> ConfiguredTask:
        """
        Casts ClassificationTask to ConfiguredTask

        :param model_path: Model path / name in Hugging Face Transformers module
        :return: Configured task using 'model_path'
        """
        return Task(
            name=self.name,
            cls=AutoModelForSequenceClassification,
            config=self.config,
            data=self.data,
            converter=ClassificationConverter(
                AutoTokenizer.from_pretrained(model_path),
                *self.tokenizer_args,
                input_field=self.input_field,
                label_field=self.label_field,
                **self.tokenizer_kwargs
            ),
            compute_metrics=self.compute_metrics
        ).configure(model_path)


AnyTask = Union[ConfiguredTask, Type[IncompleteTask]]
AnyTask.__doc__ = "Any Task classes instance"


def resolve_task(model_path: str, task: AnyTask) -> ConfiguredTask:
    """
    Resolve any kind of Task to Configured task
    :param model_path: Model path / name in Hugging Face Transformers module
    :param task: Any Task classes instance, not depending on config: Task or ConfiguredTask
    :return: ConfiguredTask instance or fails with TypeError
    """
    if isinstance(task, ConfiguredTask):
        return task
    elif issubclass(type(task), IncompleteTask):
        return task.configure(model_path)
    else:
        raise TypeError(f"Wrong type {type(task)} for 'task' argument, must be an instance of {AnyTask}")


class Tasks:
    """
    Gathers all tasks sharing the same base encoder
    """

    def __init__(self, model_path: str,
                 task: Optional[Union[AnyTask, Iterable[AnyTask]]] = None,
                 *tasks: AnyTask) -> None:
        self.encoder_path = model_path
        self.tasks = {}
        if task is not None:
            for task in itercat(task, tasks):
                resolved = resolve_task(self.encoder_path, task)
                self.tasks[resolved.name] = resolved

    def __getitem__(self, item: str) -> Optional[ConfiguredTask]:
        return self.tasks[item] if item in self.tasks else None

    def add(self, task: AnyTask, raising: bool = True) -> 'Tasks':
        if task.name in self.tasks:
            if raising:
                raise IndexError(f"Task with name {task.name} is already in Tasks")
        else:
            self.tasks[task.name] = resolve_task(self.encoder_path, task)
        return self

    def replace(self, task: AnyTask, raising: bool = True) -> 'Tasks':
        if task.name not in self.tasks:
            if raising:
                raise IndexError(f"Task with name {task.name} is not in Tasks, nothing to replace")
        else:
            self.tasks[task.name] = resolve_task(self.encoder_path, task)
        return self

    def update(self, tasks: Union[AnyTask, Iterable[AnyTask]]) -> 'Tasks':
        if isinstance(tasks, Iterable):
            for t in tasks:
                self.tasks[t.name] = resolve_task(self.encoder_path, t)
            return self
        else:
            self.update([tasks])

    def make_packed_features(
            self,
            columns: Iterable[str] = ("input_ids", "attention_mask", "labels"),
            map_kwargs: Optional[Dict[str, Any]] = None,
            task_map_kwargs: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Dict[str, Dataset]]:
        """
        Creates features for all tasks from data field (after preprocessing)

        :param columns: Columns to format in the output to torch type (columns used further for training)
        :param map_kwargs: General arguments for all tasks in datasets.Dataset.map call (default: batched=True)
        :param task_map_kwargs: Task specific arguments in datasets.Dataset.map call (higher priority then general)
        :return: Dictionary of featurized data in format task_name: str -> part_name: str -> dataset: datasets.Dataset
        """

        features: Dict[str, Dict[str, Dataset]] = {}
        map_kwargs = {} if map_kwargs is None else map_kwargs
        task_map_kwargs = {} if task_map_kwargs is None else task_map_kwargs

        if "batched" not in map_kwargs:
            map_kwargs["batched"] = True

        for name, task in self.tasks.items():
            features[name] = {}
            if name in task_map_kwargs:
                final_map_kwargs = map_kwargs.update(task_map_kwargs)
            else:
                final_map_kwargs = map_kwargs
            for split_name, split_dataset in task.data.items():
                features[name][split_name] = split_dataset.map(task.converter, **final_map_kwargs)
                features[name][split_name].set_format(type="torch", columns=list(columns))
        return features

    def make_features(
            self, splits: Union[List[str], Tuple[str]] = ("train", "validation", "test"),
            columns: Iterable[str] = ("input_ids", "attention_mask", "labels"),
            map_kwargs: Optional[Dict[str, Any]] = None,
            task_map_kwargs: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Union[Tuple[Dict[str, Dataset], ...], Dict[str, Dataset]]:
        """
        Creates features for all tasks from data field (after preprocessing) in the format used by MultitaskTrainer

        :param splits: Names of the splits to be returned (order is important, so use ordered data structures like list)
        :param columns: Columns to format in the output to torch type (columns used further for training)
        :param map_kwargs: General arguments for all tasks in datasets.Dataset.map call (default: batched=True)
        :param task_map_kwargs: Task specific arguments in datasets.Dataset.map call (higher priority then general)
        :return: If 1 split is specified => part_name: str -> dataset: datasets.Dataset,
            for multiple splits - tuple with the same representations as single,
            but for each split in the order specified in 'splits'
        """
        packed = self.make_packed_features(columns=columns, map_kwargs=map_kwargs, task_map_kwargs=task_map_kwargs)
        result = []
        for split_name in splits:
            result.append({task_name: dataset[split_name] for task_name, dataset in packed.items()})
        return tuple(result) if len(result) > 1 else result[0]

    def make_model(self, cls: Type[T] = MultitaskModel) -> T:
        """
        Make a multitask model, with forward taking multiple tasks into account - switching heads

        :param cls: Class with method .create(encoder_path: str, tasks: Mapping[str, ConfiguredTask])
        :return: Initialized model class with .forward(task_name: str, ...)
        """
        return cls.create(self.encoder_path, self.tasks)

    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """
        Compute metrics for all tasks and join all metrics to single dictionary with suffixes - task-names

        :param eval_pred: Evaluation output: predictions, label_ids
        :return: Joined dictionary for all tasks "<metric_name> (<task_name>)" -> value: float
        """
        result = {}
        for task_name, task in self.tasks.items():
            if task.compute_metrics is not None:
                for metric_name, metric_value in task.compute_metrics(eval_pred).items():
                    result[metric_name + f" ({task_name})"] = metric_value
        return result

    def make_compute_metrics(self) -> ComputeMetrics:
        return self.compute_metrics
