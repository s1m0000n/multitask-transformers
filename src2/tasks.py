"""
Containers and methods for task
"""
from datasets import Dataset, DatasetDict
from pydantic import BaseModel, validator
from typing import Any, Type, TypeVar, Dict, Callable, List, Union, Iterable, Optional, Tuple

from transformers import PretrainedConfig, AutoConfig
from transformers.tokenization_utils_base import TextInputPair, TextInput, PreTokenizedInput, EncodedInput, \
    PreTokenizedInputPair, EncodedInputPair, BatchEncoding

from .models import MultitaskModel

T = TypeVar('T')

Batch = Dict[str, Union[
    List[TextInput], List[TextInputPair],
    List[PreTokenizedInput], List[PreTokenizedInputPair],
    List[EncodedInput], List[EncodedInputPair],
]]


class ConfiguredTask(BaseModel):
    """
    Task with config initialized from pretrained model
    """
    name: str
    cls: Any
    config: Any
    data: DatasetDict
    featurize: Callable[[Batch], BatchEncoding]

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


class Task(BaseModel):
    """
    Task with config as dict of params to be passed to transformers.AutoModel.from_pretrained
    """
    name: str
    cls: Any
    config: Dict[str, Any] = {}
    data: DatasetDict
    featurize: Callable[[Batch], BatchEncoding]

    @validator('cls')
    def cls_validate(cls, value: T) -> T:
        from_pretrained = getattr(value, "from_pretrained", None)
        if (not from_pretrained) or (not callable(from_pretrained)):
            raise TypeError("Wrong type for 'cls', must have a callable method from_pretrained")
        return value

    def configure(self, model_path: str) -> ConfiguredTask:
        args = self.dict()
        args["config"] = AutoConfig.from_pretrained(model_path, **self.config)
        return ConfiguredTask(**args)


AnyTask = Union[Task, ConfiguredTask]
AnyTask.__doc__ = "Any Task classes instance, not depending on config: Task or ConfiguredTask"


def resolve_task(model_path: str, task: AnyTask) -> ConfiguredTask:
    """
    Resolve any kind of Task to Configured task
    :param model_path: Model path to BERT-like LM in Huggingface Transformers ecosystem
    :param task: Any Task classes instance, not depending on config: Task or ConfiguredTask
    :return: ConfiguredTask instance or fails with TypeError
    """
    if isinstance(task, ConfiguredTask):
        return task
    elif isinstance(task, Task):
        return task.configure(model_path)
    else:
        raise TypeError(f"Wrong type {type(task)} for 'task' argument, must be an instance of {AnyTask}")


class Tasks:
    """
    Gathers all tasks sharing the same base encoder
    """

    def __init__(self, model_path: str, tasks: Optional[Iterable[AnyTask]] = None) -> None:
        self.encoder_path = model_path
        self.tasks = {}
        if tasks:
            for task in tasks:
                self.tasks[task.name] = resolve_task(self.encoder_path, task)

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

    def make_packed_features(self, *map_args,
                             columns: Iterable[str] = ("input_ids", "attention_mask", "labels"),
                             batched: bool = True, **map_kwargs) \
            -> Dict[str, Dict[str, Dataset]]:
        features: Dict[str, Dict[str, Dataset]] = {}
        for name, task in self.tasks.items():
            features[name] = {}
            for split_name, split_dataset in task.data.items():
                features[name][split_name] = split_dataset.map(task.featurize, *map_args, batched=batched, **map_kwargs)
                features[name][split_name].set_format(type="torch", columns=list(columns))
        return features

    def make_features(self, splits: Iterable[str], *map_args,
                      columns: Iterable[str] = ("input_ids", "attention_mask", "labels"),
                      batched: bool = True, **map_kwargs) \
            -> Union[Tuple[Dict[str, Dataset], ...], Dict[str, Dataset]]:
        packed = self.make_packed_features(*map_args, columns=columns, batched=batched, **map_kwargs)
        result = []
        for split_name in splits:
            result.append({task_name: dataset[split_name]
                           for task_name, dataset in packed.items()})
        return tuple(result) if len(result) > 1 else result[0]

    def make_model(self, cls: Type[T] = MultitaskModel) -> T:
        """
        Make a multitask model, with forward taking multiple tasks into account - switching heads
        :param cls: Class with method .create(encoder_path: str, tasks: Mapping[str, ConfiguredTask])
        :return: Initialized model class with .forward(task_name: str, ...)
        """
        return cls.create(self.encoder_path, self.tasks)
