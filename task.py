from collections import UserDict
from typing import *
from utils import method, subcls, fassert
from transformers import PretrainedConfig
import msg
import datasets


class Task:
    """
    Validates and stores tasks data, config, etc.
    """
    def __init__(self, cls: Any, config: Type[PretrainedConfig],
                 converter_to_features: Callable[[Iterable[Any]], Type[UserDict]],
                 data: datasets.DatasetDict, name: Optional[str] = None) -> None:
        """
        Checking task fields for compatibility with other components of multitask learner
        :param cls:
        :param config:
        :param converter_to_features:
        :param data:
        :param name:
        """
        self.cls = fassert(method("from_pretrained") @ cls, cls, msg.task["cls"])
        self.config = fassert(subcls(config) @ PretrainedConfig, config, msg.task["config"])
        self.converter = fassert(callable(converter_to_features), converter_to_features, msg.task["converter"])
        self.data = fassert(isinstance(data, datasets.DatasetDict), data, msg.task["data"])
        self.name = name or "Untitled"

