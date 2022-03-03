from dataclasses import dataclass
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig
from typing import Any, Optional, Dict


@dataclass
class Head:
    class_: Any
    config_params: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        from_pretrained = getattr(self.class_, "from_pretrained", None)
        if (not from_pretrained) or (not callable(from_pretrained)):
            raise TypeError("Wrong type for 'cls', must have a callable method 'from_pretrained'")
        if self.config_params is None:
            self.config_params = {}

    def make_config(self, model_path: str):
        return AutoConfig.from_pretrained(model_path, **self.config_params)

    def make_model(self, model_path: str) -> PreTrainedModel:
        return self.class_.from_pretrained(model_path, config=self.make_config(model_path))


class MultitaskModel(PreTrainedModel):
    """
    Multitask analogue to transformers *Model* classes
    """

    def __init__(self, encoder, task_models):
        super().__init__(PretrainedConfig())
        self.encoder = encoder
        self.task_models = nn.ModuleDict(task_models)

    @classmethod
    def create(cls, model_path: str, heads: Dict[str, Head]):
        """
        Creating a MultitaskModel using the model class (Task.cls) and config (Task.config) from single-task models.
        Creating each single-task model, and having them share the same encoder transformer.
        :param model_path: name of encoder model, example: "bert-base-uncased"
        :param heads: dictionary of task names and corresponding heads
        :return: MultitaskModel with initialized shared encoder and task_models
        """
        shared_encoder = None
        task_models = {}
        for name, head in heads.items():
            model = head.make_model(model_path)
            if shared_encoder is None:
                shared_encoder = getattr(model, cls.get_encoder_attr_name(model))
            else:
                setattr(model, cls.get_encoder_attr_name(model), shared_encoder)
            task_models[name] = model
        return cls(encoder=shared_encoder, task_models=task_models)

    @classmethod
    def get_encoder_attr_name(cls, model):
        """
        The encoder transformer is named differently in each model "architecture".
        This method lets us get the name of the encoder attribute
        """
        name = model.__class__.__name__
        if name.startswith("Bert"):
            return "bert"
        if name.startswith("Roberta"):
            return "roberta"
        if name.startswith("Albert"):
            return "albert"
        raise NotImplementedError(f"Add support for new model {name}")

    def __getitem__(self, task_name: str):
        if task_name in self.task_models:
            return self.task_models[task_name]
        else:
            raise KeyError(f"No model found for {task_name}")

    def forward(self, task_name: str, *args, **kwargs):
        """
        Make a forward pass through encoder and a task head

        :param task_name: name of the task
        :param args: *args passed to Model(encoder -> head)
        :param kwargs: **kwargs passed to Model(encoder -> head)
        :return: typical results of forward for the model as usual
        """
        return self.task_models[task_name](*args, **kwargs)
