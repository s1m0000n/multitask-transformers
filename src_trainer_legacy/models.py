from transformers import PreTrainedModel, PretrainedConfig
from torch import nn


class MultitaskModel(PreTrainedModel):
    """
    Multitask analogue to transformers *Model* classes
    """

    def __init__(self, encoder, task_models):
        super().__init__(PretrainedConfig())
        self.encoder = encoder
        self.task_models = nn.ModuleDict(task_models)

    @classmethod
    def create(cls, encoder_path: str, tasks):
        """
        Creating a MultitaskModel using the model class (Task.cls) and config (Task.config) from single-task models.
        Creating each single-task model, and having them share the same encoder transformer.
        :param encoder_path: name of encoder model, example: "bert-base-uncased"
        :param tasks: dictionary of _tasks with names
        :return: MultitaskModel with initialized shared encoder and task_models
        """
        shared_encoder = None
        task_models = {}
        for name, task in tasks.items():
            model = task.cls.from_pretrained(encoder_path, config=task.config)
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

    def forward(self, task_name: str, *args, **kwargs):
        """
        Make a forward pass through encoder and a task head

        :param task_name: name of the task
        :param args: *args passed to Model(encoder -> head)
        :param kwargs: **kwargs passed to Model(encoder -> head)
        :return: typical results of forward for the model as usual
        """
        return self.task_models[task_name](*args, **kwargs)
