"""
Helper functions for multitask learning on Russian SuperGLUE tasks
"""
import random
from typing import Dict, Any, Union, Tuple


def preprocess_danetqa(sample: Dict[str, Any]) -> Dict[str, str]:
    """
    Preprocess Russian SuperGLUE DaNetQA sample for binary classification
    :param sample: One dict-like sample
    :return: Format for sequence classification models ('text', 'label')
    """
    return {
        'text': f'{sample["passage"]} {sample["question"]}',
        'label': sample["label"]
    }

def preprocess_parus(sample: Dict[str, str]) -> Dict[str, Union[Tuple[str, str], str]]:
    """
    Preprocesses Russian SuperGLUE PARus sample for binary classification
    :param sample: One dict-like sample of RSG's PARus in Hugging Face Datasets
    :return: Format for sequence classification models, in this case 0th or 1st option is correct ("options")
    """
    questions = {
        'cause': (
            'По какой причине?',
            'Почему?',
            'Причина этому',
            'Это произошло, потому что',
        ),
        'effect': (
            'Что произошло в результате?',
            'Из-за этого',
            'Вследствие чего',
            'В итоге получилось, что'
        )
    }
    question = questions[sample["question"]][random.randint(0, 3)]
    return {
        "label": sample["label"],
        "options": tuple(f'{sample["premise"]} {question} {sample[f"choice{j}"]}' for j in (1, 2))
    }

class InputLabelConv:
    # TODO: typing
    # -> https://github.com/s1m0000n/multitask-transformers/issues/14
    """
    Converter for input, label paired datasets
    """

    def __init__(self, tokenizer, *tokenizer_args,
                 input_name: str = 'text', label_name: str = 'label', **tokenizer_kwargs) -> None:
        self.tokenizer = tokenizer
        self.tokenizer_args = tokenizer_args
        self.tokenizer_kwargs = tokenizer_kwargs
        self.input_name = input_name
        self.label_name = label_name

    def __call__(self, batch):
        """
        Featurizes a batch
        :param batch:
        :return:
        """
        features = self.tokenizer.batch_encode_plus(
            batch[self.input_name], *self.tokenizer_args, **self.tokenizer_kwargs)
        features["labels"] = batch[self.label_name]
        return features


class OptionsLabelConv:
    # TODO: typing
    # -> https://github.com/s1m0000n/multitask-transformers/issues/14
    """
    Converter for options, label paired datasets
    """

    pass

    # def __init__(self, tokenizer, *tokenizer_args,
    #              input_name: str = 'text', label_name: str = 'label', **tokenizer_kwargs) -> None:
    #     self.tokenizer = tokenizer
    #     self.tokenizer_args = tokenizer_args
    #     self.tokenizer_kwargs = tokenizer_kwargs
    #     self.input_name = input_name
    #     self.label_name = label_name
    #
    # def __call__(self, batch):
    #     features = self.tokenizer.batch_encode_plus(
    #         batch[self.input_name], *self.tokenizer_args, **self.tokenizer_kwargs)
    #     features["labels"] = batch[self.label_name]
    #     return features