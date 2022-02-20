"""
Helper functions for multitask learning on Russian SuperGLUE tasks
"""
import random
from typing import Dict, Any, Union, Tuple, List
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.tokenization_utils_base import TextInput, TextInputPair, PreTokenizedInput, PreTokenizedInputPair, \
    EncodedInput, EncodedInputPair
from dataclasses import dataclass

parus_questions = {
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
        'В итоге получилось, следующее:'
    )
}


def preprocess_danetqa(sample: Dict[str, Any]) -> Dict[str, str]:
    """
    Preprocess Russian SuperGLUE DaNetQA sample for binary classification
    :param sample: One dict-like sample
    :return: Format for sequence classification models ('text', 'label')
    """
    passage = sample["passage"]
    question = sample["question"]
    label = sample["label"]
    return {"input": f"{passage} {question}", "label": label}


def preprocess_parus(sample: Dict[str, str]) -> Dict[str, Union[Tuple[str, str], str]]:
    """
    Preprocesses Russian SuperGLUE PARus sample for binary classification
    :param sample: One dict-like sample of RSG's PARus in Hugging Face Datasets
    :return: Format for sequence classification models, in this case 0th or 1st option is correct
    """
    question_number = random.randint(0, len(parus_questions) - 1)
    question = parus_questions[sample["question"]][question_number]
    inputs = tuple(f'{sample["premise"]} {question} {sample[f"choice{j}"]}' for j in (1, 2))
    return {"label": sample["label"], "input": inputs}


@dataclass
class NLIPreprocessor:
    """
    Preprocessing for NLI tasks with fields hypothesis, premise & label
    """
    hypothesis_field: str = "hypothesis"
    premise_field: str = "premise"
    label_field: str = "label"
    input_field: str = "input"
    label_field_in: str = "label"

    def __call__(self, sample: Dict[str, str]) -> Dict[str, Union[Tuple[str, str], str]]:
        hypothesis = sample[self.hypothesis_field]
        premise = sample[self.premise_field]
        label = sample[self.label_field_in]
        return {self.input_field: (hypothesis, premise), self.label_field: label}


class InputLabelConv:
    """
    Converter for input, label paired datasets
    """

    def __init__(self, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                 *tokenizer_args,
                 input_name: str = 'input',
                 label_name: str = 'label',
                 **tokenizer_kwargs) -> None:
        self.tokenizer = tokenizer
        self.tokenizer_args = tokenizer_args
        self.tokenizer_kwargs = tokenizer_kwargs
        self.input_name = input_name
        self.label_name = label_name

    def __call__(self, batch: Dict[str, Union[
        List[TextInput], List[TextInputPair],
        List[PreTokenizedInput], List[PreTokenizedInputPair],
        List[EncodedInput], List[EncodedInputPair],
    ]]):
        """
        Featurizes a batch of training samples: inputs are tokenized & labels prepared for torch
        :param batch: Mapping {inputs, labels} to corresponding values
        :return: features for Transformer-based LMs, like BERT
        """
        features = self.tokenizer.batch_encode_plus(
            batch[self.input_name], *self.tokenizer_args, **self.tokenizer_kwargs
        )
        features["labels"] = batch[self.label_name]
        return features
