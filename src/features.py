from typing import Union, List, Dict

import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.tokenization_utils_base import TextInput, TextInputPair, PreTokenizedInput, PreTokenizedInputPair, \
    EncodedInput, EncodedInputPair, BatchEncoding


class ClassificationConverter:
    """
    Converter to features for input, label paired classification datasets.
    Tokenizes the input using tokenizers & packs labels into torch.Tensor
    with renaming to dict with fields expected by models implementations in transformers
    """

    def __init__(self, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                 *tokenizer_args,
                 input_field: str = 'input',
                 label_field: str = 'label',
                 **tokenizer_kwargs) -> None:
        self.tokenizer = tokenizer
        self.tokenizer_args = tokenizer_args
        self.tokenizer_kwargs = tokenizer_kwargs
        self.input_field = input_field
        self.label_field = label_field

    def __call__(self, batch: Dict[str, Union[
        List[TextInput], List[TextInputPair],
        List[PreTokenizedInput], List[PreTokenizedInputPair],
        List[EncodedInput], List[EncodedInputPair],
    ]]) -> BatchEncoding:
        """
        Featurizes a batch of training samples: inputs are tokenized & labels prepared for torch
        :param batch: Mapping {inputs, labels} to corresponding values
        :return: Features in default format used by *BERT*-like models in transformers module
        """
        features = self.tokenizer.batch_encode_plus(
            batch[self.input_field],
            *self.tokenizer_args,
            **self.tokenizer_kwargs
        )
        features["labels"] = batch[self.label_field]
        return features
