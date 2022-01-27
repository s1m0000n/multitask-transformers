from typing import *


def preprocess_danetqa(sample: Dict[str, Any]) -> Dict[str, str]:
    return {
        'input': f'{sample["passage"]} {sample["question"]}',
        'label': sample["label"]
    }


class InputLabelConv:
    def __init__(self, tokenizer, input_name: str = 'input', label_name: str = 'label', *tokenizer_args, **tokenizer_kwargs) -> None:
        self.tokenizer = tokenizer
        self.tokenizer_args = tokenizer_args
        self.tokenizer_kwargs = tokenizer_kwargs
        self.input_name = input_name
        self.label_name = label_name

    def __call__(self, batch):
        features = self.tokenizer.batch_encode_plus(batch[self.input_name], *self.tokenizer_args, **self.tokenizer_kwargs)
        features["labels"] = batch[self.label_name]
        return features
