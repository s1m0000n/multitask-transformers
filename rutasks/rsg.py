from typing import *


def preprocess_danetqa(sample: Dict[str, Any]) -> Dict[str, str]:
    return {
        'input': f'{sample["passage"]} {sample["question"]}',
        'label': sample["label"]
    }


class InputLabelConv:
    def __init__(self, tokenizer, *tokenizer_args, **tokenizer_kwargs) -> None:
        self.tokenizer = tokenizer
        self.tokenizer_args = tokenizer_args
        self.tokenizer_kwargs = tokenizer_kwargs

    def __call__(self, batch):
        features = self.tokenizer.batch_encode_plus(batch['input'], *self.tokenizer_args, **self.tokenizer_kwargs)
        features["labels"] = batch["label"]
        return features
