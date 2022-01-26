task = {
    "cls": "cls is expected to have \"from_pretrained\" method (example: "
           "tranformer.AutoModelForSequenceClassification class, not instance!)",
    "config": "config is expected to be a subclass of PretrainedConfig (example: AutoConfig.from_pretrained("
              "\"bert-base-uncased\", num_labels=2)) ",
    "converter": "converter_to_features is expected to be a callable, with Iterable[Any] arg representing batch and "
                 "returning features: UserDict | transformers.BatchEncoding to be used with forward method of "
                 "transformers",
    "data": "TODO"
}
