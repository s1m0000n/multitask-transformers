# Multitask Transformers

[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FCNOK7t3n39fqMF7vZNrU_2XVykEkwTz?usp=sharing)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/s1m0000n/multitask-transformers/Pylint?label=pylint)

Research project at MSU about multitask Transformer-based language models and my 4th year graduation paper

Anyone interested in joining the research & contributing to the project is welcome! It's planned to release a paper on the results of the research (alongside graduation project), so I'd be happy to have coauthors. If you are interested, please contact me by email pogorelcevsa@gmail.com or in Telegram https://t.me/s1m00n 

## Example

Now training multitask BERT-like models is as easy as training a regular NN

Some neccessary imports for 2 classification tasks from [Russian SuperGLUE](https://russiansuperglue.com)
```python
import torch # Your good old PyTorch
from tqdm.auto import tqdm
from transformers import AdamW, get_scheduler # Your good old training stuff
# Task's data is relying on Hugging Face Datasets tools and data structures
from datasets import load_dataset

from src.preprocessing import Preprocessor # Simple preprocessing pipeline builder
from src.ru import preprocess_danetqa, preprocess_parus # Preprocessing funcs
# Main classes, managing model heads, data, metrics etc.
from src.tasks import SequenceClassificationTask, Tasks
# Configuring Hugging Face Transformers tokenizer initialization & call params
from src.tokenizers import TokenizerConfig
```

Now let's configure tasks in a declarative way
```python
rsg = "russian_super_glue"
cfg = TokenizerConfig(max_length=512)
# Base encoder path / name in transformers model ecosystem
tasks = Tasks("DeepPavlov/rubert-base-cased", [
    SequenceClassificationTask( # Both of our tasks are sequence classification
        name="danetqa", # Task name, used in verbose prints, internally
        dataset_dict=load_dataset(rsg, "danetqa"), # Loading dataset dictionary
        # dataset_dict is something like 
        # {"train": Dataset(...), "validation": Dataset(...), "test": Dataset(...)}
        # Our preprocessing pipeling are single funcs in these cases
        preprocessor=Preprocessor([preprocess_danetqa]),
        # Can be omitted in most cases, but this model is stubborn 🙊:
        # DeepPavlov/rubert-base-cased not implements max input size
        tokenizer_config=cfg
    ),
    SequenceClassificationTask(
        name="parus",
        dataset_dict=load_dataset(rsg, "parus"),
        preprocessor=Preprocessor([preprocess_parus]),
        tokenizer_config=cfg
    )
])
```

Preparing for training:
- Sampler makes mixed tasks batches with collated data, forward method & metrics for the specific task in batch
    ```python
    sampler = tasks.make_batch_sampler("train", 12)
    # Sampler implements much more options like variable per task batch sizes etc.
    # In this example it just produces collated batches from "train" part 
    # with same batch size of 12
    ```
- Default stuff for training with PyTorch
    ```python
    optimizer = AdamW(tasks.model.parameters(), lr=5e-5)
    num_epochs = 1
    num_training_steps = num_epochs * len(sampler)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    ```
- Moving model (and all the heads) to device and train mode
    ```python
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tasks.model.to(device)
    tasks.model.train()
    ```

And finally training, in this case it's the easiest possible training loop
```python
progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_epochs):
    for task_batch in sampler:
        # Note, that task_batch contains everything needed for training on this batch,
        # so you don't have to handle switching heads, datasets, tasks etc. yourself!
        batch = task_batch.data_on_device(device)
        outputs = task_batch.model_head(**batch)
        loss = outputs.loss
        print(f"Training loss: {loss} on task {task_batch.name}")
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
```

## Notebooks

### Colab

1. Training on multiple tasks for Russian with `torch`-native training loop: https://colab.research.google.com/drive/1FCNOK7t3n39fqMF7vZNrU_2XVykEkwTz?usp=sharing


