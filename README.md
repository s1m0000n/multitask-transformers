# Multitask Transformers

[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FCNOK7t3n39fqMF7vZNrU_2XVykEkwTz?usp=sharing)
![GitHub Workflow Status: pylint](https://img.shields.io/github/workflow/status/s1m0000n/multitask-transformers/Pylint?label=pylint)

Open research project about multitask Transformer-based language models and my 4th year graduation paper

Anyone interested in joining the research & contributing to the project is welcome! It's planned to release a paper on the results of the research (alongside graduation project), so I'd be happy to have coauthors. If you are interested, please contact me by email pogorelcevsa@gmail.com or in Telegram https://t.me/s1m00n 

## 🦄 Motivational example

Now training multitask BERT-like models is as easy as training a regular NN

Example for 2 binary classification tasks from [Russian SuperGLUE](https://russiansuperglue.com)

```python
rsg = "russian_super_glue"
cfg = TokenizerConfig(max_length=512)
encoder_path = "DeepPavlov/rubert-base-cased"
tasks = Tasks([
  ClassificationTask(
    name="danetqa",
    dataset_dict=load_dataset(rsg, "danetqa"),
    preprocessor=Preprocessor([preprocess_danetqa]),
    tokenizer_config=cfg,
  ),
  ClassificationTask(
    name="terra",
    dataset_dict=load_dataset(rsg, "terra"),
    preprocessor=Preprocessor([NLIPreprocess()]),
    tokenizer_config=cfg,
  )
], encoder_path)
```

Preparing for training:
- Sampler makes mixed tasks batches with collated data, forward method & metrics for the specific task in batch
    ```python
    train_sampler = MultitaskBatchSampler(tasks.data, "train", batch_size=12)
    ```
- Creating a multitask model
    ```python
    model = MultitaskModel(encoder_path, tasks.heads)
    ```
- Default stuff for training with PyTorch
    ```python
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 1
    num_training_steps = num_epochs * len(train_sampler)
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

for epoch_num in range(num_epochs):
  for batch in train_sampler:
    batch.data.to(device)
    outputs = model.forward(batch.name, **batch.data)
    loss = outputs.loss
    print(f"Training loss: {loss} on task {batch.name}")
    loss.backward()

    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    progress_bar.update(1)
```

*: Imports before all other example code

```python
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AdamW, get_scheduler

from src.classification import ClassificationTask
from src.dataloaders import MultitaskBatchSampler
from src.models import MultitaskModel
from src.preprocessing import Preprocessor, NLIPreprocess
from src.ru import preprocess_danetqa
from src.tasks import Tasks
from src.tokenizers import TokenizerConfig
```

## 🎯 Goals

This project is all about NLP, particularly Transformers, especially BERT-like models for English & Russian (highest priority) and other languages (all code is build with the idea that it could be used for any language in mind, expect for the `ru`, `en`(planned) modules, for some utils for the corresponding languages' tasks)

1. Developing a nice thing to use when you are about to do a bit multitasking
2. Studying multitask learning for English & Russian
3. Multitask learning as a way of "helping" tasks with small datasets
4. Solving optimization challenges & negative transfer for such complicated models
5. Studying perspective head architectures for complex language understanding tasks
6. Learn & become stronger in Deep Learning, NLP, Multitask Learning (personal)
7. Just have some fun


## 📒 Notebooks

### Colab

With pretrained model [DeepPavlov RuBERT base cased](https://huggingface.co/DeepPavlov/rubert-base-cased):

1. [Experiments with regular training](https://colab.research.google.com/drive/1FCNOK7t3n39fqMF7vZNrU_2XVykEkwTz?usp=sharing)
2. [Experiments with sum loss training](https://colab.research.google.com/drive/1q0Ob1eOmQSaja2cHWFoPwN28dO0id38K?usp=sharing)
3. [Experiments with sum loss & custom heads](https://colab.research.google.com/drive/1UBPOriaxwOZf44kCArKmammOMNLrupOo?usp=sharing)

With other pretrained Transformers for Russian:

TODO

#### Single task baselines (in progress)

For classification tasks - `AutoModelForSequenceClassification`

- Pretrained model: [DeepPavlov RuBERT base cased](https://huggingface.co/DeepPavlov/rubert-base-cased)
- `transformers.AutoModelForClassification` head for classification tasks
- Everything to defaults / standart, popular for the task
- No special tricks - base solutions
- Goal: have some ordinary scores to compare with (check whether multitasking helps / negative transfer)

1. [Russian SuperGLUE DaNetQA](https://colab.research.google.com/drive/1NTNKRDOzJrfgp_EqgnNllZ1PTBpJ2dDa?usp=sharing)
2. [Russian SuperGLUE TERRa](https://colab.research.google.com/drive/1O1VAgRVC4z8H_oI-YSdoWJKUg4gJZPW4?usp=sharing)
3. [Twitter Sentiment](https://colab.research.google.com/drive/1Wqb2ERqYXnZ68xSDf5nttfvP7JxRtMG5?usp=sharing)
4. [XNLI](https://colab.research.google.com/drive/1z0WBHyakEUupvDgXWoPc1JKYLUBmQNzc?usp=sharing)
5. [Headline Cause](https://colab.research.google.com/drive/1H5pJQwEFMer9HpkJZ7mFNV8Wxj3sEbKu?usp=sharing)

## Figured out stuff so far

- The code is working (most of it, in most cases)
- Converges with training on separate losses, converges better with summed loss
- Excellent (near or SOTA) results are achievable
- Tricks and unobvous stuff is required for this Frankenstein to learn something at a decent degree
- More complicated heads are working as expected - `classification.NFeedForwardsHead` is great, so interested in what heads with attention and more feed-forwards could do
- I started with a bunch of really complicated tasks from (Russian) SuperGLUE, which actually require complex heads

## ⚡️ Supported Tasks

### ✅ Batteries included

#### Russian SuperGLUE

The plans have changes, so right now Russian SuperGLUE tasks are not top priority

Current research focused more on other tasks with more usable data

DaNetQA solution is planned to be developed further, as a goal of trying to use multitask learning for tasks with smaller datasets

- [DaNetQA](https://russiansuperglue.com/tasks/task_info/DaNetQA)
- [PARus](https://russiansuperglue.com/tasks/task_info/PARus)
- [TERRa](https://russiansuperglue.com/tasks/task_info/TERRa)

#### Corus datasets

Now `ru.py` is massively extended with support for loading datasets using
[corus package of Natasha project](https://github.com/natasha/corus). `corus` implements some
loaders for a lot of popular russian datasets, while being a little inconvinient
to use, because you have to manually download datasets & load them using different funcs. In `ru.py` were implemented data classes to support this features of
`corus` & adapt them to use with `datasets` library. Also, automatic downloading
is also here along with a dataset splitter & declarative configurations (which we all like, I hope).

#### Other tasks for Russian

- [Russian part of XNLI](https://github.com/facebookresearch/XNLI)
- [Twitter sentiment classification](https://study.mokoron.com/)
- [Headline Cause](https://huggingface.co/datasets/IlyaGusev/headline_cause)

**Description**

Tasks which have everything necessary to start training with the following "batteries" included:

- Loading to `Dataset` / `DatasetDict`
- Preprocessing utils
- Support & tested head from Hugging Face Transformers
- (Optional) Predefined `nn.Module` heads
- **Automated Task Class** which casts to `Task` using `to_task()` method
- Proven convergence, ability to achieve decent results

### 📦 BYO Data

- NLI tasks (`NLIPreprocessor` is quite generalizable with configurable field names)
- Sequence classification tasks (no obvious way to generalize preprocessing (for now)) & the ones that can be solved as classification
- Regression tasks & the ones that can be solved as regression

**Description**

Bring your own data 😂

Tasks, which are supported on model level, but do not have implemented preprocessing pipelines and not tested

- Support & tested head from Hugging Face Transformers
- (Optional) Predefined `nn.Module` heads
- **Automated Task Class** which casts to `Task` using `to_task()` method


### 🤷 Not going to implement in near future

- Generational tasks (not making yet another super duper GPT-100500)
- Tasks for more languages (concentrating on Russian and English)
- Specific tasks, which are harder to implement & not (expected for) giving much profit in terms of studying generalization and other multitask-oriented stuff

**Description**

Some great stuff is just out of scope for the research I'm doing right now, sorry for that

For better or worse, these are some tasks, that I'm not currently interested in and not implementing myself, because of very limited time, lack of resources (all I have is a Colab and a laptop w/o CUDA) and some other reasons

## ✏️ Other materials

### Interesting stuff to read / implement

1. [An Overview of Multi-Task Learning in Deep Neural Networks](https://ruder.io/multi-task/)
2. [CS 330: Deep Multi-Task and Meta Learning](https://cs330.stanford.edu/)
3. [Yu et al. Gradient Surgery for Multi-Task Learning. 2020](https://arxiv.org/abs/2001.06782)
4. [Обучение универсальной модели NLU c BERT и Multitask Learning](https://habr.com/ru/company/sberdevices/blog/560748/) (a very inspiring (not very techical) article, which interested me and i decided to develop the ideas further in my research)
