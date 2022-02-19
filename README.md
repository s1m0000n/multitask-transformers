# Multitask Transformers

![GitHub Workflow Status](https://img.shields.io/github/workflow/status/s1m0000n/multitask-transformers/Pylint?label=pylint)
[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FCNOK7t3n39fqMF7vZNrU_2XVykEkwTz?usp=sharing)

4th year research project at MSU about multitask Transformer-based language models

## Roadmap

### Research

- Multitask learning on NLU tasks for Transformers
- Base models and heads aren't enough? More levels could be useful?
  - For sentence / paragraphs special layers between heads & base model?
  - Question answering base head with subheads
- Synthetic tasks for representation learning
- Efficient learning with emphasis on quality & generalization
- Handling data deficiency via augmenting / generating datasets
- Testing different multitask learning approaches
- Adaptation of techniques for Russian

### Code

- Language-independent code based around Hugging Face ecosystem
- Further development of Multitask Trainer
  - Right now: training with basic algorithm
  - Planned: 
    - Multiple strategies / user-defined training strategies (like inherited class with override methods)
    - Better evaluation implementation
- Automated generation of inference API (using FastAPI)
