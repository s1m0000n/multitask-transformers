# Multitask Transformers

[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FCNOK7t3n39fqMF7vZNrU_2XVykEkwTz?usp=sharing)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/s1m0000n/multitask-transformers/Pylint?label=pylint)

My 4th year research project at MSU about multitask Transformer-based language models

Anyone interested in joining the research & contributing to the project is welcome! It's planned to release a paper on the results of the research (alongside graduation project), so I'd be happy to have coauthors. If you are interested, please contact me by email pogorelcevsa@gmail.com or in Telegram https://t.me/s1m00n 

## Notebooks

### Stable

New stuff gets here after experiments and tests (working, positive changes) with a bit polished code, using code from the master branch

Link(same as in the shield): https://colab.research.google.com/drive/1FCNOK7t3n39fqMF7vZNrU_2XVykEkwTz?usp=sharing

### Experimental

Active research here with possibly awful code / failing to work, implementing new tasks etc. , using code from the experimental branch

Link: https://colab.research.google.com/drive/1-OwwTMixRXMuOBvEH1O17L-vb5rwc__F?usp=sharing

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
