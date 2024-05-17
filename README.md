COMPSCI 685 course project on classifier-guided style constrained generation.
We follow [FUDGE](https://arxiv.org/pdf/2104.05218)'s future classifier guided generator framework and optimize it with a [Setfit](https://arxiv.org/pdf/2209.11055)-based classifier for label-efficiency.

This repository implements the main method, baselines and evaluation. Implementation of token-level setfit has been done in https://github.com/molereddy/token-level-setfit fork of SetFit.

Install
```
pip install -r requirements.txt
pip install git+https://github.com/molereddy/token-level-setfit.git
```
