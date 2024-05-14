Code to train a T5-based formality classifier on the GYFAC dataset.
Part of a COMPSCI 685 course project on classifier-guided style constrained generation.


Environment setup must cover libraries like:
Need libraries: 
torch, numpy, pandas, sklearn, huggingface, huggingface datasets, transformers

Install our version of setfit that takes tokenized inputs.
```python
pip install git+https://github.com/molereddy/token-level-setfit.git
```

Install sentencepiece for the T5 tokenizer.
```python
pip install sentencepiece
```
