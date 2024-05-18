# Classifier-guided style constrained generation via SetFit.

## COMPSCI 685 course project 


We follow [FUDGE](https://arxiv.org/pdf/2104.05218)'s future classifier guided generator framework and optimize it with a [SetFit](https://arxiv.org/pdf/2209.11055)-based classifier for label-efficiency.

This repository implements the main method, baselines and evaluation. Implementation of token-level setfit has been done in https://github.com/molereddy/token-level-setfit fork of SetFit.

Install
```
pip install -r requirements.txt
pip install git+https://github.com/molereddy/token-level-setfit.git
pip install sentence_transformers
```

`bert_regressor_train.py` trains a BERT-based formality regression model on the [Pavlick formality dataset](https://huggingface.co/datasets/osyvokon/pavlick-formality-scores). `bert_regressor_eval.py` uses this trained model to evaluate the generations. Its classifier evaluation counterpart is [model](https://huggingface.co/s-nlp/deberta-large-formality-ranker) trained on GYAFC based on the DeBERTa architecture.

`gold_classifier.py` trains with and does sample evaluations for encoder-decoder T5-based formality classifier model on the [GYAFC dataset](https://arxiv.org/abs/1803.06535). GYAFC is not publicly available, please contact the authors.

`setftit_classifier.py` trains the T5 based model (fewshot) on [GYAFC dataset](https://arxiv.org/abs/1803.06535), which is used to guide our generative architecture. 

Paraphrasing quality is evaluated with a [RoBERTa based paraphrase ranker](https://huggingface.co/cross-encoder/nli-roberta-base) from the SentenceTransformers package.

`setfit_T5_fudge.py` consists of the T5 generator and a classifier for guidance, generates the style-guided sentences.
