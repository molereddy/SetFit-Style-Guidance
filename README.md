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

`bert_regressor_train.py` trains a BERT-based formality regression model on the [Pavlick formality dataset](https://huggingface.co/datasets/osyvokon/pavlick-formality-scores). `bert_regressor_eval.py` uses this trained model to evaluate the generations. Its classifier evaluation counterpart, `deberta_evaluator.py` is [model](https://huggingface.co/s-nlp/deberta-large-formality-ranker) trained on GYAFC based on the DeBERTa architecture. 

`gold_classifier.py` trains with and does sample evaluations for encoder-decoder T5-based formality classifier model on the [GYAFC dataset](https://arxiv.org/abs/1803.06535). GYAFC is not publicly available, please contact the authors.

`setftit_classifier.py` trains the T5 based model (fewshot) on [GYAFC dataset](https://arxiv.org/abs/1803.06535), which is used to guide our generative architecture. 

Paraphrasing quality is evaluated with a [RoBERTa based paraphrase ranker](https://huggingface.co/cross-encoder/nli-roberta-base) from the SentenceTransformers package, in `roberta_evaluator.py`.

`setfit_T5_fudge.py` consists of the T5 generator and a classifier for guidance, generates the style-guided sentences.


```
# To evaluate the formality of the generated sentences (in summary.csv), run the classifiers as shown below
# -fp file_path
python deberta_evaluator.py -fp "generation_results_V1/humarin/setfit_gyfac_partial_40/daily_dialog/summary.csv"
python bert_regressor_eval.py -fp "generation_results_V1/humarin/setfit_gyfac_partial_40/daily_dialog/summary.csv"

# To evaluate the content preservation of the generated sentences (in summary.csv), run the classifier as shown below
# -fp file_path
python gs_paraphrase_classifier.py -fp "generation_results/humarin/setfit_gyfac_partial_40/daily_dialog/summary.csv"

# To train the setfit classifier for guidance, run the file as shown below
# -p base_path -f formal_data_file -if informal_data_file -n number_of_fewshot_examples_per_class
python setfit_classifier.py -p "gyfac_pilot" -f "shuffled_gyfac50_0.txt" -if "shuffled_gyfac50_1.txt" -n 40

# To generate the sentences using our architecture, run the file as shown below
python fudge_T5_setfit.py \
    --generation_model_name "humarin" \
    --classifier_model_name "setfit_gyfac_partial_40" \
    --dataset_name "daily_dialog" \
    --samples 500 \
    --batch_size 32 \
    --results_dir "generation_results"

```
