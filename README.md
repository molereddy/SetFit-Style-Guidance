# Classifier-guided style constrained generation via SetFit.

## COMPSCI 685 course project 


We follow [FUDGE](https://arxiv.org/pdf/2104.05218)'s future classifier guided generator framework and optimize it with a [SetFit](https://arxiv.org/pdf/2209.11055)-based classifier for label-efficiency.

This repository implements the main method, baselines and evaluation. Implementation of token-level setfit has been done in [https://github.com/molereddy/token-level-setfit](https://github.com/molereddy/token-level-setfit) fork of SetFit.

Install
```
pip install -r requirements.txt
pip install git+https://github.com/molereddy/token-level-setfit.git
```

**Classify** 

in the `classifiers/` directory
`train_fudge_classifier.py` trains an encoder-decoder T5-based formality classifier model on the [GYAFC dataset](https://arxiv.org/abs/1803.06535) as in FUDGE.

`train_setfit_classifier.py` (our method) trains the T5 based model on a fewshot example set from [GYAFC](https://arxiv.org/abs/1803.06535), to optimize the FUDGE classifier training efficiency.
```
# To train the setfit classifier for guidance, run the file as shown below (each file must have example sentences in each line)
# -p base_path
# -f formal_data_file
# -if informal_data_file
# -n number_of_fewshot_examples_per_class

python train_setfit_classifier.py \
    -p "gyafc_pilot" \
    -f "shuffled_gyafc50_0.txt" \
    -if "shuffled_gyafc50_1.txt" \
    -n 40
```

**Generate**

`generate.py` uses a T5 generator ([humarin paraphraser version](https://huggingface.co/humarin/chatgpt_paraphraser_on_T5_base)) and a classifier from above to guide the generation, to generate styled paraphrases on the given datasets.
```
# To generate the sentences using our architecture, run the file as shown below
python generate.py \
    --generation_model_name "humarin" \
    --classifier_model_name "setfit_gyafc_partial_40" \
    --dataset_name "daily_dialog" \
    --samples 500 \
    --batch_size 32 \
    --results_dir "generation_results"
```

**Evaluate** 

in the `eval/` directory
`bert_style_reg_train.py` trains a BERT-based formality regression model on the Pavlick formality dataset. `bert_style_reg_eval.py` uses this trained model to evaluates our generations. Its classifier evaluation counterpart, `deberta_style_class_eval.py`, loads a [model](https://huggingface.co/s-nlp/deberta-large-formality-ranker) trained on [GYAFC dataset](https://arxiv.org/abs/1803.06535) using DeBERTa architecture and evaluates our generations.
`roberta_nli_eval.py` evaluates Paraphrasing quality is evaluated with a RoBERTa based [paraphrase ranker](https://huggingface.co/cross-encoder/nli-roberta-base) from SentenceTransformers.
```
# To evaluate the formality of the generated sentences (in summary.csv), 
python deberta_style_class_eval.py \
    -fp "generation_results/setfit_gyafc_partial_40/daily_dialog/summary.csv"
python bert_style_reg_eval.py \
    -fp "generation_results/setfit_gyafc_partial_40/daily_dialog/summary.csv"

# To evaluate the content preservation of the generated sentences (in summary.csv),
python roberta_nli_eval.py \
    -fp "generation_results/setfit_gyafc_partial_40/daily_dialog/summary.csv"
```

**Datasets**

Daily Dialog (conversational dataset) loaded [from here](https://huggingface.co/datasets/daily_dialog).
GYAFC style dataset is not publicly available, please contact the authors.
Pavlick formality dataset (aka SQUINKY) loaded [from here](https://huggingface.co/datasets/osyvokon/pavlick-formality-scores).
PAWS (paraphrasing dataset) loaded [from here](https://huggingface.co/datasets/paws-x).
