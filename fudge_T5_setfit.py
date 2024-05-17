
import os
import time
import torch
import argparse
import pandas as pd
from math import ceil
from tqdm import tqdm
from setfit import SetFitModel
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import LogitsProcessor,LogitsProcessorList

def get_generation_model(model_name):
    if model_name == "humarin":    
        model_id = "humarin/chatgpt_paraphraser_on_T5_base"
        # Load Generative model
        tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base", use_fast = True)
        model = T5ForConditionalGeneration.from_pretrained(model_id)
    else:
        raise NotImplementedError
    return model, tokenizer


def get_classifier_model(model_name):
    if model_name == "setfit_gyfac":
        classifier_model_path = "./model_checkpoints/setfit/gyfac_finetuned"
        classifier_model = SetFitModel.from_pretrained(classifier_model_path)
    elif model_name == "setfit_gyfac_5":
        classifier_model_path = "./model_checkpoints/setfit/gyfac_finetuned_5"
        classifier_model = SetFitModel.from_pretrained(classifier_model_path)
    elif model_name == "setfit_gyfac_10":
        classifier_model_path = "./model_checkpoints/setfit/gyfac_finetuned_10"
        classifier_model = SetFitModel.from_pretrained(classifier_model_path)
    elif model_name == "setfit_gyfac_40":
        classifier_model_path = "./model_checkpoints/setfit/gyfac_finetuned_40"
        classifier_model = SetFitModel.from_pretrained(classifier_model_path)
    elif model_name == "setfit_gyfac_5_v2":
        classifier_model_path = "./model_checkpoints/setfit/gyfac_finetuned_5_v2"
        classifier_model = SetFitModel.from_pretrained(classifier_model_path)
    elif model_name == "setfit_gyfac_10_v2":
        classifier_model_path = "./model_checkpoints/setfit/gyfac_finetuned_10_v2"
        classifier_model = SetFitModel.from_pretrained(classifier_model_path)
    elif model_name == "setfit_gyfac_20_v2":
        classifier_model_path = "./model_checkpoints/setfit/gyfac_finetuned_20_v2"
        classifier_model = SetFitModel.from_pretrained(classifier_model_path)
    elif model_name == "setfit_gyfac_40_withpartial":
        classifier_model_path = "./model_checkpoints/setfit/gyfac_finetuned_withpartial_40"
        classifier_model = SetFitModel.from_pretrained(classifier_model_path)
    elif model_name == "setfit_gyfac_partial_40":
        classifier_model_path = "./model_checkpoints/setfit/gyfac_finetuned_partial_40"
        classifier_model = SetFitModel.from_pretrained(classifier_model_path)
    elif model_name == "setfit_gyfac_partial_20":
        classifier_model_path = "./model_checkpoints/setfit/gyfac_finetuned_partial_20"
        classifier_model = SetFitModel.from_pretrained(classifier_model_path)
    elif model_name == "setfit_gyfac_partial_10":
        classifier_model_path = "./model_checkpoints/setfit/gyfac_finetuned_partial_10"
        classifier_model = SetFitModel.from_pretrained(classifier_model_path)
    elif model_name == "setfit_gyfac_partial_5":
        classifier_model_path = "./model_checkpoints/setfit/gyfac_finetuned_partial_5"
        classifier_model = SetFitModel.from_pretrained(classifier_model_path)
    elif model_name == "setfit_chatgpt":
        classifier_model_path = "./model_checkpoints/setfit/chatgpt_finetuned"
        classifier_model = SetFitModel.from_pretrained(classifier_model_path)
    else:
        raise NotImplementedError
    return classifier_model

def get_dataset(dataset_name, samples):
    if dataset_name == "paws-x":
        pawsx_data = load_dataset("paws-x", "en", split="test")
        data = pawsx_data['sentence1'][:samples]
    elif dataset_name == "gyfac":
        gyafc_formal_dataset = load_dataset("DattaBS/style_transfer", "formal", split="train", ignore_verifications=True)
        gyafc_informal_dataset = load_dataset("DattaBS/style_transfer", "informal", split="train", ignore_verifications=True)
        data1 = gyafc_formal_dataset['Sentence'][:samples]
        data2= gyafc_informal_dataset['Sentence'][:samples]
        data=data1+data2
    elif dataset_name == "daily_dialog":
        dataset = load_dataset("daily_dialog", split="train", ignore_verifications=True)
        data = []
        it = 0
        while len(data)<samples:
            data += dataset["dialog"][it]
            it+=1
        data = data[:samples]
    return data


class ClassifierGuidance(LogitsProcessor):
    def __init__(self, classifier_model, prediction_topk, condition_lambda, condition_class):
        self.classifier_model = classifier_model
        self.prediction_topk = prediction_topk
        self.condition_lambda = condition_lambda
        self.condition_class = condition_class
        self.softmax = torch.nn.Softmax(dim=-1)


    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        # input_ids : batch x seq
        # scores : batch x vocab
        batch_size = input_ids.shape[0]
        top_logits, top_indices = scores.topk(self.prediction_topk, dim=1) # batch x topk
        scores_topk_only = torch.full_like(scores, -float('inf'))  # Initialize all to -inf
        for i in range(batch_size):
            scores_topk_only[i].scatter_(0, top_indices[i], top_logits[i])

        if self.condition_lambda == 0:
            condition_logits_full = torch.zeros_like(scores_topk_only).float()
        else:
            # tplus1_candidates = torch.cat([input_ids.unsqueeze(1).expand(-1, self.prediction_topk, -1), top_indices.unsqueeze(2)], dim=2)[:, :, 1:] # batch x topk x seq+1, with pad dropped
            tplus1_candidates = torch.cat([input_ids.unsqueeze(1).expand(-1, self.prediction_topk, -1), top_indices.unsqueeze(2)], dim=2)[:, :, :] # batch x topk x seq+1
            # classifier_prob : batch*topk x n_class
            classifier_prob = self.classifier_model.predict_proba(
                                                tplus1_candidates.flatten(0, 1).cpu().tolist(), # batch*topk x seq+1
                                                )
            classifier_prob = classifier_prob.to(dtype=torch.float32, device=scores.device)

            classifier_prob = classifier_prob[:, self.condition_class]
            classifier_prob = classifier_prob.view(batch_size, self.prediction_topk) # batch x topk
            condition_logits = torch.log(classifier_prob) # batch x topk
            condition_logits_full = torch.full_like(scores, -float('inf'))  # Initialize all to -inf
            for i in range(batch_size):
                condition_logits_full[i].scatter_(0, top_indices[i], condition_logits[i])
        scores_topk_only = scores_topk_only + self.condition_lambda * condition_logits_full
        return scores_topk_only


def parse_arguments():
    parser = argparse.ArgumentParser(description="Argument Parser for GPTZero Configuration")
    
    # Generation models
    parser.add_argument('--generation_model_name', type=str, default="humarin",
                        help="Name of the generation model (default: 'humarin')")
    
    # Classifier models
    parser.add_argument('--classifier_model_name', type=str, default="setfit_gyfac",
                        help="Name of the classifier model (default: 'setfit_gyfac')")
    
    # Dataset name
    parser.add_argument('--dataset_name', type=str, default="paws-x",
                        help="Name of the dataset (default: 'paws-x')")
    
    # Generation Arguments
    parser.add_argument('--samples', type=int, default=500,
                        help="Number of samples for generation (default: 500)")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for generation (default: 32)")
    
    # Results directory
    parser.add_argument('--results_dir', type=str, default="generation_results",
                        help="Directory to save generation results (default: 'generation_results')")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    print(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"


    # Generation models
    generation_model_name = args.generation_model_name

    # Clasifier models
    classifier_model_name = args.classifier_model_name

    # datasetname
    dataset_name = args.dataset_name

    # Generation Arguments
    samples = args.samples
    batch_size = args.batch_size

    # results
    results_dir = "generation_results"

    model, tokenizer = get_generation_model(generation_model_name)
    classifier_model = get_classifier_model(classifier_model_name)

    model.to(device)
    classifier_model.to(device)

    # Load dataset
    dataset = get_dataset(dataset_name, samples)

    save_dir = os.path.join(results_dir, generation_model_name, classifier_model_name, dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    file_name = os.path.join(save_dir, "paraphrase.csv")
    if not os.path.exists(file_name):
        generated_paraphrase_texts = []
        start=time.time()
        for i in tqdm(range(ceil(samples/batch_size))):
            input_ids = tokenizer(dataset[i*batch_size:min(samples,(i+1)*batch_size)], return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids
            result = model.generate(
                input_ids.to(device),
                max_new_tokens=100,
                do_sample=False,
                return_dict_in_generate=True,
                output_hidden_states=True,
                output_scores=True
            )
            generated_paraphrase_texts+=tokenizer.batch_decode(result['sequences'], skip_special_tokens=True)
        print("paraphrase gen time: " + str(time.time()-start))
        dict = {'Sentence': dataset[:samples], 'Paraphrase': generated_paraphrase_texts}
        df = pd.DataFrame(dict)
        df.to_csv(file_name, index=False)
    else:
        print(f"Filed Already exixts: {file_name}")

    condition_lamdas = [10, 15, 20]
    # condition_lamdas = [20]
    condition_classes = [0, 1]

    for condition_class in condition_classes:
        for condition_lambda in condition_lamdas:
            save_dir = os.path.join(results_dir, generation_model_name, classifier_model_name, dataset_name)
            column_name = f"{'Formal' if condition_class==0 else 'Informal'}_{condition_lambda}"
            os.makedirs(save_dir, exist_ok=True)
            file_name = os.path.join(save_dir, f"{column_name}.csv")
            if not os.path.exists(file_name):
                generated_texts = []
                start=time.time()
                for i in tqdm(range(ceil(samples/batch_size))):
                    input_ids = tokenizer(dataset[i*batch_size:min(samples,(i+1)*batch_size)], return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids
                    logits_processor_list = LogitsProcessorList([
                            ClassifierGuidance(classifier_model, 20, condition_lambda=condition_lambda, condition_class=condition_class),
                        ])
                    result = model.generate(
                            input_ids.to(device),
                            max_new_tokens=100,
                            do_sample=False,
                            return_dict_in_generate=True,
                            output_hidden_states=True,
                            output_scores=True,
                            logits_processor=logits_processor_list,
                            no_repeat_ngram_size = 2
                        )
                    generated_texts += tokenizer.batch_decode(result['sequences'], skip_special_tokens=True)
                print("Formal gen time: " + str(time.time()-start))

                dict = {'Sentence': dataset[:len(generated_texts)], column_name: generated_texts}
                df = pd.DataFrame(dict)
                df.to_csv(file_name, index=False)
            else:
                print(f"Filed Already exixts: {file_name}")

    save_dir = os.path.join(results_dir, generation_model_name, classifier_model_name, dataset_name)
    df =  pd.read_csv(os.path.join(save_dir,'paraphrase.csv'))
    summary_dict = {'Sentence': list(df[:samples]['Sentence']), "Paraphrase":list(df[:samples]['Paraphrase'])}

    for condition_class in condition_classes:
        for condition_lambda in condition_lamdas:
            column_name = f"{'Formal' if condition_class==0 else 'Informal'}_{condition_lambda}"
            file_name = os.path.join(save_dir, f"{column_name}.csv")
            _df = pd.read_csv(file_name)
            summary_dict[column_name] = list(_df[column_name])
    df = pd.DataFrame(summary_dict)
    df.to_csv(os.path.join(save_dir, "summary.csv"), index=False)
        