import os
import random
import numpy as np
from datasets import load_from_disk, Dataset
from transformers import T5Tokenizer
from tqdm import tqdm

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def read_gyfac_split(data_path, split='train'):
    texts, labels = [], []
    for label in [0, 1]:
        file = open(os.path.join(data_path, f"{split}_{label}.txt"))
        sentences = [sentence.strip() for sentence in file.readlines()]
        print(f"{len(sentences)} examples with label {label} in split {split}")
        texts += sentences
        labels += [label for _ in sentences]
        file.close()
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    return texts, labels

def prepare_split(data_path, split, tokenizer, cut_k=0):
    texts, labels = read_gyfac_split(data_path, split)
    data_dict = {"text": texts, "label": labels}
    dataset = Dataset.from_dict(data_dict)
    
    if cut_k>0:
        print(f"cutting sequences chi-squared({cut_k})")
    
    def tokenize_function(examples):
        tokenized_outputs = tokenizer(examples["text"], truncation=True, padding=False)
        if cut_k==0:
            return tokenized_outputs
        
        # cut it according to a chi-squared distribution
        truncation_lengths = np.floor(np.random.chisquare(cut_k, size=len(examples["text"]))).astype(int)
        original_lengths = [len(sequence)-1 for sequence in tokenized_outputs["input_ids"]] # original length without eos

        for i in range(len(tokenized_outputs["input_ids"])):
            original_length = original_lengths[i]
            new_length = max(0, original_length - truncation_lengths[i])
            tokenized_outputs["input_ids"][i] = tokenized_outputs["input_ids"][i][:new_length-1] + tokenized_outputs["input_ids"][i][-1:]
            tokenized_outputs["attention_mask"][i] = tokenized_outputs["attention_mask"][i][:new_length-1] + tokenized_outputs["attention_mask"][i][-1:]

        return tokenized_outputs

    dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    for seq in dataset:
        print(seq.keys())
        break
    avg_sequence_length = np.mean([len(seq['input_ids']) for seq in dataset])
    print(f"Average sequence length after: {avg_sequence_length}")
    
    return dataset



def load_gyfac(tokenizer, data_path = "/work/pi_dhruveshpate_umass_edu/project_18/gyfac_pilot", seed=1, val_split=0.05, cut_k=0, refresh=False):
    directory = f"{seed}_{val_split}_{cut_k}"
    full_path = os.path.join(data_path, directory)
    
    if os.path.exists(full_path) or not refresh:
        print("loading from saved dataset")
        train_dataset = load_from_disk(os.path.join(full_path, "train_dataset"))
        val_dataset = load_from_disk(os.path.join(full_path, "val_dataset"))
        test_dataset = load_from_disk(os.path.join(full_path, "test_dataset"))
    else:
        print("preparing split newly")
        full_train_dataset = prepare_split(data_path, "train", tokenizer, cut_k)
        train_splits = full_train_dataset.train_test_split(test_size=val_split, shuffle=True, seed=seed)
        train_dataset = train_splits["train"]
        val_dataset = train_splits["test"]
        test_dataset = prepare_split(data_path, "test", tokenizer, cut_k)
        
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        train_dataset.save_to_disk(os.path.join(full_path, "train_dataset"))
        val_dataset.save_to_disk(os.path.join(full_path, "val_dataset"))
        test_dataset.save_to_disk(os.path.join(full_path, "test_dataset"))
    print("Train Dataset:", train_dataset)
    print("Validation Dataset:", val_dataset)
    print("Test Dataset:", test_dataset)
    
    return train_dataset, val_dataset, test_dataset


def main():
    seed = 1
    val_split = 0.05
    cut_k = 2
    set_seed(seed)
    
    hf_key = "t5-base"
    tokenizer = T5Tokenizer.from_pretrained(hf_key, use_fast=True, legacy=True, legacy=True)
    data_path = "/work/pi_dhruveshpate_umass_edu/project_18/gyfac_pilot"
    train_dataset, val_dataset, test_dataset = load_gyfac(tokenizer, data_path, seed, val_split, cut_k)

if __name__ == "__main__":
    main()
