import os
import random
import numpy as np
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer
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

def prepare_split(data_path, split, tokenizer, cut=False):
    texts, labels = read_gyfac_split(data_path, split)
    data_dict = {"text": texts, "label": labels}
    dataset = Dataset.from_dict(data_dict)

    def tokenize_function(examples):
        tokenized_outputs = tokenizer(examples["text"], truncation=True, padding=False)
        if not cut:
            return tokenized_outputs
        # cut it according to a chi-squared distribution
        truncation_lengths = np.floor(np.random.chisquare(1, size=len(examples["text"]))).astype(int)
        original_lengths = [len(sequence)-1 for sequence in tokenized_outputs["input_ids"]] # original length without eos

        for i in range(len(tokenized_outputs["input_ids"])):
            original_length = original_lengths[i]
            new_length = max(0, original_length - truncation_lengths[i])
            tokenized_outputs["input_ids"][i] = tokenized_outputs["input_ids"][i][:new_length-1] + tokenized_outputs["input_ids"][i][-1:]
            tokenized_outputs["attention_mask"][i] = tokenized_outputs["attention_mask"][i][:new_length-1] + tokenized_outputs["attention_mask"][i][-1:]

        return tokenized_outputs

    dataset = dataset.map(tokenize_function, batched=True)
    return dataset



def load_gyfac(tokenizer, data_path = "/work/pi_dhruveshpate_umass_edu/project_18/gyfac_pilot", seed=1, val_split=0.05, cut=False):
    directory = f"{seed}_{val_split}_{cut}"
    full_path = os.path.join(data_path, directory)
    
    if os.path.exists(full_path):
        train_dataset = load_from_disk(os.path.join(full_path, "train_dataset"))
        val_dataset = load_from_disk(os.path.join(full_path, "val_dataset"))
        test_dataset = load_from_disk(os.path.join(full_path, "test_dataset"))
    else:
        full_train_dataset = prepare_split(data_path, "train", tokenizer, cut)
        train_splits = full_train_dataset.train_test_split(test_size=val_split, shuffle=True, seed=seed)
        train_dataset = train_splits["train"]
        val_dataset = train_splits["test"]
        test_dataset = prepare_split(data_path, "test", tokenizer, cut)
        
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
    cut=True
    set_seed(seed)
    
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/sentence-t5-base")
    data_path = "/work/pi_dhruveshpate_umass_edu/project_18/gyfac_pilot"
    train_dataset, val_dataset, test_dataset = load_gyfac(tokenizer, data_path, seed, val_split, cut)

if __name__ == "__main__":
    main()
