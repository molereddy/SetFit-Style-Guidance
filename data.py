import os
import random
import numpy as np
from datasets import load_from_disk, Dataset, load_dataset
from transformers import T5Tokenizer
from tqdm import tqdm

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def show_random_elements(dataset, tokenizer, num_examples=5):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    examples = []
    for pick in picks:
        example = dataset[pick]
        sentence = tokenizer.decode(example['input_ids'], skip_special_tokens=True)
        label = example['label']
        examples.append({'Sentence': sentence, 'Label': label})
    
    pd.set_option('display.max_colwidth', None)
    df = pd.DataFrame(examples)
    print(df)

def prepare_pavlick_formality(tokenizer, seed=42, refresh=False):
    def preprocess_dataset_cols(dataset):
        dataset = dataset.rename_column("avg_score", "label")
        dataset = dataset.remove_columns(["domain", "sentence"])
        return dataset

    def preprocess_tokenize(examples):
        return tokenizer(examples['sentence'], padding='max_length', truncation=True)

    data_path = "pavlick" # path to pavlick dataset object after storage

    if os.path.exists(os.path.join(data_path, "train_dataset")) and not refresh:
        print("loading from saved dataset")
        train_dataset = load_from_disk(os.path.join(data_path, "train_dataset"))
        val_dataset = load_from_disk(os.path.join(data_path, "val_dataset"))
        test_dataset = load_from_disk(os.path.join(data_path, "test_dataset"))
    else:
        print("preparing split newly")
        dataset = load_dataset("osyvokon/pavlick-formality-scores")
        dataset = dataset.map(preprocess_tokenize, batched=True)
        dataset = preprocess_dataset_cols(dataset)
        for split in dataset.keys():
            print(f"Attributes in {split} split: {dataset[split].column_names}")
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        
        train_dataset = dataset['train'].shuffle(seed=42)
        test_dataset = dataset['test']
        eval_splits = test_dataset.train_test_split(test_size=0.5, shuffle=True, seed=seed)
        test_dataset = eval_splits["train"]
        val_dataset = eval_splits["test"]
        
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        train_dataset.save_to_disk(os.path.join(data_path, "train_dataset"))
        val_dataset.save_to_disk(os.path.join(data_path, "val_dataset"))
        test_dataset.save_to_disk(os.path.join(data_path, "test_dataset"))
    
    print("Train Dataset:", train_dataset)
    print("Validation Dataset:", val_dataset)
    print("Test Dataset:", test_dataset)
    
    return train_dataset, val_dataset, test_dataset


def prepare_gyafc_split(data_path, split, tokenizer, cut_k=0):
    
    def read_gyafc_split(data_path, split='train'):
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
    
    texts, labels = read_gyafc_split(data_path, split)
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



def load_gyafc(tokenizer, data_path = "PATH TO GYAFC DATASET", seed=1, val_split=0.05, cut_k=0, refresh=False):
    
    directory = f"{seed}_{val_split}_{cut_k}"
    full_path = os.path.join(data_path, directory)
    
    if os.path.exists(os.path.join(full_path, "train_dataset")) and not refresh:
        print("loading from saved dataset")
        train_dataset = load_from_disk(os.path.join(full_path, "train_dataset"))
        val_dataset = load_from_disk(os.path.join(full_path, "val_dataset"))
        test_dataset = load_from_disk(os.path.join(full_path, "test_dataset"))
    else:
        print("preparing split newly")
        full_train_dataset = prepare_gyafc_split(data_path, "train", tokenizer, cut_k)
        train_splits = full_train_dataset.train_test_split(test_size=val_split, shuffle=True, seed=seed)
        train_dataset = train_splits["train"]
        val_dataset = train_splits["test"]
        test_dataset = prepare_gyafc_split(data_path, "test", tokenizer, cut_k)
        
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
    tokenizer = T5Tokenizer.from_pretrained(hf_key, use_fast=True, legacy=True)
    data_path = "PATH TO GYAFC DATASET"
    train_dataset, val_dataset, test_dataset = load_gyafc(tokenizer, data_path, seed, val_split, cut_k)

if __name__ == "__main__":
    main()
