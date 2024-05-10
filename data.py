import random
from datasets import Dataset, DatasetDict

def read_gyfac_split(data_path, split='train'):
    texts, labels = [], []
    for label in [0, 1]:
        file = open(os.path.join(data_path, f"{split}_{label}.txt"))
        sentences = [sentence.strip() for sentence in file.readlines()]
        texts += sentences
        labels += [label for _ in sentences]
        file.close()
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    return texts, labels

def prepare_split(data_path, split, tokenizer):
    texts, labels = read_gyfac_split(data_path, split)
    data_dict = {"text": texts, "label": labels}
    dataset = Dataset.from_dict(data_dict)
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True)
    dataset = dataset.map(tokenize_function, batched=True)
    return dataset

def load_gyfac(data_path = "/work/pi_dhruveshpate_umass_edu/project_18/gyfac_pilot", val_split=0.1):
    full_train_dataset = prepare_split(data_path, "train", tokenizer)
    train_splits = full_train_dataset.train_test_split(test_size=val_split, shuffle=True)
    train_dataset = train_splits["train"]
    val_dataset = train_splits["test"]
    test_dataset = prepare_split(data_path, "test", tokenizer)
    return train_dataset, val_dataset, test_dataset