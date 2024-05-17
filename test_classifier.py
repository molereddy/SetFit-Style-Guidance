import os, sys
import random
from pathlib import Path
import pandas as pd
import torch
from transformers import T5Tokenizer, AutoModelForSequenceClassification, logging, DataCollatorWithPadding
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from data import load_gyfac

def evaluate_sample_by_sample(model, dataset, tokenizer):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for example in dataset:
        inputs = tokenizer(example['text'], return_tensors="pt", padding=True, truncation=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_label = torch.argmax(logits, dim=1).cpu().numpy()[0]
        
        print(f"Text: {example['text']}")
        print(f"True Label: {example['label']}, Predicted Label: {predicted_label}")
        print("---")

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

def evaluate_model(model, test_dataset, batch_size, collator):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    true_labels = []
    predictions = []
    
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, collate_fn=collator)
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_labels = torch.argmax(logits, dim=1)

            predictions.extend(predicted_labels.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
    accuracy = accuracy_score(true_labels, predictions)

    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, Accuracy: {accuracy:.4f}")

test_samples = [
    {"text": "Hey, what’s up? Just got a new phone, it's so cool!", "label": 1},
    {"text": "Gonna grab some coffee, wanna join?", "label": 1},
    {"text": "Lol, did you see that cat video? Hilarious!", "label": 1},
    {"text": "I can't believe it's already Friday! Time flies.", "label": 1},
    {"text": "Y’all ready for the game tonight? It’s gonna be epic!", "label": 1},
    {"text": "Just chilling at home, what about you?", "label": 1},
    {"text": "OMG, this pizza is amazing! Best I've ever had.", "label": 1},
    {"text": "Can't wait for the weekend, got so many plans!", "label": 1},
    {"text": "Ugh, I hate Mondays. Wish weekends were longer.", "label": 1},
    {"text": "Do you know any good places to hang out around here?", "label": 1},
    {"text": "The company launched its new product last week; it's expected to significantly increase market share.", "label": 0},
    {"text": "Local weather forecast predicts rain for the entire weekend.", "label": 0},
    {"text": "Annual revenue exceeded expectations by 15%, marking a notable success in the fiscal year.", "label": 0},
    {"text": "The construction project down by the river will wrap up by late fall.", "label": 0},
    {"text": "Observations confirm that the migration patterns are shifting earlier each year.", "label": 0},
    {"text": "Further research is required to understand the implications of this change fully.", "label": 0},
    {"text": "The latest novel by the author was released to critical acclaim but mixed audience reviews.", "label": 0}
]


def main():
    hf_key = "t5-base"
    tokenizer = T5Tokenizer.from_pretrained(hf_key, use_fast=True, legacy=True)
    NUM_LABELS = 2
    seed, val_split, cut_k = 1, 0.05, 0
    model_path = "results_1_0.05_0/b_checkpoint-3744"
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=NUM_LABELS)

    _, _, test_dataset = load_gyfac(tokenizer, seed=seed, val_split=val_split, cut_k=cut_k)
    batch_size = 256
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=None)
    
    log_file_path = os.path.join(model_path, "eval_logs.txt")
    with open(log_file_path, "w") as f:
        sys.stdout = f

        show_random_elements(test_dataset, tokenizer)
        evaluate_model(model, test_dataset, batch_size, data_collator)
        evaluate_sample_by_sample(model, test_samples, tokenizer)
    
if __name__ == "__main__":
    main()
