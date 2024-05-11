import os
import random
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging, DataCollatorWithPadding
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
        print("\n---\n")

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
    
    test_dataset=test_dataset.remove_columns(['text'])
    print(test_dataset.column_names)
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
    {"text": "The company launched its new product last week; it's expected to significantly increase market share.", "label": 0},
    {"text": "Local weather forecast predicts rain for the entire weekend.", "label": 1},
    {"text": "Annual revenue exceeded expectations by 15%, marking a notable success in the fiscal year.", "label": 0},
    {"text": "The construction project down by the river will wrap up by late fall.", "label": 1},
    {"text": "Observations confirm that the migration patterns are shifting earlier each year.", "label": 0},
    {"text": "He’s always late to meetings, isn’t he?", "label": 1},
    {"text": "Further research is required to understand the implications of this change fully.", "label": 0},
    {"text": "Cats are known for their independence compared to other pets.", "label": 1},
    {"text": "The latest novel by the author was released to critical acclaim but mixed audience reviews.", "label": 0},
    {"text": "Make sure to check out the new cafe downtown—they’ve got the best coffee!", "label": 1}
]

def main():
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/sentence-t5-base")
    NUM_LABELS = 2

    model_path = "best_model"
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=NUM_LABELS)

    # _, _, test_dataset = load_gyfac(tokenizer, seed=1, val_split=0.05, cut=True)
    # show_random_elements(test_dataset, tokenizer)
    # batch_size = 512
    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=None)
    # evaluate_model(model, test_dataset, batch_size, data_collator)
    evaluate_sample_by_sample(model, test_samples, tokenizer)
    
if __name__ == "__main__":
    main()
