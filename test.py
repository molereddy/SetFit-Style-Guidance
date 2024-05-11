import os
import random
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from data import load_gyfac

def show_random_elements(dataset, tokenizer):
    sample = random.choice(dataset)
    print(tokenizer.decode(sample['input_ids'], skip_special_tokens=True))

def evaluate_model(model, test_dataset, batch_size):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    true_labels = []
    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            labels = batch['labels'].to(device)
            predicted_labels = torch.argmax(model(input_ids, attention_mask=attention_mask).logits, dim=1)

            predictions.extend(predicted_labels.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
    accuracy = accuracy_score(true_labels, predictions)

    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, Accuracy: {accuracy:.4f}")

def main():
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/sentence-t5-base")
    NUM_LABELS = 2

    logging.set_verbosity_error()
    model_path = "results/checkpoint-2496"
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=NUM_LABELS)
    logging.set_verbosity_warning()

    _, _, test_dataset = load_gyfac(tokenizer, seed=1, val_split=0.05, cut=True)
    show_random_elements(test_dataset, tokenizer)
    batch_size = 512
    evaluate_model(model, test_dataset, batch_size)

if __name__ == "__main__":
    main()
