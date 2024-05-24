import os, random, sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from data import load_gyafc
import torch
from transformers import T5Tokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments, logging, DataCollatorWithPadding, TrainerCallback

from data import load_gyafc, show_random_elements

class LoggingCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            print(f"Step {state.global_step}: Evaluation loss: {metrics.get('eval_loss', 'No loss computed')}")


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



def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions[0].argmax(-1)
    # guessing the first attribute of predictions has logits
    # https://discuss.huggingface.co/t/using-trainer-class-with-t5-what-is-returned-in-evalprediction-dict/1041/4
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    accuracy = accuracy_score(labels, preds)
    return {
        'accuracy': round(accuracy*100,1),
        'precision': round(precision*100,1),
        'recall': round(recall*100,1),
        'f1': round(f1*100,1)
    }

def main():
    
    num_epochs = 3
    batch_size = 24
    gradient_accumulation_steps = 2
    
    seed, val_split, cut_k = 1, 0.05, 0
    hf_key = "t5-base"
    NUM_LABELS = 2
    tokenizer = T5Tokenizer.from_pretrained(hf_key, use_fast=True, legacy=True)
    model = AutoModelForSequenceClassification.from_pretrained(hf_key, num_labels=NUM_LABELS)
    
    train_dataset, val_dataset, test_dataset = load_gyafc(tokenizer, seed=seed, 
                                                          val_split=val_split, 
                                                          cut_k=cut_k)
    show_random_elements(train_dataset, tokenizer)
    
    steps_per_epoch = len(train_dataset)//(batch_size*gradient_accumulation_steps)
    print(f"steps_per_epoch:\t{steps_per_epoch}")
    max_steps = int(num_epochs*len(train_dataset))//(batch_size*gradient_accumulation_steps)
    print(f"max_steps: {max_steps}")
    log_steps = steps_per_epoch//20
    eval_steps = log_steps*4
    save_steps = eval_steps
    
    training_args = TrainingArguments(
        output_dir=os.path.join(f'results_{seed}_{val_split}_{cut_k}'),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=256,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=500,
        max_steps=max_steps,
        weight_decay=0.01,
        logging_dir=os.path.join(f'logs'),
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        logging_strategy="steps",
        logging_steps=log_steps,
        save_strategy="steps",
        save_steps=save_steps,
        load_best_model_at_end=True,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=None),
        callbacks=[LoggingCallback()]
    )
    # trainer.evaluate()
    trainer.train()
    eval_result = trainer.evaluate(test_dataset)
    print("gyafc test result", eval_result)
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
