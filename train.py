import os, random
from pathlib import Path
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import torch
from transformers import T5Tokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments, logging, DataCollatorWithPadding, TrainerCallback

from data import load_gyfac

class LoggingCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            print(f"Step {state.global_step}: Evaluation loss: {metrics.get('eval_loss', 'No loss computed')}")


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
    
    train_dataset, val_dataset, test_dataset = load_gyfac(tokenizer, seed=seed, 
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
    print("GYFAC test result", eval_result)

if __name__ == "__main__":
    main()
