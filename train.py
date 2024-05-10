import os
import random
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from data import load_gyfac

def show_random_elements(dataset, tokenizer, num_examples=10):
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

    df = pd.DataFrame(examples)
    print(df)

def main():
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/sentence-t5-base")
    NUM_LABELS = 2
    model = AutoModelForSequenceClassification.from_pretrained("sentence-transformers/sentence-t5-base", num_labels=NUM_LABELS)

    train_dataset, val_dataset, test_dataset = load_gyfac()

    training_args = TrainingArguments(
        output_dir=os.path.join('results'),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join('logs'),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()

    show_random_elements(train_dataset, tokenizer)

if __name__ == "__main__":
    main()
