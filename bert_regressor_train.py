import torch, os, random
import pandas as pd
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig
from data import prepare_pavlick_formality, set_seed, show_random_elements
from transformers import logging, DataCollatorWithPadding, TrainerCallback

class LoggingCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            print(f"Step {state.global_step}: Evaluation loss: {metrics.get('eval_loss', 'No loss computed')}")

class BertForRegression(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 1
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.regressor = torch.nn.Linear(config.hidden_size, 1)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds)
        pooled_output = self.dropout(outputs[1])
        logits = self.regressor(pooled_output)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1), labels.view(-1))
        return (loss, logits) if loss is not None else logits

def evaluate_sentence_by_sentence(model, tokenizer, sentences):
    model.eval()
    scores = []
    with torch.no_grad():
        for sentence in sentences:
            inputs = tokenizer(sentence, return_tensors='pt', padding='max_length', truncation=True, max_length=512).to("cuda")
            outputs = model(**inputs)
            prediction = outputs.to("cpu").item()
            print(sentence, prediction)

def evaluate_sentences_in_batches(model, tokenizer, sentences, batch_size=60):
    model.eval()
    all_scores=[]
    num_batches=len(sentences)//batch_size

    with torch.no_grad():
        for i in range(num_batches+1):
            batch_sentences = sentences[i*batch_size: (i+1)*batch_size]
            inputs = tokenizer(batch_sentences, return_tensors='pt', padding='max_length', truncation=True, max_length=512).to("cuda")
            scores = model(**inputs).squeeze().to("cpu").tolist()
            all_scores.extend(scores)

    avg_score = sum(all_scores)/len(all_scores) if all_scores else 0
    return avg_score

def read_gyfac_split(data_path, label=0, split='train'):
    file_path = os.path.join(data_path, f"{split}_{label}.txt")
    with open(file_path, 'r') as file:
        sentences = [sentence.strip() for sentence in file.readlines()]
    print(f"{len(sentences)} examples with label {label} in split {split}")
    return sentences

def main():
    # cite
    # @article{PavlickAndTetreault-2016:TACL,
    #   author =  {Ellie Pavlick and Joel Tetreault},
    #   title =   {An Empirical Analysis of Formality in Online Communication},
    #   journal = {Transactions of the Association for Computational Linguistics},
    #   year =    {2016},
    #   publisher = {Association for Computational Linguistics}
    # }

    # @article{Lahiri-2015:arXiv,
    #   title={{SQUINKY! A} Corpus of Sentence-level Formality, Informativeness, and Implicature},
    #   author={Lahiri, Shibamouli},
    #   journal={arXiv preprint arXiv:1506.02306},
    #   year={2015}
    # }
    set_seed(42)
    num_epochs=6
    batch_size=48
    gradient_accumulation_steps=1
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train, val, test = prepare_pavlick_formality(tokenizer, seed=42, refresh=False)
    show_random_elements(train, tokenizer, 20)
    
    steps_per_epoch = len(train)//(batch_size*gradient_accumulation_steps)
    print(f"steps_per_epoch:\t{steps_per_epoch}")
    max_steps = int(num_epochs*len(train))//(batch_size*gradient_accumulation_steps)
    print(f"max_steps: {max_steps}")
    log_steps = steps_per_epoch//10
    eval_steps = log_steps*2
    save_steps = eval_steps
    
    config = BertConfig.from_pretrained("bert-base-uncased")
    config.num_labels = 1
    model = BertForRegression.from_pretrained("bert-base-uncased", config=config)

    training_args = TrainingArguments(
        output_dir='./results_evaluator',
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size*2,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./results_evaluator',
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
        train_dataset=train,
        eval_dataset=val,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=None),
        callbacks=[LoggingCallback()]
        
    )

    trainer.train()
    eval_result = trainer.evaluate(test)
    print("Test result", eval_result)
    trainer.save_model("./results_evaluator")
    
    gyfac_train = "/work/pi_dhruveshpate_umass_edu/project_18/gyfac_pilot"
    print("Evaluating informal sentences...")
    informal_sentences = read_gyfac_split(gyfac_train, label=1, split='test')
    print("avg score", evaluate_sentences_in_batches(model, tokenizer, informal_sentences))
    evaluate_sentence_by_sentence(model, tokenizer, random.sample(informal_sentences, 10))

    formal_sentences = read_gyfac_split(gyfac_train, label=0, split='test')
    print("avg score", evaluate_sentences_in_batches(model, tokenizer, formal_sentences))
    evaluate_sentence_by_sentence(model, tokenizer, random.sample(formal_sentences, 10))
    
    random_indices = random.sample(range(500), 10)
    
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # config = BertConfig.from_pretrained("bert-base-uncased")
    # config.num_labels = 1
    # model_path = "/work/pi_dhruveshpate_umass_edu/amekala_umass_edu/tf-gyfac-classifier/results_evaluator/checkpoint-380"
    # model = BertForRegression.from_pretrained(model_path, config=config).to("cuda")
    print("Evaluating setfit_gyfac sentences...")
    setfit_gyfac_file = "/work/pi_dhruveshpate_umass_edu/yashwanthbab_umass_edu/nlp/generation_results/humarin/setfit_gyfac/daily_dialog/summary.csv"
    setfit_gyfac_sentences = pd.read_csv(setfit_gyfac_file)['Informal_15'].dropna().tolist()
    print(len(setfit_gyfac_sentences))
    print("avg score", evaluate_sentences_in_batches(model, tokenizer, setfit_gyfac_sentences))
    print("\n")
    sample = [setfit_gyfac_sentences[i] for i in random_indices]
    evaluate_sentence_by_sentence(model, tokenizer, sample)
    print("\n\n")
    
    print("Evaluating t5_gyfac_full sentences...")
    t5_gyfac_full_file = "/work/pi_dhruveshpate_umass_edu/yashwanthbab_umass_edu/nlp/generation_results/humarin/t5_gyfac_full/daily_dialog/summary.csv"
    t5_gyfac_full_sentences = pd.read_csv(t5_gyfac_full_file)['Informal_15'].dropna().tolist()
    print(len(t5_gyfac_full_sentences))
    print("avg score", evaluate_sentences_in_batches(model, tokenizer, t5_gyfac_full_sentences))
    print("\n")
    sample = [t5_gyfac_full_sentences[i] for i in random_indices]
    evaluate_sentence_by_sentence(model, tokenizer, sample)


if __name__ == "__main__":
    main()
