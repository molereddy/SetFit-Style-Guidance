import torch
import torch.nn.functional as F
import os 
import pandas as pd 
import argparse
import json
import random
import numpy as np

from transformers import BertTokenizer, BertModel, BertForSequenceClassification, BertConfig


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    

class Classifier():
    def __init__(self, seed) -> None:
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        model_path = 'model_checkpoints/checkpoint-380'
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        config = BertConfig.from_pretrained("bert-base-uncased")
        config.num_labels = 1
        self.model = BertForRegression.from_pretrained(model_path, config=config).to("cuda")
    
    def get_input_data(self, file):
        df = pd.read_csv(file)
        input_strings = df['Paraphrase'].tolist() 
        formal_inputs1 = df['Formal_10'].tolist() 
        informal_inputs1 = df['Informal_10'].tolist() 
        formal_inputs2 = df['Formal_15'].tolist() 
        informal_inputs2 = df['Informal_15'].tolist() 
        formal_inputs3 = df['Formal_20'].tolist() 
        informal_inputs3 = df['Informal_20'].tolist() 
        data = {
            "original" : input_strings,
            "formal_10" : formal_inputs1,
            "formal_15" : formal_inputs2,
            "formal_20" : formal_inputs3,
            "informal_10" : informal_inputs1,
            "informal_15" : informal_inputs2,
            "informal_20" : informal_inputs3,
        }
        return data     
    
    def make_result_file(self, file, data):
        dir_path = os.path.dirname(file)
        file_path = os.path.join(dir_path, 'formality_results_regressor.json')
        with open(file_path, "w") as json_file:
            json.dump(data, json_file)

        print(f"{os.path.basename(dir_path)} regressor results saved succesfully")

    def evaluate_sentences_in_batches(self, sentences, batch_size=60):
        self.model.eval()
        all_scores=[]
        num_batches=len(sentences)//batch_size

        with torch.no_grad():
            for i in range(num_batches+1):
                batch_sentences = sentences[i*batch_size: (i+1)*batch_size]
                inputs = self.tokenizer(batch_sentences, return_tensors='pt', padding='max_length', truncation=True, max_length=512).to("cuda")
                scores = self.model(**inputs).squeeze().to("cpu").tolist()
                all_scores.extend(scores)

        avg_score = sum(all_scores)/len(all_scores) if all_scores else 0
        return avg_score

    def classify(self, file):
        print("Classifying formality")
        data = self.get_input_data(file)
        result = {}
        for key in data.keys():
            res = {}
            print(f'    {key} colomn')
            if "Doc_length" not in result.keys():
                result['Doc_length'] = len(data[key])
            res['formal_score'] = self.evaluate_sentences_in_batches(data[key])
            result[key] = res

        self.make_result_file(file, result)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fp', '--file', help='Input File Path')

    args = parser.parse_args()
    if args.file:
        classifier = Classifier(42)
        classifier.classify(args.file)
    else:
        print("No input file. use -fp for file path")
