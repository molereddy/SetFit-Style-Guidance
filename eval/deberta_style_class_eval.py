import torch
import torch.nn.functional as F
import os 
import pandas as pd 
import argparse
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Classifier():
    def __init__(self) -> None:
        model_name = 's-nlp/deberta-large-formality-ranker'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    
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

    def get_tokens(self, data):  
        return self.tokenizer(data, return_tensors="pt", padding=True, truncation=True)
    
    def make_result_file(self, file, data):
        dir_path = os.path.dirname(file)
        file_path = os.path.join(dir_path, 'formality_results.json')
        with open(file_path, "w") as json_file:
            json.dump(data, json_file)

        print(f"{os.path.basename(dir_path)} results saved succesfully")

    def classify(self, file):
        print("Classifying formality")
        data = self.get_input_data(file)
        result = {}
        for key in data.keys():
            res = {}
            print(f'    {key} colomn')
            inp = self.get_tokens(data[key]).to(device)
            if "Doc_length" not in result.keys():
                result["Doc_length"] = len(inp.input_ids)
                total = len(inp.input_ids)
            batch_size = 20
            with torch.no_grad():
                formal_count = 0
                informal_count = 0
                formal_probs = 0
                for i in range(0, len(inp.input_ids), batch_size):
                    batch_inp = {}
                    for k,t in inp.items():
                        batch_inp[k] = t[i:i+batch_size]
                    logits = self.model(**batch_inp).logits
                    probs = F.softmax(logits, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    formal_pred_indices = torch.where(preds==0)[0]
                    formal_probs += probs[formal_pred_indices].sum(dim=0)[0].item()
                    formal_count += torch.sum(preds == 0).item()
                    informal_count += torch.sum(preds == 1).item()
                percent_formal = (formal_count/total)*100
                res["formal_count"] = formal_count
                res["informal_count"] = informal_count
                res["formal_precent"] = percent_formal
                res["soft_score"] = (formal_probs/total)*100
                result[key] = res
        
        self.make_result_file(file, result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fp', '--file', help='Input File Path')

    args = parser.parse_args()
    if args.file:
        classifier = Classifier()
        classifier.classify(args.file)
    else:
        print("No input file. use -fp for file path")



