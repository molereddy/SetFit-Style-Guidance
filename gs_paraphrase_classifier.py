import os
import pandas as pd
import argparse
import json

from sentence_transformers import CrossEncoder





class Classifier():
    def __init__(self) -> None:
        self.label_mapping = ['contradiction', 'entailment', 'neutral']
        self.model = CrossEncoder('cross-encoder/nli-roberta-base')

    def get_sentence_pairs(self, file):
        df = pd.read_csv(file)
        input_strings = df['Sentence'].tolist() 
        paraphrase_inputs = df['Paraphrase'].tolist() 
        formal_inputs1 = df['Formal_10'].tolist() 
        informal_inputs1 = df['Informal_10'].tolist() 
        formal_inputs2 = df['Formal_15'].tolist() 
        informal_inputs2 = df['Informal_15'].tolist() 
        formal_inputs3 = df['Formal_20'].tolist()
        informal_inputs3 = df['Informal_20'].tolist() 
        data = {
            "paraphrase" : [(x, y) for x, y in zip(input_strings, paraphrase_inputs)],
            "formal_10" : [(x, y) for x, y in zip(input_strings, formal_inputs1)],
            "formal_15" : [(x, y) for x, y in zip(input_strings, formal_inputs2)],
            "formal_20" : [(x, y) for x, y in zip(input_strings, formal_inputs3)],
            "informal_10" : [(x, y) for x, y in zip(input_strings, informal_inputs1)],
            "informal_15" : [(x, y) for x, y in zip(input_strings, informal_inputs2)],
            "informal_20" : [(x, y) for x, y in zip(input_strings, informal_inputs3)],
        }
        return data
    
    def make_result_file(self, file, data):
        dir_path = os.path.dirname(file)
        file_path = os.path.join(dir_path, 'paraphrase_results.json')
        with open(file_path, "w") as json_file:
            json.dump(data, json_file)

        print(f"{os.path.basename(dir_path)} results saved succesfully")

    def classify(self, file):
        print("Classifying entailment")
        data = self.get_sentence_pairs(file)
        batch_size = 20
        result = {}
        for key in data.keys(): 
            print(f'    {key} colomn')
            res = {}
            entail_count = 0
            total = len(data[key])
            if "Doc_length" not in result.keys():
                result["Doc_length"] = total
            batches = [data[key][i:i+batch_size] for i in range(0, total, batch_size)]
            for batch in batches:
                scores = self.model.predict(batch)
                labels = [self.label_mapping[score_max] for score_max in scores.argmax(axis=1)]
                entail_count += labels.count('entailment')
            percent_entailed = (entail_count/total)*100
            res["entail_count"] = entail_count
            res["entail_percent"] = percent_entailed
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