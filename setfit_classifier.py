import os
import torch
import random
import argparse
import numpy as np

from datasets import load_dataset, Dataset
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class StyleClassifier():
    def __init__(self) -> None:
        self.model = SetFitModel.from_pretrained("sentence-transformers/sentence-t5-base").to(device)

    def init_trainer(self, finetune_dataset, batch_size=16, num_iterations=20, num_epochs=20):
        self.trainer = SetFitTrainer(
                        model=self.model,
                        train_dataset=finetune_dataset,
                        # eval_dataset=test_ds,
                        loss_class=CosineSimilarityLoss,
                        batch_size=batch_size,
                        num_iterations=num_iterations, # Number of text pairs to generate for contrastive learning
                        num_epochs=num_epochs # Number of epochs to use for contrastive learning
                    )
    
    def get_training_data(self, FORMAL_SENTENCES, INFORMAL_SENTENCES):
        data = {
            "text": FORMAL_SENTENCES + INFORMAL_SENTENCES,
            "label": [0]*len(FORMAL_SENTENCES) + [1]*len(INFORMAL_SENTENCES),
            "label_text": ["formal"]*len(FORMAL_SENTENCES) + ["informal"]*len(INFORMAL_SENTENCES),
        }
        return Dataset.from_dict(data)

    def get_data(self, file_name):
        data = []
        with open(file_name, 'r')as f:
            for line in f:
                data.append(line.strip())
        return data
    
    def get_partials(self, sentences):
        partial_sentences = []
        for s in sentences:
            words = s.split()
            # random_indices = sorted(random.sample(range(1, len(words)-1), 5))
            len_words = len(words)
            n_partials = 6
            if len_words > n_partials:
                indices = [i for i in range(len_words//n_partials, len_words, len_words//n_partials)]
                partials = [' '.join(words[:index]) for index in indices[:n_partials-1]]
                partial_sentences += partials
            else:
                partials = [' '.join(words[:index]) for index in range(1, len_words)]
                partial_sentences += partials
            partial_sentences.append(s)
        return partial_sentences
    
    def finte_tune(self, file1, file2, n):
        formal_sentences = self.get_data(file1)[:n]
        informal_sentences = self.get_data(file2)[:n]
        formal_sentences = self.get_partials(formal_sentences)
        informal_sentences = self.get_partials(informal_sentences)
        # print("Total partial sentences: ", len(formal_sentences + informal_sentences))
        # print(formal_sentences, informal_sentences)
        finetune_dataset = self.get_training_data(formal_sentences, informal_sentences)
        self.init_trainer(finetune_dataset)
        self.trainer.train()
        dataset = os.path.basename(file1).split('_')[0]
        model_path = 'model_checkpoints/setfit/gyfac_finetuned_partial_' + str(n)
        # if 'gyfac' in dataset:
        #     model_path += 'gyfac_finetuned_partial_' + str(n)
        # else:
        #     model_path += 'chatgpt_finetuned_partial_' + str(n)
        self.model.save_pretrained(model_path)
        print("Model Successfully saved at", model_path, "!")








if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='Input File Path')
    parser.add_argument('-f', '--formal', help='Formal file')
    parser.add_argument('-if', '--informal', help='Informal file')
    parser.add_argument('-n', '--training_length')

    args = parser.parse_args()
    if args.path and args.formal and args.informal:
        set_seed(42)
        setfit = StyleClassifier()
        formal_file = os.path.join(args.path, args.formal)
        informal_file = os.path.join(args.path, args.informal)
        # print(formal_file, informal_file)
        setfit.finte_tune(formal_file, informal_file, n=int(args.training_length))
        # setfit.finte_tune(formal_file, informal_file, n=5)
        # setfit.finte_tune(formal_file, informal_file, n=10)
    else:
        print("No input files. use help")
    pass