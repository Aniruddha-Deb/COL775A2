#!/usr/bin/env python
# coding: utf-8
import os
from glob import glob
from collections import Counter
import transformers
import torch
import torch.nn as nn
import torch.optim as optim
from torch.profiler import profile, record_function, ProfilerActivity

from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

import torchvision
# from skimage import io
import cv2

import copy, json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from sklearn.metrics import accuracy_score

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

options = ['0', '1', '2', '3', '4', '5', 'yes', 'no', 'rubber', 'metal', 'sphere', 'cube', 'cylinder', 'gray', 'brown', 'green', 'red', 'blue', 'purple', 'yellow', 'cyan']
option_id_map = {
    o:i for i,o in enumerate(options)
}
id_option_map = {
    i:o for i,o in enumerate(options)
}
task_heads = ['descriptive', 'explanatory', 'predictive', 'counterfactual']
binary_id_map = {'wrong': 0, 'correct': 1}
id_binary_map = {0: 'wrong', 1: 'correct'}
ques_dict_keys = ["tokens", "q_ids", "choice_ids"]

class ProcessQuestions:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        pass
        
    def get_qa_batch(self, ques_list):
        '''
        INPUT:
        ques_list: list of question_data dictionary
        OUTPUT: question_dict, answer_dict
        descriptive: 
            question_list: list of <question> [SEP] <question_subtype>
            answer_list: list of respective answer as option_id_map
        explanatory:
            question_list: list of <question> [SEP] <choice_k>
            answer_list: list of respective answer as binary_id_map correct = 1 / wrong = 0
        predictive:
            question_list: list of <question> [SEP] <choice_k>
            answer_list: list of respective answer as binary_id_map correct = 1 / wrong = 0
        counterfactual:
            question_list: list of <question> [SEP] <choice_k>
            answer_list: list of respective answer as binary_id_map correct = 1 / wrong = 0
        '''
        
        question_dict = {i:{j:[] for j in ques_dict_keys} for i in task_heads}
        answer_dict = {i:[] for i in task_heads}
        for j, q in enumerate(ques_list):
            question_type = q['question_type']
            
            if question_type == "descriptive":
                question_dict[question_type]['tokens'].append(q['question'] + " [SEP] " + q['question_subtype'])
                question_dict[question_type]['q_ids'].append(q['question_id'])
                answer_dict[question_type].append(option_id_map[q['answer']])

            elif question_type == "explanatory":                
                question = q['question']
                q_id = q['question_id']                
                for c, choice in enumerate(q['choices']):
                    question_dict[question_type]['tokens'].append(question + " [SEP] " + choice['choice'])
                    question_dict[question_type]['q_ids'].append(q_id)
                    question_dict[question_type]['choice_ids'].append(choice['choice_id'])
                    answer_dict[question_type].append(binary_id_map[choice['answer']])

            elif question_type == "predictive":               
                question = q['question']
                q_id = q['question_id']                
                for c, choice in enumerate(q['choices']):
                    question_dict[question_type]['tokens'].append(question + " [SEP] " + choice['choice'])
                    question_dict[question_type]['q_ids'].append(q_id)
                    question_dict[question_type]['choice_ids'].append(choice['choice_id'])
                    answer_dict[question_type].append(binary_id_map[choice['answer']])
      
            elif question_type == "counterfactual":               
                question = q['question']
                q_id = q['question_id']                
                for c, choice in enumerate(q['choices']):
                    question_dict[question_type]['tokens'].append(question + " [SEP] " + choice['choice'])
                    question_dict[question_type]['q_ids'].append(q_id)
                    question_dict[question_type]['choice_ids'].append(choice['choice_id'])
                    answer_dict[question_type].append(binary_id_map[choice['answer']])
        
        for th in task_heads:
            if answer_dict[th]:
                question_dict[th]['tokens'] = self.tokenizer(question_dict[th]['tokens'], return_tensors='pt', padding=True)
                question_dict[th]['q_ids'] = torch.tensor(question_dict[th]['q_ids'], dtype=torch.long)
                question_dict[th]['choice_ids'] = torch.tensor(question_dict[th]['choice_ids'], dtype=torch.long)
                answer_dict[th] = torch.tensor(answer_dict[th], dtype=torch.long)
                
                if th != 'descriptive':
                    answer_dict[th] = answer_dict[th].float()
            else:
                del question_dict[th]
                del answer_dict[th]
        
        return question_dict, answer_dict
    
    def get_q_batch(self, ques_list):
        '''
        INPUT:
        ques_list: list of question_data dictionary
        OUTPUT: question_dict
        descriptive: 
            question_list: list of <question> [SEP] <question_subtype>
        explanatory:
            question_list: list of <question> [SEP] <choice_k>
        predictive:
            question_list: list of <question> [SEP] <choice_k>
        counterfactual:
            question_list: list of <question> [SEP] <choice_k>
        '''
        
        question_dict = {i:{j:[] for j in ques_dict_keys} for i in task_heads}
        for j, q in enumerate(ques_list):
            question_type = q['question_type']
            
            if question_type == "descriptive":
                question_dict[question_type]['tokens'].append(q['question'] + " [SEP] " + q['question_subtype'])
                question_dict[question_type]['q_ids'].append(q['question_id'])

            elif question_type == "explanatory":                
                question = q['question']
                q_id = q['question_id']                
                for c, choice in enumerate(q['choices']):
                    question_dict[question_type]['tokens'].append(question + " [SEP] " + choice['choice'])
                    question_dict[question_type]['q_ids'].append(q_id)
                    question_dict[question_type]['choice_ids'].append(choice['choice_id'])

            elif question_type == "predictive":               
                question = q['question']
                q_id = q['question_id']                
                for c, choice in enumerate(q['choices']):
                    question_dict[question_type]['tokens'].append(question + " [SEP] " + choice['choice'])
                    question_dict[question_type]['q_ids'].append(q_id)
                    question_dict[question_type]['choice_ids'].append(choice['choice_id'])
      
            elif question_type == "counterfactual":               
                question = q['question']
                q_id = q['question_id']                
                for c, choice in enumerate(q['choices']):
                    question_dict[question_type]['tokens'].append(question + " [SEP] " + choice['choice'])
                    question_dict[question_type]['q_ids'].append(q_id)
                    question_dict[question_type]['choice_ids'].append(choice['choice_id'])
        
        for th in task_heads:
            if question_dict[th]['tokens']:
                question_dict[th]['tokens'] = self.tokenizer(question_dict[th]['tokens'], return_tensors='pt', padding=True)
                question_dict[th]['q_ids'] = torch.tensor(question_dict[th]['q_ids'], dtype=torch.long)
                question_dict[th]['choice_ids'] = torch.tensor(question_dict[th]['choice_ids'], dtype=torch.long)
                
                
            else:
                del question_dict[th]
        
        return question_dict

        
    
class CLEVRERDataset(Dataset):
    def __init__(self, data_dir, frame_dir, tokenizer, test=False):
        # TODO load annotations
        assert os.path.isdir(data_dir)
        assert os.path.isdir(frame_dir)
        
        with open(os.path.join(data_dir, data_dir.split("/")[-1] + ".json"), "r") as f:
            self.json_data = json.load(f)
        self.frame_dir = frame_dir
        self.test = test

        self.process_questions = ProcessQuestions(tokenizer)
        
    
    def __len__(self):
        # get length from directory
        return len(self.json_data)
    
    def __getitem__(self, idx):
        """
        TODO: 
        1. Change here hardcoded path in frame_paths to os.path.join(self.frame_dir, f"sim_{vid_id}", "*.png")
        2. Check normalization mean and std values used in image transform.
        """
        
        vid_json = self.json_data[idx]
        vid_id = vid_json['scene_index']
        
        frame_dir = os.path.join(self.frame_dir, f"sim_{vid_id:05d}", "*.png")
        frame_paths = sorted(glob(frame_dir))
        frames = torch.stack([torchvision.io.read_image(img).float() for img in frame_paths[::5]])
        if self.test:
            ques_dict = self.process_questions.get_q_batch(vid_json['questions'])
            return {'scene_index': vid_id, 'frames': frames, 'ques_dict': ques_dict}
        else:
            ques_dict, ans_dict = self.process_questions.get_qa_batch(vid_json['questions'])
            return {'scene_index': vid_id, 'frames': frames, 'ques_dict': ques_dict, 'ans_dict': ans_dict}
             

class DescriptiveTaskHead(nn.Module):
	
	def __init__(self, n_classes=21, p=0.2, input_dim=768*2):
		super().__init__()
		self.clf = nn.Sequential(
			nn.Linear(input_dim, 1024),
			nn.Dropout(p=0.2),
			nn.ReLU(),
			nn.Dropout(p=0.2),
			nn.Linear(1024, n_classes)
		)

	def forward(self, features):
		return self.clf(features)

class ExplanatoryTaskHead(nn.Module):
	
	def __init__(self, p=0.2, input_dim=768*2):
		super().__init__()
		self.clf = nn.Sequential(
			nn.Linear(input_dim, 1024),
			nn.Dropout(p=0.2),
			nn.ReLU(),
			nn.Dropout(p=0.2),
			nn.Linear(1024, 1),
            nn.Sigmoid()
		)

	def forward(self, features):
		return self.clf(features).reshape(-1)

class PredictiveTaskHead(nn.Module):
	
	def __init__(self, p=0.2, input_dim=768*2):
		super().__init__()
		self.clf = nn.Sequential(
			nn.Linear(input_dim, 1024),
			nn.Dropout(p=0.2),
			nn.ReLU(),
			nn.Dropout(p=0.2),
			nn.Linear(1024, 1),
            nn.Sigmoid()
		)

	def forward(self, features):
		return self.clf(features).reshape(-1)

class CounterfactualTaskHead(nn.Module):
	
	def __init__(self, p=0.2, input_dim=768*2):
		super().__init__()
		self.clf = nn.Sequential(
			nn.Linear(input_dim, 1024),
			nn.Dropout(p=0.2),
			nn.ReLU(),
			nn.Dropout(p=0.2),
			nn.Linear(1024, 1),
            nn.Sigmoid()
		)

	def forward(self, features):
		return self.clf(features).reshape(-1)

class BertCNNModel(nn.Module):
    
    def __init__(self, hidden_size=768):
        super().__init__()
        self.cnn = torchvision.models.resnet50(pretrained=True)
        self.cnn.fc = nn.Identity()
        
        for name, param in self.cnn.named_parameters():
            if not name.startswith('layer4'):
                param.requires_grad = False
        
        self.lstm = nn.LSTM(
                input_size=2048,
                batch_first=True,
                hidden_size=hidden_size,
                num_layers=1
            )
        self.bert = transformers.BertModel.from_pretrained('bert-base-cased')
        
        self.h0 = nn.Parameter(torch.empty(1,hidden_size).normal_(0, 0.1))
        self.c0 = nn.Parameter(torch.empty(1,hidden_size).normal_(0, 0.1))
        
        self.descriptive_head = DescriptiveTaskHead()
        self.explanatory_head = ExplanatoryTaskHead()
        self.predictive_head = PredictiveTaskHead()
        self.counterfactual_head = CounterfactualTaskHead()
        
        self.head_map = {
            'descriptive': self.descriptive_head,
            'predictive': self.predictive_head,
            'explanatory': self.explanatory_head,
            'counterfactual': self.counterfactual_head
        }
        
    def forward(self, example):
        
        N, C, H, W = example['frames'].shape
        # i = 0
        # bs = 8
        # frame_emb = []
        # while (i*bs < N):
        #     frame_emb += [self.cnn(example['frames'][i*bs:(i+1)*bs])]
        #     i += 1
        #     
        # frame_emb = torch.vstack(frame_emb)
        frame_emb = self.cnn(example['frames'])
        frame_encs, (video_enc, last_cell_state) = self.lstm(frame_emb, (self.h0, self.c0))
        
        # faster to batch everything and send, but this works for now
        preds = {}
        for task, ques_data in example['ques_dict'].items():

            bert_output = self.bert(**ques_data['tokens'])

            # feature vector
            features = torch.hstack([video_enc.repeat(bert_output.pooler_output.size(0),1), bert_output.pooler_output])

            preds[task] = self.head_map[task](features)
        
        return preds


def dl_collate_fn(data):
    return data[0]

def ques_to_device(d):
    return {k: {k_dash: v_dash.to(device) for k_dash, v_dash in v.items()} for k,v in d.items()}

def ans_to_device(d):
    return {k: v.to(device) for k,v in d.items()}

def process_example(example, transform):
    return {
        'frames': transform(example['frames'].to(device)),
        'ques_dict': ques_to_device(example['ques_dict']),
        'ans_dict': ans_to_device(example['ans_dict']) if 'ans_dict' in example else None
    }

def load_checkpoint(file_path):

    print("loading model:", file_path)
    model = torch.load(file_path).to(device)
    print("model load successful...")

    return model

class MULTICALSS_PREDS:
    def __init__(self):
        pass
    
    def get_pred(self, output):
        return output.argmax(dim=1)

class BINARY_PREDS:
    def __init__(self):
        pass
    
    def get_pred(self, output):
        return output.round()

def model_eval(model, eval_dl, output_prefix=None, test=False):
    
    pred_fns = {
        'descriptive': MULTICALSS_PREDS(),
        'predictive': BINARY_PREDS(),
        'explanatory': BINARY_PREDS(),
        'counterfactual': BINARY_PREDS()
    }
    
    pred_dict = {k: [] for k in task_heads}
    gold_dict = {k: [] for k in task_heads}
    predictions_json = []
    per_opt_accuracy_dict = {k: -1 for k in task_heads}
    per_ques_accuracy_dict = {k: -1 for k in task_heads}
    per_ques_dict = {k: [] for k in task_heads}

    model.eval()
    with torch.no_grad():
        for example in tqdm(eval_dl):
            scene_pred = {'scene_index': example['scene_index'], 'questions': []}
            
            example = process_example(example, img_transform)

            outputs = model(example)
            
            for task, output in outputs.items():
                pred = pred_fns[task].get_pred(output).detach().to('cpu').tolist()
                pred_dict[task].extend(pred)
                if not test:
                    gold = example['ans_dict'][task].detach().to('cpu').tolist()
                    gold_dict[task].extend(gold)
                
                
                if task == 'descriptive':
                    q_ids = example['ques_dict'][task]['q_ids'].detach().to('cpu').tolist()
                    for i in range(len(q_ids)):
                        scene_pred['questions'].append({"question_id" : q_ids[i], "answer": id_option_map[pred[i]]})
                
                else:
                    q_ids = example['ques_dict'][task]['q_ids']
                    assert len(q_ids) == len(pred)

                    choice_ids = example['ques_dict'][task]['choice_ids'].detach().to('cpu').tolist()
                    unique_q_ids = q_ids.unique().detach().to('cpu').tolist()
                    temp_ques_list = [{"question_id": q, "choices": []} for q in unique_q_ids]
                    temp_dict = {q: [] for q in unique_q_ids}
                    
                    for i, q in enumerate(q_ids):

                        if not test:
                            temp_dict[q.item()].append(pred[i] == gold[i])
                        temp_ques_list[unique_q_ids.index(q)]["choices"].append({"choice_id": choice_ids[i], "answer": id_binary_map[pred[i]]})
                    
                    if not test:
                        for q in unique_q_ids:                        
                            per_ques_dict[task].append(1 if len(Counter(temp_dict[q])) == 1 else 0)
                         
                    scene_pred['questions'].extend(temp_ques_list)

            predictions_json.append(scene_pred)            
    
    if test:
        with open(f"{output_prefix}-test-gold.json", "w+") as file:
            json.dump(gold_dict, file)
        with open(f"{output_prefix}-test-pred.json", "w+") as file:
            json.dump(pred_dict, file)
        with open(f"{output_prefix}-test-predictions.json", "w+") as file:
            json.dump(predictions_json, file)
    else:        
        for th in task_heads:
            per_opt_accuracy_dict[th] = accuracy_score(gold_dict[th], pred_dict[th])  
        # print(per_ques_dict)
        for th in task_heads:
            if th != 'descriptive':
                per_ques_accuracy_dict[th] = Counter(per_ques_dict[th])[1]/len(per_ques_dict[th])

        with open(f"{output_prefix}-pre-opt-val-accuracy.json", "w+") as file:
            json.dump(per_opt_accuracy_dict, file)
        with open(f"{output_prefix}-pre-ques-val-accuracy.json", "w+") as file:
            json.dump(per_ques_accuracy_dict, file)
    
    


if __name__ == '__main__':
    
    img_transform = torchvision.transforms.Compose([torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                     (0.2023, 0.1994, 0.2010))])
    tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-cased')

    test_ds = CLEVRERDataset("../../../data/data/test", "../../../COL775A2_data/frames", tokenizer, test=True)
    eval_ds = CLEVRERDataset("../../../data/data/validation", "../../../COL775A2_data/frames", tokenizer)

    DEBUG = False
    if DEBUG:
        eval_ds.json_data = eval_ds.json_data[:8]

    eval_dl = DataLoader(eval_ds, batch_size=1, collate_fn=dl_collate_fn, shuffle=False, num_workers=8, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=1, collate_fn=dl_collate_fn, shuffle=False, num_workers=8, pin_memory=True)
    
    checkpoints_eval_list = [4, 6, 8]

    for cp in checkpoints_eval_list:        

        model_file = f'../models/baseline-{cp}.pt'
        print("EVALUATING", model_file)

        model = load_checkpoint(model_file)
        
        # model_eval(model, test_dl, f'../results/{model_file.split("/")[-1].split(".")[0]}', test=True)
        model_eval(model, eval_dl, f'../results/{model_file.split("/")[-1].split(".")[0]}')

