#!/usr/bin/env python
# coding: utf-8
import os
from glob import glob

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
from sklearn.metrics import accuracy_score
sns.set_style('whitegrid')

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
ques_dict_keys = ["tokens", "q_ids", "choice_ids"]

class ProcessQuestions:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        pass
        
    def get_qa_batch(self, ques_list):
        #TODO: get qa batches for the current task_head
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
            question_list: list of <question> [SEP] <choice_1 >  [SEP] <choice_2>
            answer_list: OHE vector of respective answer as binary_id_map correct = 1 / wrong = 0
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
                
#                 question_dict[question_type].append(question + " [SEP] " + q['choices'][0]['choice'] + " [SEP] " + q['choices'][1]['choice'])               
#                 answer_dict[question_type].append(binary_id_map[q['choices'][1]['answer']]) #  binary_id_map[q['choices'][0]['answer']], binary_id_map[q['choices'][1]['answer']]])

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
        
class CLEVRERDataset(Dataset):
    
    def __init__(self, data_dir, frame_dir, tokenizer):
        # TODO load annotations
        assert os.path.isdir(data_dir)
        assert os.path.isdir(frame_dir)
        
        with open(os.path.join(data_dir, data_dir.split("/")[-1] + ".json"), "r") as f:
            self.json_data = json.load(f)
        self.frame_dir = frame_dir
        
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
                
        ques_dict, ans_dict = self.process_questions.get_qa_batch(vid_json['questions'])

        return {'frames': frames, 'ques_dict': ques_dict, 'ans_dict': ans_dict}

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


        # UPDATE: REMOVED MODEL FREEZE
        # for name, param in self.cnn.named_parameters():
        #     if not name.startswith('layer4'):
        #         param.requires_grad = False
        
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
        'ans_dict': ans_to_device(example['ans_dict'])
    }

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

def train(model, train_dl, val_dl, optimizer, scheduler=None, max_epochs=10, patience_lim=2, ckpt_freq=1, ckpt_prefix='../models/baseline'):

    best_model = None
    best_val_loss = 10000
    10000
    val_losses = {t:[] for t in task_heads}
    train_losses = {t:[] for t in task_heads}
    val_question_count = {t:0 for t in task_heads}
    per_opt_accuracy_dict = {t:[] for t in task_heads}
    
    patience = 0
    
    loss_fns = {
        'descriptive': nn.CrossEntropyLoss(),
        'predictive': nn.BCELoss(),
        'explanatory': nn.BCELoss(),
        'counterfactual': nn.BCELoss()
    }

    pred_fns = {
        'descriptive': MULTICALSS_PREDS(),
        'predictive': BINARY_PREDS(),
        'explanatory': BINARY_PREDS(),
        'counterfactual': BINARY_PREDS()
    }
    
    for epoch in range(max_epochs):

        print(f"\n\n|----------- EPOCH: {epoch} -----------|")

        model.train()        
        train_loss = {t:0 for t in task_heads}
        eg_no = 0
        for example in tqdm(train_dl):
            
            example = process_example(example, img_transform)

            optimizer.zero_grad()
            outputs = model(example)
            loss = 0
            for task, output in outputs.items():
                task_loss = loss_fns[task](output, example['ans_dict'][task])
                loss += task_loss
                train_loss[task] += task_loss.item()
            
            # train_loss += loss.detach().cpu()
            
            loss.backward()
            optimizer.step()
            eg_no += 1

        train_loss = {t: l/len(train_dl) for t,l in train_loss.items()}
        train_losses = {task: train_losses[task] + [train_loss[task]] for task in task_heads}
 
        model.eval()
        val_loss = {t:0 for t in task_heads}
        pred_dict = {k: [] for k in task_heads}
        gold_dict = {k: [] for k in task_heads}
        with torch.no_grad():
            for example in tqdm(val_dl):

                example = process_example(example, img_transform)

                outputs = model(example)
                loss = 0
                for task, output in outputs.items():
                    
                    task_loss = loss_fns[task](output, example['ans_dict'][task])
                    loss += task_loss
                    val_loss[task] += task_loss.item()

                    pred = pred_fns[task].get_pred(output).detach().to('cpu').tolist()
                    pred_dict[task].extend(pred)
                    gold = example['ans_dict'][task].detach().to('cpu').tolist()
                    gold_dict[task].extend(gold)

                # val_loss += loss.detach().cpu()

        val_loss = {t: l/len(val_dl) for t,l in val_loss.items()}
        val_losses = {task: val_losses[task] + [val_loss[task]] for task in task_heads}
        for th in task_heads:
            accu = round(accuracy_score(gold_dict[th], pred_dict[th]), 3)
            per_opt_accuracy_dict[th].append(accu)
            print(f"Val Accuracy {th} = {accu}%")

        with open(f"../results/all-epochs-pre-opt-val-accuracy.json", "w+") as file:
            json.dump(per_opt_accuracy_dict, file)

        if scheduler:
            scheduler.step()
        
        if (epoch+1)%ckpt_freq == 0:
            print('Checkpointing model...')
            torch.save(model, f'{ckpt_prefix}-{epoch+1}.pt')
          
        # if val_loss >= best_val_loss:
        #     if patience >= patience_lim:
        #         break
        #     else:
        #         patience += 1
        # else:
        #     patience = 0
        #     best_val_loss = val_loss
        
        plt.figure(figsize=(12,8), dpi=150)
        for th in task_heads:
            plt.plot(train_losses[th], label=th)
        plt.legend()
        plt.savefig('../results/train_loss_curve.pdf')

        plt.figure(figsize=(12,8), dpi=150)
        for th in task_heads:
            plt.plot(val_losses[th], label=th)
        plt.legend()
        plt.savefig('../results/val_loss_curve.pdf')

        plt.figure(figsize=(12,8), dpi=150)
        for th in task_heads:
            plt.plot(per_opt_accuracy_dict[th], label=th)
        plt.legend()
        plt.savefig('../results/pre_opt_accu_curve.pdf')

    return train_losses, val_losses

if __name__ == '__main__':
    
    n_epochs=15
    img_transform = torchvision.transforms.Compose([torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                         (0.2023, 0.1994, 0.2010))])
    tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-cased')
    
    train_ds = CLEVRERDataset("../../../data/data/train", "../../../COL775A2_data/frames", tokenizer)
    val_ds = CLEVRERDataset("../../../data/data/validation", "../../../COL775A2_data/frames", tokenizer)
    val_ds.json_data = val_ds.json_data[:2000]
    
    DEBUG = False
    if DEBUG:
        train_ds.json_data = train_ds.json_data[:32]
        #train_ds.json_data = [train_ds.json_data[322]]
        val_ds.json_data = val_ds.json_data[:8]
        n_epochs=4

    train_dl = DataLoader(train_ds, batch_size=1, collate_fn=dl_collate_fn, shuffle=True, num_workers=8, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=1, collate_fn=dl_collate_fn, shuffle=False, num_workers=8, pin_memory=True)
    
    model = BertCNNModel().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, verbose=False)
    
    train_losses, val_losses = train(model, train_dl, val_dl, optimizer, max_epochs=n_epochs)
    
    # plt.figure(figsize=(12,8), dpi=150)
    # plt.plot(train_losses, label='train')
    # plt.plot(val_losses, label='val')
    # plt.legend()
    # plt.savefig('loss_curve.pdf')

