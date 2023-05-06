#!/usr/bin/env python
# coding: utf-8
import os
from glob import glob
import argparse

from typing_extensions import *
import transformers
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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

import torchmetrics.functional as tmF

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

config = None

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
        
        question_dict = {i:[] for i in task_heads}
        answer_dict = {i:[] for i in task_heads}
        for j, q in enumerate(ques_list):
            question_type = q['question_type']
            
            if question_type == "descriptive":
                # TODO should we use the sep token here?
                question_dict[question_type].append(q['question'] + " (" + q['question_subtype'] + ")")
                answer_dict[question_type].append(option_id_map[q['answer']])

            else:
                question = q['question']
                i = 97
                answer = [-1]*5
                for c, choice in enumerate(q['choices']):
                    question += f" ({chr(i)}) "
                    question += choice['choice']
                    answer[c] = binary_id_map[choice['answer']]

                answer_dict[question_type].append(answer)
                question_dict[question_type].append(question)

        for th in task_heads:
            if answer_dict[th]:
                question_dict[th] = self.tokenizer(question_dict[th], return_tensors='pt', padding=True)
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
			nn.Dropout(p=p),
			nn.ReLU(),
			nn.Linear(1024, n_classes)
		)

	def forward(self, features):
		return self.clf(features)

class NonDescriptiveTaskHead(nn.Module):
	
	def __init__(self, n_classes=5, p=0.2, input_dim=768*2):
		super().__init__()
		self.clf = nn.Sequential(
			nn.Linear(input_dim, 1024),
			nn.Dropout(p=p),
			nn.ReLU(),
			nn.Linear(1024, n_classes),
            nn.Sigmoid()
		)

	def forward(self, features):
		return self.clf(features)

class BertCNNModel(nn.Module):
    
    # TODO ideas!
    # - Bidirectional LSTM
    # - Attention
    # - Transformer video encoder (too little data?)
    # - Use the object masks
    # - Annotations?
    # - Physics engine - will need a program generator (either LSTM or AR
    #   text model eg GPT2/T5)
    def __init__(self, head_type: Literal['descriptive', 'nondescriptive'], hidden_size=768):
        super().__init__()
        self.cnn = torchvision.models.resnet50(pretrained=True)
        self.cnn.fc = nn.Identity()

        self.lstm = nn.LSTM(
                input_size=2048,
                batch_first=True,
                hidden_size=hidden_size,
                num_layers=1
            )
        self.bert = transformers.BertModel.from_pretrained('bert-base-cased')

        self.h0 = nn.Parameter(torch.empty(1,hidden_size).normal_(0, 0.1))
        self.c0 = nn.Parameter(torch.empty(1,hidden_size).normal_(0, 0.1))
        
        if head_type == 'descriptive':
            self.head = DescriptiveTaskHead(input_dim=768+hidden_size)
        else:
            self.head = NonDescriptiveTaskHead(input_dim=768+hidden_size)

        self.head_type = head_type
        
    def forward(self, frames, tokens):
        
        N, C, H, W = frames.shape
        frame_emb = self.cnn(frames)
        frame_encs, (video_enc, last_cell_state) = self.lstm(frame_emb, (self.h0, self.c0))

        bert_output = self.bert(**tokens)

        features = torch.hstack([video_enc.repeat(bert_output.pooler_output.size(0),1), bert_output.pooler_output])

        return self.head(features)

class BertCNNMetaModel(nn.Module):

    def __init__(self, hidden_size=768):
        super().__init__()
        
        self.descriptive_model = BertCNNModel('descriptive', hidden_size=hidden_size)
        self.nondescriptive_model = BertCNNModel('nondescriptive', hidden_size=hidden_size)

    def forward(self, example):
    
        nondescr_keys = [key for key in example['ques_dict'] if key != 'descriptive']
        all_nondescr_keys = task_heads[1:]
        
        n_pred, n_expl, n_cf = [example['ques_dict'][key]['input_ids'].size(0) if key in example['ques_dict'] else 0 for key in all_nondescr_keys]
        max_sizes = {
            outer_key: max([example['ques_dict'][key][outer_key].size(1) if key in example['ques_dict'] else 0 for key in all_nondescr_keys])
            for outer_key in example['ques_dict']['descriptive']
        }
    
        # itna bada gamble chal gaya yaar ~ aman 2023
        nondescr_pred_tokens = {
            outer_key: torch.vstack([
                F.pad(
                    example['ques_dict'][key][outer_key], 
                    [0, max_sizes[outer_key]-example['ques_dict'][key][outer_key].size(1), 0, 0]
                ) for key in nondescr_keys
            ]) for outer_key in example['ques_dict']['descriptive']
        }

        nondescr_preds = self.nondescriptive_model(example['frames'], nondescr_pred_tokens)

        preds = {
            'descriptive': self.descriptive_model(example['frames'], example['ques_dict']['descriptive']),
            'predictive': nondescr_preds[:n_pred],
            'explanatory': nondescr_preds[n_pred:n_pred+n_expl],
            'counterfactual': nondescr_preds[n_pred+n_expl:]
        }

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


def get_multi_class_preds(preds):
    return preds.argmax(dim=1)

def packed_binary_tensor_to_list(t):
    l = []
    for row in t:
        l.append(t[t >= 0])
    return l

def get_binary_preds(preds):
    pred_list = []
    for row in preds:
        pred_list.append(row[row >= 0].round())
    return pred_list

def masked_binary_cross_entropy(pred, gold):

    mask = (gold >= 0)
    return F.binary_cross_entropy(pred, gold, weight=mask)

def train(model, train_dl, val_dl, optimizer, scheduler=None, max_epochs=10, early_stop=False,
               patience_lim=2, ckpt_freq=1, ckpt_prefix='../models/baseline'):

    best_model = None
    best_val_loss = 10000
    val_losses = {t:[] for t in task_heads}
    train_losses = {t:[] for t in task_heads}
    val_question_count = {t:0 for t in task_heads}
    per_opt_accuracy_dict = {t:[] for t in task_heads}
    
    patience = 0
    
    loss_fns = {
        'descriptive': nn.CrossEntropyLoss(),
        'predictive': masked_binary_cross_entropy,
        'explanatory': masked_binary_cross_entropy,
        'counterfactual': masked_binary_cross_entropy
    }

    pred_fns = {
        'descriptive': get_multi_class_preds,
        'predictive': get_binary_preds,
        'explanatory': get_binary_preds,
        'counterfactual': get_binary_preds
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

                    pred = pred_fns[task](output)

                    if task == 'descriptive':
                        gold = example['ans_dict'][task].cpu()
                        pred_dict[task].append(pred)
                        gold_dict[task].append(gold)
                    else:
                        gold = packed_binary_tensor_to_list(example['ans_dict'][task].cpu())
                        pred_dict[task].extend(pred)
                        gold_dict[task].extend(gold)

        val_loss = {t: l/len(val_dl) for t,l in val_loss.items()}
        val_losses = {task: val_losses[task] + [val_loss[task]] for task in task_heads}

        acc_dict = {}

        for th in task_heads:
            if th == 'descriptive':
                acc = tmF.classification.multiclass_accuracy(
                        torch.hstack(pred_dict[th]), torch.hstack(gold_dict[th]), 21).item()
                print(f"Val Accuracy {th} = {acc}%")
                acc_dict[th] = acc
            else:
                per_q_acc = sum([(a==b).all() for a,b in zip(pred_dict[task], gold_dict[task])])
                per_opt_acc = tmF.classification.binary_accuracy(
                        torch.hstack(pred_dict[th]), torch.hstack(gold_dict[th]))
                print(f"Val Accuracy Per Q {th} = {per_q_acc}%")
                print(f"             Per opt {th} = {per_opt_acc}%")
                acc_dict[th] = {
                    'per_q': per_q_acc,
                    'per_opt': per_opt_acc
                }

        with open(f"../results/acc_dict.json", "w+") as file:
            json.dump(acc_dict, file)

        if scheduler:
            scheduler.step()
        
        if (epoch+1)%ckpt_freq == 0:
            print('Checkpointing model...')
            torch.save(model, f'{ckpt_prefix}-{epoch+1}.pt')
          
        if early_stop:
            if val_loss >= best_val_loss:
                if patience >= patience_lim:
                    break
                else:
                    patience += 1
            else:
                patience = 0
                best_val_loss = val_loss
        
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
            if th == 'descriptive':
                plt.plot(acc_dict[th], label=th)
            else:
                plt.plot(acc_dict[th]['per_q'], label=f'{th}_per_q')
                plt.plot(acc_dict[th]['per_opt'], label=f'{th}_per_opt')
        plt.legend()
        plt.savefig('../results/acc_curve.pdf')

    return train_losses, val_losses

def parse_args():
    
    parser = argparse.ArgumentParser(prog='COL775 A2', 
            description='Visual Question Answering on CLEVRER')
    
    subparsers = parser.add_subparsers(dest='command')
    train_parser = subparsers.add_parser('train')
    #test_parser = subparsers.add_parser('test')

    train_parser.add_argument('train_anno_path', type=str)
    train_parser.add_argument('val_anno_path', type=str)
    train_parser.add_argument('frames_path', type=str)
    train_parser.add_argument('--max-epochs', type=int, default=15)
    train_parser.add_argument('--patience', type=int, default=2)
    train_parser.add_argument('--n-workers', type=int, default=8)
    train_parser.add_argument('--early-stop', action='store_true')
    train_parser.add_argument('--hidden-size', type=int, default=768)
    train_parser.add_argument('--save-name', type=str, default='model_b.pt')
    train_parser.add_argument('--lr', type=float, default=1e-5)
    train_parser.add_argument('--debug', action='store_true')
    train_parser.add_argument('--debug-train-len', type=int, default=128)
    train_parser.add_argument('--debug-val-len', type=int, default=32)

    # test_parser.add_argument('test_ds_path', type=str)
    # test_parser.add_argument('outpath', type=str)
    # test_parser.add_argument('--n-beams', type=int, default=5)
    # test_parser.add_argument('--t5-path', type=str, default='model_b.pt')
    # test_parser.add_argument('--n-workers', type=int, default=2)
    # test_parser.add_argument('--roberta-batch-size', type=int, default=32)
    # test_parser.add_argument('--t5-batch-size', type=int, default=32)
    # test_parser.add_argument('--roberta-path', type=str, default='model_a.pt')
    # test_parser.add_argument('--max-new-tokens', type=int, default=128)

    global config
    config = parser.parse_args()

if __name__ == '__main__':
    
    parse_args()

    img_transform = torchvision.transforms.Compose([torchvision.transforms.Normalize((129.2153, 128.6306, 127.7490),
                                                                         (14.3266, 14.8389, 15.6828))])
    tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-cased')
    
    train_ds = CLEVRERDataset(config.train_anno_path, config.frames_path, tokenizer)
    val_ds = CLEVRERDataset(config.val_anno_path, config.frames_path, tokenizer)
    
    if config.debug:
        train_ds.json_data = train_ds.json_data[:config.debug_train_len]
        val_ds.json_data = val_ds.json_data[:config.debug_val_len]
        config.max_epochs = 2

    train_dl = DataLoader(train_ds, batch_size=1, collate_fn=dl_collate_fn, shuffle=True, num_workers=config.n_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=1, collate_fn=dl_collate_fn, shuffle=False, num_workers=config.n_workers, pin_memory=True)
    
    model = BertCNNMetaModel(hidden_size=config.hidden_size).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    
    train_losses, val_losses = train(model, train_dl, val_dl, optimizer, 
                            early_stop=config.early_stop, max_epochs=config.max_epochs,
                            ckpt_prefix=config.save_name)
