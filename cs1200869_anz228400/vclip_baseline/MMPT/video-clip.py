#!/usr/bin/env python
# coding: utf-8
import os
from glob import glob

import transformers
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

import torchvision
# from skimage import io
import cv2

import copy, json, pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
sns.set_style('whitegrid')
from mmpt.models import MMPTModel
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
ques_dict_keys = ["caps", "cmasks", "q_ids", "choice_ids"]

class ProcessQuestions:
    def __init__(self, tokenizer, aligner):
        self.tokenizer = tokenizer
        self.aligner = aligner
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
                caps, cmasks = self.aligner._build_text_seq(self.tokenizer(q['question'] + " [SEP] " + q['question_subtype'], add_special_tokens=False)["input_ids"])
                # caps, cmasks = caps[None, :], cmasks[None, :]  # bsz=1
                question_dict[question_type]['caps'].append(caps)
                question_dict[question_type]['cmasks'].append(cmasks)

                question_dict[question_type]['q_ids'].append(q['question_id'])
                answer_dict[question_type].append(option_id_map[q['answer']])

            elif question_type == "explanatory":                
                question = q['question']
                q_id = q['question_id']                
                for c, choice in enumerate(q['choices']):
                    caps, cmasks = self.aligner._build_text_seq(self.tokenizer(question + " [SEP] " + choice['choice'], add_special_tokens=False)["input_ids"])
                    # caps, cmasks = caps[None, :], cmasks[None, :]  # bsz=1
                    question_dict[question_type]['caps'].append(caps)
                    question_dict[question_type]['cmasks'].append(cmasks)
                    
                    question_dict[question_type]['q_ids'].append(q_id)
                    question_dict[question_type]['choice_ids'].append(choice['choice_id'])
                    answer_dict[question_type].append(binary_id_map[choice['answer']])

            elif question_type == "predictive":               
                question = q['question']
                q_id = q['question_id']                
                for c, choice in enumerate(q['choices']):
                    caps, cmasks = self.aligner._build_text_seq(self.tokenizer(question + " [SEP] " + choice['choice'], add_special_tokens=False)["input_ids"])
                    # caps, cmasks = caps[None, :], cmasks[None, :]  # bsz=1
                    question_dict[question_type]['caps'].append(caps)
                    question_dict[question_type]['cmasks'].append(cmasks)
                    
                    question_dict[question_type]['q_ids'].append(q_id)
                    question_dict[question_type]['choice_ids'].append(choice['choice_id'])
                    answer_dict[question_type].append(binary_id_map[choice['answer']])
                
#                 question_dict[question_type].append(question + " [SEP] " + q['choices'][0]['choice'] + " [SEP] " + q['choices'][1]['choice'])               
#                 answer_dict[question_type].append(binary_id_map[q['choices'][1]['answer']]) #  binary_id_map[q['choices'][0]['answer']], binary_id_map[q['choices'][1]['answer']]])

            elif question_type == "counterfactual":               
                question = q['question']
                q_id = q['question_id']                
                for c, choice in enumerate(q['choices']):
                    caps, cmasks = self.aligner._build_text_seq(self.tokenizer(question + " [SEP] " + choice['choice'], add_special_tokens=False)["input_ids"])
                    # caps, cmasks = caps[None, :], cmasks[None, :]  # bsz=1
                    question_dict[question_type]['caps'].append(caps)
                    question_dict[question_type]['cmasks'].append(cmasks)
                    
                    question_dict[question_type]['q_ids'].append(q_id)
                    question_dict[question_type]['choice_ids'].append(choice['choice_id'])
                    answer_dict[question_type].append(binary_id_map[choice['answer']])
        
        for th in task_heads:
            if answer_dict[th]:
                
                question_dict[th]['caps'] = torch.stack(question_dict[th]['caps'])
                question_dict[th]['cmasks'] = torch.stack(question_dict[th]['cmasks'])
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
    
    def __init__(self, data_dir, frame_dir, tokenizer, aligner):
        # TODO load annotations
        assert os.path.isdir(data_dir)
        assert os.path.isdir(frame_dir)
        
        with open(os.path.join(data_dir, data_dir.split("/")[-1] + ".json"), "r") as f:
            self.json_data = json.load(f)
        self.frame_dir = frame_dir
        
        self.process_questions = ProcessQuestions(tokenizer, aligner)
        
    
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
        frames = torch.stack([torchvision.io.read_image(img).float() for img in frame_paths])
                
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

class VideoCLIPModel(nn.Module):
    
    def __init__(self, video_clip):
        super().__init__()
        self.video_clip = video_clip
        
        self.max_video_len = video_clip.max_video_len
        self.video_encoder = video_clip.video_encoder
        self.mmmodel = video_clip.model
        
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
        # print(example['frames'].shape)
        bs = 30
        prev = 2
        frame_emb = []
        while (prev+bs <= N):
            frame_emb += [example['frames'][prev:prev+bs].permute(0,2,3,1)]
            prev += bs+2

        video_frames = torch.stack(frame_emb).unsqueeze(0).reshape(1, (N-8)//bs, bs, H, W, C)
        del frame_emb
        
        # you can change max_video_len in how2.yaml file as all videos are sanme length
        '''BELOW SNIPPET IS COPIED FROM MMFUSION FORWARD. DOES THIS COUNT IN PLAGIARISM?'''
        bsz = video_frames.size(0)
        assert bsz == 1, "only bsz=1 is supported now."
        seq_len = video_frames.size(1)
        video_frames = video_frames.view(-1, *video_frames.size()[2:])
        vfeats = self.video_encoder(video_frames.permute(0, 4, 1, 2, 3))
        vfeats = vfeats['video_embedding']
        vfeats = vfeats.view(bsz, seq_len, vfeats.size(-1))
        padding = torch.zeros(
            bsz, self.max_video_len - seq_len, vfeats.size(-1)).to(device)
        
        vfeats = torch.cat([vfeats, padding], dim=1)
        vmasks = torch.cat([
            torch.ones((bsz, seq_len), dtype=torch.bool),
            torch.zeros((bsz, self.max_video_len - seq_len), dtype=torch.bool)
            ],
            dim=1
        ).to(device)        
        
        # faster to batch everything and send, but this works for now
        preds = {}
        for task, ques_data in example['ques_dict'].items():

            caps_all = ques_data['caps']
            cmasks_all = ques_data['cmasks']
            features = []
            for t in range(caps_all.shape[0]):
                output = self.mmmodel(caps_all[t].unsqueeze(0), cmasks_all[t].unsqueeze(0), vfeats, vmasks)
                features.append(torch.hstack([output['pooled_video'][0], output['pooled_text'][0]]))
                                              
            # feature vector
            # assert features.shape[1] == 768*2
            preds[task] = self.head_map[task](torch.stack(features))
        
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

def train(model, train_dl, val_dl, optimizer, scheduler=None, start_epoch = 0, max_epochs=10, patience_lim=2, ckpt_freq=1, ckpt_prefix='./runs/retri/videoclip/models/vc-baseline'):

    best_model = None
    best_val_loss = 10000
    val_losses = []
    train_losses = []
    val_question_count = {t:0 for t in task_heads}
    
    patience = 0
    
    loss_fns = {
        'descriptive': nn.CrossEntropyLoss(),
        'predictive': nn.BCELoss(),
        'explanatory': nn.BCELoss(),
        'counterfactual': nn.BCELoss()
    }
    
    for epoch in range(start_epoch, max_epochs):

        print(f"\n\n|----------- EPOCH: {epoch} -----------|")

        model.train()
        train_loss = 0
        eg_no = 0
        for example in tqdm(train_dl):
            example = process_example(example, img_transform)
                        
            optimizer.zero_grad()
            outputs = model(example)
            loss = 0
            for task, output in outputs.items():
                loss += loss_fns[task](output, example['ans_dict'][task])
            
            train_loss += loss.detach().cpu()
            
            loss.backward()
            optimizer.step()
            eg_no += 1

        train_loss /= len(train_dl)
        train_losses.append(train_loss)

        #Saving Model before validation in case node terminates during val then progress will be lost
        if (epoch+1)%ckpt_freq == 0:
            print('Checkpointing model...')
            torch.save(model, f'{ckpt_prefix}-{epoch+1}.pt')

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for example in tqdm(val_dl):

                example = process_example(example, img_transform)

                outputs = model(example)
                loss = 0
                for task, output in outputs.items():
                    loss += loss_fns[task](output, example['ans_dict'][task])

                val_loss += loss.detach().cpu()

        val_loss /= len(val_dl)
        val_losses.append(val_loss)
            
        if scheduler:
            scheduler.step()
        
        
          
        # if val_loss >= best_val_loss:
        #     if patience >= patience_lim:
        #         break
        #     else:
        #         patience += 1
        # else:
        #     patience = 0
        #     best_val_loss = val_loss
        plt.figure(figsize=(12,8), dpi=150)
        plt.plot(train_losses, label='train')
        plt.plot(val_losses, label='val')
        plt.legend()
        plt.savefig('./runs/retri/videoclip/results/loss_curve.pdf')

    return train_losses, val_losses

if __name__ == '__main__':
    
    n_epochs=13
    start_epoch=0
    img_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224))])

    vclip_model, tokenizer, aligner = MMPTModel.from_pretrained("projects/retri/videoclip/how2.yaml")
    
    LOAD_CHECKPOINT = True
    if LOAD_CHECKPOINT:
        start_epoch = 8
        del vclip_model
        torch.cuda.empty_cache()
        checkpoint = f"./runs/retri/videoclip/models/vc-baseline-{start_epoch}.pt"
        model = torch.load(checkpoint).to(device)
        print("@|---------SUCCESSFULLY LOADED CHECKPOINT", checkpoint)
    else:
        model = VideoCLIPModel(vclip_model).to(device)        
        print("@|---------FRESH MODEL INITIALIZED")

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, verbose=False)

    train_ds = CLEVRERDataset("../../../data/data/train", "../../../COL775A2_data/frames", tokenizer, aligner)
    val_ds = CLEVRERDataset("../../../data/data/validation", "../../../COL775A2_data/frames", tokenizer, aligner)
    val_ds.json_data = val_ds.json_data[:1000]

    DEBUG = False
    if DEBUG:
        train_ds.json_data = train_ds.json_data[:128]
        #train_ds.json_data = [train_ds.json_data[322]]
        val_ds.json_data = val_ds.json_data[:32]
        n_epochs=2

    train_dl = DataLoader(train_ds, batch_size=1, collate_fn=dl_collate_fn, shuffle=True, num_workers=16, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=1, collate_fn=dl_collate_fn, shuffle=False, num_workers=16, pin_memory=True)
    train_losses, val_losses = train(model, train_dl, val_dl, optimizer, start_epoch=start_epoch, max_epochs=n_epochs)

    plt.figure(figsize=(12,8), dpi=150)
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.legend()
    plt.savefig('loss_curve.pdf')

