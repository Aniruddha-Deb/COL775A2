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
from skimage import io
import cv2

import copy, json
import numpy as np
from pytorch_memlab import LineProfiler

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

class ProcessQuestions:
    def __init__(self, task_head):
        self.task_head = task_head
        
    def get_qa_batch(self, ques_list):
        #TODO: get qa batches for the current task_head
        
        if self.task_head == "descriptive":
            return self._get_descriptive_qa(ques_list)
        elif self.task_head == "explanatory":
            return self._get_explanatory_qa(ques_list)
        elif self.task_head == "predictive":
            return self._get_predictive_qa(ques_list)
        elif self.task_head == "counterfactual":
            return self._get_counterfactual_qa(ques_list)
        else:
            pass       
        
        return # Tokenized Question answer Pairs
    
    def _get_descriptive_qa(self, ques_list):
        '''
        ques_list: list of question_data dictionary
        question_list: list of <question> [SEP] <question_subtype>
        answer_list: list of respective answer as option_id_map
        '''
        question_list = list()
        answer_list = list()
        
        for j, q in enumerate(ques_list):
            
            if q['question_type'] == self.task_head:                
                question = q['question']
                question_subtype = q['question_subtype']
                answer = q['answer']
                
                question_list.append(question + " [SEP] " + question_subtype)
                answer_list.append(option_id_map[answer])
        
        return tokenizer(question_list, return_tensors='pt', padding=True), torch.tensor(answer_list)
    
    def _get_explanatory_qa(self, ques_list):
        '''
        ques_list: list of question_data dictionary
        question_list: list of <question> [SEP] <choice_k>
        answer_list: list of respective answer as binary_id_map correct = 1 / wrong = 0
        '''
        question_list = list()
        answer_list = list()
        
        for j, q in enumerate(ques_list):
            
            if q['question_type'] == self.task_head:                
                question = q['question']
                
                for c, choice in enumerate(q['choices']):
                    question_list.append(question + " [SEP] " + choice['choice'])
                    answer_list.append(binary_id_map[choice['answer']])
            
        if len(question_list) > 0:
            return tokenizer(question_list, return_tensors='pt', padding=True), torch.tensor(answer_list).float()
        else:
            return torch.LongTensor([]), torch.LongTensor([]) # HANDLE THIS IN TRAINING
        
    def _get_predictive_qa(self, ques_list):
        '''
        ques_list: list of question_data dictionary
        question_list: list of <question> [SEP] <choice_k>
        answer_list: list of respective answer as binary_id_map correct = 1 / wrong = 0
        '''
        question_list = list()
        answer_list = list()
        
        for j, q in enumerate(ques_list):
            
            if q['question_type'] == self.task_head:                
                question = q['question']
                
                for c, choice in enumerate(q['choices']):
                    question_list.append(question + " [SEP] " + choice['choice'])
                    answer_list.append(binary_id_map[choice['answer']])
                    
#         print(len(question_list), question_list, len(answer_list), answer_list)
        if len(question_list) > 0:
            return tokenizer(question_list, return_tensors='pt', padding=True), torch.tensor(answer_list)
        else:
            return torch.LongTensor([]), torch.LongTensor([]) # HANDLE THIS IN TRAINING
    
    def _get_counterfactual_qa(self, ques_list):
        '''
        ques_list: list of question_data dictionary
        question_list: list of <question> [SEP] <choice_k>
        answer_list: list of respective answer as binary_id_map correct = 1 / wrong = 0
        '''
        question_list = list()
        answer_list = list()
        
        for j, q in enumerate(ques_list):
            
            if q['question_type'] == self.task_head:                
                question = q['question']
                
                for c, choice in enumerate(q['choices']):
                    question_list.append(question + " [SEP] " + choice['choice'])
                    answer_list.append(binary_id_map[choice['answer']])
            
        if len(question_list) > 0:
            return tokenizer(question_list, return_tensors='pt', padding=True), torch.tensor(answer_list).float()
        else:
            return torch.LongTensor([]), torch.LongTensor([]) # HANDLE THIS IN TRAINING
        
class CLEVRERDataset(Dataset):
    
    def __init__(self, data_dir, frame_dir, task_head='descriptive', img_transform=None):
        # TODO load annotations
        assert os.path.isdir(data_dir)
        assert os.path.isdir(frame_dir)
        
        with open(os.path.join(data_dir, data_dir.split("/")[-1] + ".json"), "r") as f:
            self.json_data = json.load(f)
        self.frame_dir = frame_dir
        self.task_head = task_head
        
        # self.img_transform = img_transform
        self.process_questions = ProcessQuestions(task_head)
        
    
    def __len__(self):
        # get length from directory
        return len(self.json_data)
    
    def __getitem__(self, idx):
        """
        TODO: 
        1. Change here hardcoded path in frame_paths to os.path.join(self.frame_dir, f"sim_{vid_id}", "*.png")
        2. Check normalization mean and std values used in image transform
        3. Add tokenized questions + concatinate options (where applicable) and answer token
        4. There are certain videos for which there are no predictive questions. Handle it during training loop
            coz dataloader will return torch.LongTensor([]), torch.LongTensor([]). This may happen for explanatory and counterfactual questions as well.
        """
        
        vid_json = self.json_data[idx]
        vid_id = vid_json['scene_index']
        frame_dir = os.path.join(self.frame_dir, f"sim_{vid_id:05d}", "*.png")
        frame_paths = glob(frame_dir)
        frames = torch.stack([torchvision.io.read_image(img).float() for img in frame_paths])
                
        ques_toks, answers = self.process_questions.get_qa_batch(vid_json['questions'])
#         answers = torch.LongTensor(answers)
        return {'frames': frames, 'ques_toks': ques_toks, 'answers': answers}
    
def get_task_head(epoch):
    task_head = ''
    for t in range(4):
        if (epoch+1) % (t+1) == 0:
            task_head = task_heads[t]
    return task_head
        

class PositionalEmbedding(nn.Module):

    def __init__(self, dim_y, dim_x, max_len=300, p=0.2):
        super().__init__()
        
        # Compute the positional encodings once in log space.
        self.pe = torch.zeros(max_len, dim_y, dim_x)

        pos = torch.arange(0,max_len).unsqueeze(1).unsqueeze(2)

        div_term_x = torch.exp(torch.arange(0, dim_x, 2).expand((dim_y//2,dim_x//2)) * -(np.log(10000.0) / dim_x))
        div_term_y = torch.exp(torch.arange(0, dim_y, 2).unsqueeze(1).expand((dim_y//2,dim_x//2)) * -(np.log(10000.0) / dim_y))

        self.pe[:, 0::2, 0::2] = (torch.sin(pos * div_term_x) + torch.sin(pos * div_term_y))/2
        self.pe[:, 1::2, 1::2] = (torch.cos(pos * div_term_x) + torch.cos(pos * div_term_y))/2
        self.pe = self.pe.unsqueeze(0).repeat(3,1,1,1).transpose(0,1).reshape((3*max_len,dim_y,dim_x))
        self.pe = self.pe * 0.05
        # assert((self.pe[0] == self.pe[1]) & (self.pe[1] == self.pe[2])).all()
        # self.pe = self.pe.unsqueeze(0)
        # self.register_buffer("pe", self.pe)

        # self.dropout = nn.Dropout(p)

    def forward(self, x):
        return x+self.pe[:x.shape[0],:,:].requires_grad_(False)

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
        
        # self.pos_emb = PositionalEmbedding(320, 480)
        
        self.h0 = nn.Parameter(torch.empty(1,hidden_size).normal_(0, 0.1))
        self.c0 = nn.Parameter(torch.empty(1,hidden_size).normal_(0, 0.1))
        
    def forward(self, frames, tokens):
        
        # frames = (n_frames, channels, h, w)
        N, C, H, W = frames.shape
        i = 0
        bs = 8
        frame_emb = []
        while (i*bs < N):
            frame_emb += [self.cnn(frames[i*bs:(i+1)*bs])]
            i += 1
            
        frame_emb = torch.vstack(frame_emb)
        frame_encs, (video_enc, last_cell_state) = self.lstm(frame_emb, (self.h0, self.c0))
        
        bert_output = self.bert(**tokens)
        
        # feature vector - 1768-dimensional
        features = torch.hstack([video_enc.repeat(bert_output.pooler_output.size(0),1), bert_output.pooler_output])
        
        return features

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
		return self.clf(features).squeeze()

class PredictiveTaskHead(nn.Module):
	
	def __init__(self, n_classes=2, p=0.2, input_dim=768*2):
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
		return self.clf(features).squeeze()

def dl_collate_fn(data):
    return data[0]

def train(model, heads, train_dl, val_dl, optimizer, scheduler=None, max_epochs=4, patience_lim=2):

    best_model = None
    best_val_loss = 10000
    val_losses = {t:[] for t in task_heads}
    train_losses = {t:[] for t in task_heads}
    val_question_count = {t:0 for t in task_heads}
    
    patience = 0
    
    loss_fns = {
        'descriptive': nn.CrossEntropyLoss(reduction='sum'),
        'predictive': nn.CrossEntropyLoss(reduction='sum'),
        'explanatory': nn.BCELoss(reduction='sum'),
        'counterfactual': nn.BCELoss(reduction='sum')
    }
    
    for epoch in range(max_epochs):

        print(f"\n\n|----------- EPOCH: {epoch} -----------|")
        for task in task_heads:
            train_dl.dataset.task_head = task
            train_dl.dataset.process_questions = ProcessQuestions(task)
            val_dl.dataset.task_head = task
            val_dl.dataset.process_questions = ProcessQuestions(task)
            print(f"  Training for {task} task head.")
            n_train_questions = 0
            n_val_questions = 0
            
            train_loss = 0
            model.train()
            for batch in tqdm(train_dl):
                if len(batch['answers']) == 0:
                    continue
                frames = img_transform(batch['frames'].to(device)) #torch.Size([128, 3, 320, 480])
                ques_toks = batch['ques_toks'].to(device) # returns [N, k_question || choice, toks_len] for bert we may need to squeeze ques_toks['input_ids'],ques_toks['token_type_ids'], ques_toks['attention_mask'] 
                answers = batch['answers'].to(device)
                # print("\nSHAPES frames: {}, ques_toks['input_ids']: {}, ques_toks['token_type_ids']: {}, ques_toks['attention_mask']: {}, answers: {}".format(frames.shape, ques_toks['input_ids'].shape, ques_toks['token_type_ids'].shape, 
                #       ques_toks['attention_mask'].shape, answers))
                n_questions = ques_toks['input_ids'].size(1)
                n_train_questions += n_questions
            
                optimizer.zero_grad()
                outputs = model(frames, ques_toks)
                outputs = heads[task](outputs)
                
                loss = loss_fns[task](outputs, answers)
                mean_loss = loss / n_questions
                mean_loss.backward()
                optimizer.step()

                train_loss += loss.detach()
                
            train_loss = train_loss.cpu() / n_train_questions
            print(f'  {task} Train Loss: {train_loss}')
            train_losses[task].append(train_loss)

            val_loss = 0
            model.eval()
            for batch in tqdm(val_dl):
                if len(batch['answers']) == 0:
                    continue
                frames = img_transform(batch['frames'].to(device)) #torch.Size([128, 3, 320, 480])
                ques_toks = batch['ques_toks'].to(device) # returns [N, k_question || choice, toks_len] for bert we may need to squeeze ques_toks['input_ids'],ques_toks['token_type_ids'], ques_toks['attention_mask'] 
                
                answers = batch['answers'].to(device)
                n_questions = ques_toks['input_ids'].size(1)
                n_val_questions += n_questions
            
                outputs = model(frames, ques_toks)
                outputs = heads[task](outputs)
                loss = loss_fns[task](outputs, answers)

                val_loss += loss.detach()
            
            val_question_count[task] = n_val_questions
                
            val_loss = val_loss.cpu() / n_val_questions
            print(f'  {task} Val Loss: {val_loss}')
            val_losses[task].append(val_loss)
            print('')

        if scheduler:
            scheduler.step()

        # early stopping
        agg_val_loss = sum([val_losses[t][-1]*val_question_count[t] for t in task_heads])/sum(val_question_count.values())
        if agg_val_loss >= best_val_loss:
            if patience >= patience_lim:
                break
            else:
                patience += 1
        else:
            patience = 0
            best_val_loss = agg_val_loss
            best_model = copy.deepcopy(model)
            best_model = best_model.cpu()
    
    return best_model, (train_losses, val_losses)

if __name__ == "__main__":
    # ## Training
    img_transform = torchvision.transforms.Compose([torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                         (0.2023, 0.1994, 0.2010))])
    
    train_ds = CLEVRERDataset(data_dir="../../data/data/train", frame_dir="../../clevrer_code/frames")
    val_ds = CLEVRERDataset(data_dir="../../data/data/validation", frame_dir="../../clevrer_code/frames")
    
    DEBUG = True
    if DEBUG:
        train_ds.json_data = train_ds.json_data[:4]
        val_ds.json_data = val_ds.json_data[:2]
    
    train_dl = DataLoader(train_ds, batch_size=1, collate_fn=dl_collate_fn, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=1, collate_fn=dl_collate_fn, shuffle=True, num_workers=4)
    
    model = BertCNNModel().to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-cased')
    
    heads = {
        'descriptive': DescriptiveTaskHead(),
        'predictive': PredictiveTaskHead(),
        'explanatory': ExplanatoryTaskHead(),
        'counterfactual': CounterfactualTaskHead()
    }
    
    heads = {a:b.to(device) for a,b in heads.items()}
    
    optimizer = optim.Adam([{'params': model.parameters(), 'lr': 1e-5}] + [{'params': head.parameters(), 'lr': 1e-4} for head in heads.values()])
    
    best_model, (train_losses, val_losses) = train(model, heads, train_dl, val_dl, optimizer, max_epochs=1)
    torch.save(best_model, '../models/bert_cnn_baseline.pt')
    torch.save(heads, '../models/heads.pt')
