import random
import numpy as np
from copy import deepcopy

import torch
from torch.utils.data import Dataset

from utils import colorstr



def seed_worker(worker_id):  # noqa
    """Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class DLoader(Dataset):
    def __init__(self, data, tokenizer, config, mode):
        # self.data = self.data_augmentation(data) if mode == 'train' else data
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = config.max_len
        self.length = len(self.data)


    def data_augmentation(self, data):
        total_data = []
        for i in range(len(data)):
            prompt = data[i]['prompt']
            preferred_response = data[i]['preferred_response']
            style_id = data[i]['style_id']
            for j in range(len(data)):
                tmp_data = {'prompt': prompt, 'preferred_response': preferred_response, 'non_preferred_response': data[j]['non_preferred_response'], 'style_id': style_id}
                total_data.append(tmp_data)
        return total_data
    

    def get_sep_token_id(self, style_id):
        if style_id == 0:
            return self.tokenizer.style1_token_id
        elif style_id == 1:
            return self.tokenizer.style2_token_id
        return self.tokenizer.style3_token_id
        

    def make_data(self, data):
        prompt, preferred_res, nonpreferred_res, style_id = data['prompt'], data['preferred_response'], data['non_preferred_response'], data['style_id']

        sep_token_id = self.get_sep_token_id(style_id)
        prompt_token = [self.tokenizer.cls_token_id] + self.tokenizer.encode(prompt) + [sep_token_id]
        preferred_prompt_token = prompt_token + self.tokenizer.encode(preferred_res) + [self.tokenizer.eos_token_id]
        nonpreferred_prompt_token = prompt_token + self.tokenizer.encode(nonpreferred_res) + [self.tokenizer.eos_token_id]

        preferred_loss_mask = [0] * len(prompt_token) + [1] * (len(preferred_prompt_token) - len(prompt_token))
        nonpreferred_loss_mask = [0] * len(prompt_token) + [1] * (len(nonpreferred_prompt_token) - len(prompt_token))
        autoregressive_label = [self.tokenizer.pad_token_id] * len(prompt_token) + self.tokenizer.encode(preferred_res) + [self.tokenizer.eos_token_id]

        assert len(preferred_prompt_token) == len(autoregressive_label)
        assert len(preferred_prompt_token) == len(preferred_loss_mask) and len(nonpreferred_prompt_token) == len(nonpreferred_loss_mask)

        return preferred_prompt_token, nonpreferred_prompt_token, preferred_loss_mask, nonpreferred_loss_mask, style_id, autoregressive_label
    

    @staticmethod
    def padding(x:list, length, pad_token_id):
        x = x[:length]
        x += [pad_token_id] * (length - len(x))
        return x


    def __getitem__(self, idx):
        prompt, preferred_response = self.data[idx]['prompt'], self.data[idx]['preferred_response'] + self.tokenizer.eos_token
        preferred_prompt, nonpreferred_prompt, preferred_loss_mask, nonpreferred_loss_mask, style_id, autoregressive_label = self.make_data(self.data[idx])
        
        preferred_prompt, nonpreferred_prompt = \
            self.padding(preferred_prompt, self.max_len, self.tokenizer.pad_token_id), self.padding(nonpreferred_prompt, self.max_len, self.tokenizer.pad_token_id)
        preferred_loss_mask, nonpreferred_loss_mask = \
            self.padding(preferred_loss_mask, self.max_len, 1), self.padding(nonpreferred_loss_mask, self.max_len, 1)
        
        autoregressive_label = self.padding(autoregressive_label, self.max_len, self.tokenizer.pad_token_id)

        batch = {
            'preferred_prompt': torch.tensor(preferred_prompt, dtype=torch.long),
            'nonpreferred_prompt': torch.tensor(nonpreferred_prompt, dtype=torch.long),
            'preferred_loss_mask': torch.tensor(preferred_loss_mask, dtype=torch.long),
            'nonpreferred_loss_mask': torch.tensor(nonpreferred_loss_mask, dtype=torch.long),
            'style_id': torch.tensor(style_id, dtype=torch.long),
            'prompt': prompt,
            'preferred_response': preferred_response,
            'autoregressive_label': torch.tensor(autoregressive_label, dtype=torch.long)
        }
        return batch


    def __len__(self):
        return self.length