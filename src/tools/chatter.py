import gc
import numpy as np

import torch
import torch.nn as nn

from models import GPT2
from tools.tokenizers import CustomGPT2Tokenizer



class Chatter:
    def __init__(self, config, device, resume_path=None):
        # Initialize config, tokenizer, and model
        self.config = config
        self.device = torch.device('cpu') if device == 'cpu' else torch.device(f'cuda:{device}')
        self.tokenizer = CustomGPT2Tokenizer(self.config)
        self.config.vocab_size = self.tokenizer.vocab_size
        self.style_train_mode = self.config.style_train_mode
        self.lm_model = GPT2(
            self.config,
            self.tokenizer,
            self.style_train_mode,
            is_training=False
        ).to(self.device)
        self.max_len = self.config.max_len
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # Load trained model checkpoint
        if resume_path != None:
            checkpoints = torch.load(resume_path, map_location=self.device)
            self.lm_model.load_state_dict(checkpoints['model'])
            del checkpoints
            torch.cuda.empty_cache()
            gc.collect()


    def make_prompt_input(self, query):
        query = [self.tokenizer.cls_token_id] + self.tokenizer.encode(query) + [self.tokenizer.sep_token_id]
        query = torch.tensor(query, dtype=torch.long).to(self.device)
        return query.unsqueeze(0)


    def generate(self, prompt, style_id=-1):
        if isinstance(style_id, int):
            style_id = torch.tensor(style_id).to(self.device)

        response = self.lm_model.inference(
            prompt=prompt,
            max_len=self.max_len,
            device=self.device,
            style_id=style_id,
            style_train_mode=self.style_train_mode,
            include_end_token=False,
        )
        return prompt, response
    

    def calculate_ppl(self, prompt, response, style_id=-1):
        if isinstance(style_id, int):
            style_id = torch.tensor(style_id).to(self.device)

        logits, label = self.lm_model.calculate_logits(prompt, response, self.device, style_id, self.style_train_mode)
        loss = self.criterion(logits[:, :-1, :].reshape(-1, logits.size(-1)), label[:, 1:].reshape(-1))
        return np.exp(loss.item())