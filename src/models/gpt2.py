from transformers import GPT2LMHeadModel

import torch
import torch.nn as nn

from utils import LOGGER, colorstr



class GPT2(nn.Module):
    def __init__(self, config, tokenizer, style_train_mode='sft', is_training=True):
        super(GPT2, self).__init__()
        self.pretrained_model = config.pretrained_model
        self.model = GPT2LMHeadModel.from_pretrained(self.pretrained_model)
        self.model.resize_token_embeddings(config.vocab_size - config.style_num if is_training and style_train_mode == 'sft'  else config.vocab_size)
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id

        if style_train_mode == 'diayn':
            self.style_num = config.style_num
            self.style_embedding = nn.Embedding(self.style_num, 768)
            self.style_layer = nn.Linear(768, 768)
            
            # Initialize style-related networks
            self._init_style_networks()


    def style_token_resize_embeddings(self, vocab_size):
        self.model.resize_token_embeddings(vocab_size)


    def _init_style_networks(self):
        with torch.no_grad():
            self.style_embedding.weight.data.zero_()
            self.style_layer.weight.copy_(torch.eye(768))
            self.style_layer.bias.zero_()


    def make_mask(self, x):
        pad_mask = torch.where(x==self.pad_token_id, 0, 1)
        return pad_mask


    def forward(self, x, style_id=None):
        pad_mask = self.make_mask(x)
        
        if style_id != None:
            style_embedding = self.style_layer(self.style_embedding(style_id)).unsqueeze(1)
            token_embedding = self.model.transformer.wte(x)
            inputs_embeds = style_embedding + token_embedding
            output = self.model(inputs_embeds=inputs_embeds, attention_mask=pad_mask, output_hidden_states=True)
        else:
            output = self.model(input_ids=x, attention_mask=pad_mask, output_hidden_states=True)
        
        return output.logits, output.hidden_states[-1]
    

    def inference(self, prompt, max_len, device, style_id, style_train_mode='sft'):
        # Prompt setup
        if style_id == 0:
            sep_token_id = self.tokenizer.style1_token_id
        elif style_id == 1:
            sep_token_id = self.tokenizer.style2_token_id
        else:
            sep_token_id = self.tokenizer.style3_token_id

        prompt_token = [self.tokenizer.cls_token_id] + self.tokenizer.encode(prompt) + [sep_token_id]
        prompt_l = len(prompt_token)
        prompt_token = torch.tensor(prompt_token, dtype=torch.long).unsqueeze(0).to(device)
        style_id = style_id.unsqueeze(0)

        while prompt_token.size(1) < max_len:
            output, _ = self.forward(prompt_token, style_id if style_train_mode == 'diayn' else None)
            prompt_token = torch.cat((prompt_token, torch.argmax(output[:, -1], dim=-1).unsqueeze(1)), dim=1)
            if prompt_token[0, -1] == self.tokenizer.eos_token_id:
                break

        response = self.tokenizer.decode(prompt_token[0].tolist()[prompt_l:])
        return response

