import gc
from sconf import Config

import torch

from models import GPT2
from tools.tokenizers import CustomGPT2Tokenizer



class Chatter:
    def __init__(self, args):
        # Initialize config, tokenizer, and model
        self.config = Config(args.config)
        self.device = torch.device('cpu') if args.device == 'cpu' else torch.device(f'cuda:{args.device}')
        self.tokenizer = CustomGPT2Tokenizer(self.config)
        self.config.vocab_size = self.tokenizer.vocab_size
        self.lm_model = GPT2(self.config, self.tokenizer).to(self.device)
        self.max_len = self.config.max_len

        # Load trained model checkpoint
        if args.resume_path:
            checkpoints = torch.load(args.resume_path, map_location=self.device)
            self.lm_model.load_state_dict(checkpoints['model'])
            del checkpoints
            torch.cuda.empty_cache()
            gc.collect()


    def make_prompt_input(self, query):
        query = [self.tokenizer.cls_token_id] + self.tokenizer.encode(query) + [self.tokenizer.sep_token_id]
        query = torch.tensor(query, dtype=torch.long).to(self.device)
        return query.unsqueeze(0)


    def generate(self, query):
        answer = []
        query_tokens = self.make_prompt_input(query)
        while 1:
            output = self.lm_model(query_tokens)
            pred_token = torch.argmax(output[:, -1], dim=-1)
            answer.append(pred_token.item())
            query_tokens = torch.cat((query_tokens, pred_token.unsqueeze(1)), dim=1)

            if pred_token == self.tokenizer.sep_token_id:
                answer.pop()
                break
            elif pred_token == self.tokenizer.eos_token_id:
                answer.pop()
                break
            
            if query_tokens.size(1) >= self.max_len:
                break
            
        answer = self.tokenizer.decode(answer)
        return query, answer
    