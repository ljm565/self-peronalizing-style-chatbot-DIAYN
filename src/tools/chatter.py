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

        # Load trained model checkpoint
        if args.resume_path:
            checkpoints = torch.load(args.resume_path, map_location=self.device)
            self.lm_model.load_state_dict(checkpoints['model'])
            del checkpoints
            torch.cuda.empty_cache()
            gc.collect()


    def make_prompt_input(self, query):
        query = [self.tokenizer.cls_token_id] + self.tokenizer.encode(query) + [self.tokenizer.sep_token_id]
        return query

    