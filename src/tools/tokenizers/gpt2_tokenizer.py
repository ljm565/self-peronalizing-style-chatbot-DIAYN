from transformers import GPT2Tokenizer



class CustomGPT2Tokenizer:
    def __init__(self, config):
        self.pretrained_model = config.pretrained_model
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.pretrained_model)
        self.tokenizer.add_special_tokens({'cls_token': '[CLS]', 'sep_token': '[SEP]', 'pad_token': '[PAD]'})
        
        self.pad_token, self.pad_token_id = self.tokenizer.pad_token, self.tokenizer.pad_token_id
        self.cls_token, self.cls_token_id = self.tokenizer.cls_token, self.tokenizer.cls_token_id
        self.sep_token, self.sep_token_id = self.tokenizer.sep_token, self.tokenizer.sep_token_id
        self.eos_token, self.eos_token_id = self.tokenizer.eos_token, self.tokenizer.eos_token_id
        self.unk_token, self.unk_token_id = self.tokenizer.unk_token, self.tokenizer.unk_token_id

        # Add style tokens to vocab
        tokens = ['[Style1]', '[Style2]', '[Style3]']
        self.add_style_tokens(tokens)
        self.style1_token, self.style2_token, self.style_token3 = tokens
        self.style1_token_id, self.style2_token_id, self.style3_token_id = [self.tokenizer.convert_tokens_to_ids(token) for token in tokens]

        self.vocab_size = len(self.tokenizer)


    def tokenize(self, s):
        return self.tokenizer.tokenize(s)


    def encode(self, s):
        return self.tokenizer.encode(s, add_special_tokens=False)


    def decode(self, tok):
        return self.tokenizer.decode(tok)
    

    def add_style_tokens(self, tokens):
        self.tokenizer.add_tokens(tokens, special_tokens=True)