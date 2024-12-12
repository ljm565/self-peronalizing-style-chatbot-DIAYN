from transformers import BertModel

import torch
import torch.nn as nn


# BERT
class BERT(nn.Module):
    def __init__(self, style_num):
        super(BERT, self).__init__()
        self.style_num = style_num
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.model.config.hidden_size, self.style_num)


    def forward(self, inputs_embeds):
        output = self.model(inputs_embeds=inputs_embeds)
        output = self.fc(output['pooler_output'])
        output = torch.softmax(output, dim=-1)

        return output