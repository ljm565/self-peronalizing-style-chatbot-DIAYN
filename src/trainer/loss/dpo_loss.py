import torch
import torch.nn as nn
import torch.nn.functional as F



class DPOLoss(nn.Module):
    def __init__(self, beta=0.1):
        super(DPOLoss, self).__init__()
        self.beta = beta


    @staticmethod
    def get_log_prob(logits, labels):
        log_probs = F.log_softmax(logits, dim=-1)
        return torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1).mean(-1)
        

    def calculate_loss(self,
                       model_preferred_log_prob,
                       model_nonpreferred_log_prob,
                       ref_preferred_log_prob,
                       ref_nonpreferred_log_prob):
        
        preferred_log_prob = model_preferred_log_prob - ref_preferred_log_prob
        nonpreferred_log_prob = model_nonpreferred_log_prob - ref_nonpreferred_log_prob

        reward_acc = (preferred_log_prob > nonpreferred_log_prob).float().mean(dim=-1)
        reward_margins = (preferred_log_prob - nonpreferred_log_prob).mean(dim=-1)

        loss = -F.logsigmoid(self.beta * (preferred_log_prob - nonpreferred_log_prob)).mean(dim=-1)

        return loss, preferred_log_prob.mean(dim=-1), nonpreferred_log_prob.mean(dim=-1), reward_acc, reward_margins


    def forward(self,
                preferred_token,
                nonpreferred_token,
                model_preferred_logits,
                model_nonpreferred_logits, 
                ref_preferred_logits, 
                ref_nonpreferred_logits):
        
        model_preferred_log_prob = self.get_log_prob(model_preferred_logits, preferred_token)
        model_nonpreferred_log_prob = self.get_log_prob(model_nonpreferred_logits, nonpreferred_token)
        
        ref_preferred_log_prob = self.get_log_prob(ref_preferred_logits, preferred_token)
        ref_nonpreferred_log_prob = self.get_log_prob(ref_nonpreferred_logits, nonpreferred_token)

        loss, preferred_log_prob, nonpreferred_log_prob, reward_acc, reward_margins = self.calculate_loss(
            model_preferred_log_prob=model_preferred_log_prob,
            model_nonpreferred_log_prob=model_nonpreferred_log_prob,
            ref_preferred_log_prob=ref_preferred_log_prob,
            ref_nonpreferred_log_prob=ref_nonpreferred_log_prob,
        )

        return loss, preferred_log_prob, nonpreferred_log_prob, reward_acc, reward_margins