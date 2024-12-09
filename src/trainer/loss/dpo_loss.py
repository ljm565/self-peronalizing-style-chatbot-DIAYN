import torch
import torch.nn as nn
import torch.nn.functional as F



class DPOLoss(nn.Module):
    def __init__(self, beta=0.1):
        super(DPOLoss, self).__init__()
        self.beta = beta


    @staticmethod
    def get_log_prob(logits, labels, mask=None):
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        
        if mask == None:
            return torch.gather(log_probs, -1, labels[:, 1:].unsqueeze(-1)).squeeze(-1).mean(-1)
        
        mask = mask[:, 1:]
        per_token_logps = torch.gather(log_probs, -1, labels[:, 1:].unsqueeze(-1)).squeeze(-1)
        per_token_logps[~mask.bool()] = 0
        return per_token_logps.sum(-1) / mask.sum(-1)


    def calculate_loss(self,
                       model_preferred_log_prob,
                       model_nonpreferred_log_prob,
                       ref_preferred_log_prob,
                       ref_nonpreferred_log_prob):
        # For cacluate loss
        model_logratios = model_preferred_log_prob - model_nonpreferred_log_prob
        ref_logratios = ref_preferred_log_prob - ref_nonpreferred_log_prob
        loss = -F.logsigmoid(self.beta * (model_logratios - ref_logratios)).mean(dim=-1)
        
        #  For calcuate metrics
        preferred_log_prob = model_preferred_log_prob - ref_preferred_log_prob
        nonpreferred_log_prob = model_nonpreferred_log_prob - ref_nonpreferred_log_prob

        reward_acc = (preferred_log_prob > nonpreferred_log_prob).float().mean(dim=-1)
        reward_margins = (preferred_log_prob - nonpreferred_log_prob).mean(dim=-1)

        return loss, preferred_log_prob.mean(dim=-1), nonpreferred_log_prob.mean(dim=-1), reward_acc, reward_margins


    def forward(self,
                preferred_token,
                nonpreferred_token,
                model_preferred_logits,
                model_nonpreferred_logits, 
                ref_preferred_logits, 
                ref_nonpreferred_logits,
                preferred_loss_mask=None,
                nonpreferred_loss_mask=None):
        
        model_preferred_log_prob = self.get_log_prob(model_preferred_logits, preferred_token, preferred_loss_mask)
        model_nonpreferred_log_prob = self.get_log_prob(model_nonpreferred_logits, nonpreferred_token, nonpreferred_loss_mask)
        
        ref_preferred_log_prob = self.get_log_prob(ref_preferred_logits, preferred_token, preferred_loss_mask)
        ref_nonpreferred_log_prob = self.get_log_prob(ref_nonpreferred_logits, nonpreferred_token, nonpreferred_loss_mask)

        loss, preferred_log_prob, nonpreferred_log_prob, reward_acc, reward_margins = self.calculate_loss(
            model_preferred_log_prob=model_preferred_log_prob,
            model_nonpreferred_log_prob=model_nonpreferred_log_prob,
            ref_preferred_log_prob=ref_preferred_log_prob,
            ref_nonpreferred_log_prob=ref_nonpreferred_log_prob,
        )

        return loss, preferred_log_prob, nonpreferred_log_prob, reward_acc, reward_margins