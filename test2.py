import torch
import torch.nn.functional as F




logits = torch.randn((2, 5, 10))
tokens = torch.randint(low=0, high=9, size=(2, 5))


log_probs = F.log_softmax(logits, dim=-1)

tmp = torch.gather(logits, -1, tokens.unsqueeze(-1))

print(tokens)
print(logits[0])

print()
print(tmp)
print(tmp.size())
print(tmp.squeeze(-1).mean(-1))
print(tmp.squeeze(-1).mean(-1).size())


tmp2 = torch.randn((2))
print(tmp2)
print(tmp.squeeze(-1).mean(-1) > tmp2)