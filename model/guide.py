import torch
import torch.nn as nn
import pdb


class ValueGuide(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, cond, t):
        output = self.model(x, cond, t) #COMM :(BS, horizon, state+action dim), dict, (BS,) -> (BS, 1)
        return output.squeeze(dim=-1)

    def gradients(self, x, *args):
        x.requires_grad_()
        y = self(x, *args)
        grad = torch.autograd.grad([y.sum()], [x])[0] # COMM: 这里把不同batch的梯度加起来了
        x.detach()
        return y, grad