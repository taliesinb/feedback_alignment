import torch

from torch import nn
from torch.autograd import Function

import torch.nn.functional as F
import numpy as np

# in first forward pass, nothing happens. 
# in first backward pass, gradient is saved. 
class AP1(Function):    

    @staticmethod
    def forward(ctx, input, grad):
        ctx.save_for_backward(grad)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        grad, = ctx.saved_tensors
        grad.copy_(grad_output)
        return grad_output, None

# in second forward pass, gradient is added to activity.
# in second backward pass, new gradient is compared with old gradient.
class AP2(Function):    

    @staticmethod
    def forward(ctx, input, grad, lr, curvature_bias):
        ctx.save_for_backward(grad)
        ctx.lr = lr / grad.numel()
        ctx.curvature_bias = curvature_bias
        return input - ctx.lr * grad

    @staticmethod
    def backward(ctx, grad):
        o_grad, = ctx.saved_tensors
        g_diff = o_grad - grad
        # bias away from zero, limits updates at low curvature
        g_diff += torch.sign(g_diff) * ctx.curvature_bias
        g_diff += torch.sign(g_diff + 1e-6) * ctx.curvature_bias
        n_grad = ctx.lr * (o_grad ** 2) / g_diff
        return n_grad, None, None, None

class SOAP(nn.Module):
    def __init__(self, lr=0.01, curvature_bias=1e-3):
        super().__init__()
        self.lr = lr
        self.is_first = True
        self.eval_count = 0
        self.curvature_bias = curvature_bias

    def forward(self, input):
        if not self.training:
            return input
        self.eval_count += 1
        if self.eval_count < 11:
            return input
        if self.is_first:
            self.is_first = False
            if not hasattr(self, 'grad') or self.grad.shape != input.shape:
                self.register_buffer('grad', torch.zeros(*input.shape))
            return AP1.apply(input, self.grad)
        else:
            self.is_first = True
            return AP2.apply(input, self.grad, self.lr, self.curvature_bias)