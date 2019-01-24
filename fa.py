import torch

from torch import nn
from torch.nn import init
from torch.autograd import Function

import math
import torch.nn.functional as F
import numpy as np

# Inherit from Function
class FALinearFunction(Function):

    @staticmethod
    def forward(ctx, input, weight, bias, b_matrix):
        ctx.save_for_backward(input, weight, bias, b_matrix)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):

        input, weight, bias, b_matrix = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(b_matrix)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, None

class FALinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

        b_matrix = torch.zeros(out_features, in_features)
        init.xavier_normal_(b_matrix)
        self.register_buffer('b_matrix', b_matrix)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, input):
        return FALinearFunction.apply(input, self.weight, self.bias, self.b_matrix)

    def update_b_matrix(self):
        self.b_matrix.copy_(torch.sign(self.weight))

    def reset_parameters(self):
        init.xavier_normal_(self.weight)
        if self.bias is not None:
            init.uniform_(self.bias, 0, 0)
            #fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            #bound = 1 / math.sqrt(fan_in)
            #init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class BPLinear(nn.Linear):
    def __init__(self, *args, **kw_args):
        super().__init__(*args, **kw_args)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

def update_mlp_bmatrix(net):
    for layer in net:
        if isinstance(layer, FALinear):
            layer.update_b_matrix()