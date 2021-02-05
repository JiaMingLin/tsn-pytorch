import torch
import math


class Identity(torch.nn.Module):
    def forward(self, input):
        return input

class SegmentConsensus(torch.autograd.Function):

    # def __init__(self, consensus_type, dim=1):
    #     self.consensus_type = consensus_type
    #     self.dim = dim
    #     self.shape = None

    consensus_type_mapping = {
        'avg': 1,
        'identity': 2
    }
    @staticmethod
    def forward(ctx, input_tensor, dim, consensus_type='avg'):
        shape = input_tensor.size()
        if consensus_type == 'avg':
            output = input_tensor.mean(dim=dim, keepdim=True)
        elif consensus_type == 'identity':
            output = input_tensor
        else:
            output = None

        ctx.save_for_backward(input_tensor, torch.tensor(dim), torch.tensor(consensus_type_mapping[consensus_type]))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        saved_input, dim, cons_type = ctx.saved_tensors
        shape = saved_input.size()
        if cons_type == 1:
            grad_in = grad_output.expand(shape) / float(shape[dim.item()])
        elif cons_type == 2:
            grad_in = grad_output
        else:
            grad_in = None

        return grad_in

class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input):
        SegCons = SegmentConsensus.apply
        return SegCons(input, self.dim, consensus_type = self.consensus_type)
