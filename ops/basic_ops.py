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

    dim = 1
    consensus_type = 'avg'

    @staticmethod
    def forward(ctx, input_tensor):
        shape = input_tensor.size()
        if SegmentConsensus.consensus_type == 'avg':
            output = input_tensor.mean(dim=SegmentConsensus.dim, keepdim=True)
        elif SegmentConsensus.consensus_type == 'identity':
            output = input_tensor
        else:
            output = None

        ctx.save_for_backward(input_tensor)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        saved_input, = ctx.saved_tensors
        shape = saved_input.size()
        if SegmentConsensus.consensus_type == 'avg': # avg
            grad_in = grad_output.expand(shape) / float(shape[SegmentConsensus.dim])
        elif SegmentConsensus.consensus_type == 'identity': # identity
            grad_in = grad_output
        else:
            grad_in = None

        return grad_in

    @staticmethod
    def config_dim_type(dim, consensus_type):
        SegmentConsensus.dim = dim
        SegmentConsensus.consensus_type = consensus_type

class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input):
        SegmentConsensus.config_dim_type(self.dim, self.consensus_type)
        SegCons = SegmentConsensus.apply
        return SegCons(input)
