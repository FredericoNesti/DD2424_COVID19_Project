from torch import *
from torch.nn import Cross_Entropy_loss
from torch.nn import LogSigmoid
from torch.nn import Module
from torch.nn import Softmax

import torch
from torch import autograd
from torch import nn

'''
class CrossEntropyLoss(Module):
    log_softmax = nn.LogSoftmax()
    def __init__(self, class_weights):
        super().__init__()
        self.class_weights = autograd.Variable(torch.FloatTensor(class_weights).cuda())

    def forward(self, logits, target):
        log_probabilities = self.log_softmax(logits)
        # NLLLoss(x, class) = -weights[class] * x[class]
        return -self.class_weights.index_select(0, target) * log_probabilities.index_select(-1, target).diag()
'''

class SoftmaxCELogits_v2(Module):
    __constants__ = ['reduction']
    # margin = ?
    # size_average = ?
    # reduce = ?
    def __init__(self, margin=0., size_average=None, reduce=None, reduction='mean'):
        super(SoftmaxCELogits_v2, self).__init__(size_average, reduce, reduction)
        self.margin = margin

    def forward(self, input1, target):
        result = Softmax(Cross_Entropy_loss(output = exp(LogSigmoid(input1)), target = target))
        return result
