from torch.nn import MultiLabelSoftMarginLoss
from torch.nn import Module
from torch.nn import Softmax

class SoftmaxCELogits_v2(Module):
    __constants__ = ['reduction']
    def __init__(self, margin=0., size_average=None, reduce=None, reduction='mean'):
        super(SoftmaxCELogits_v2, self).__init__(size_average, reduce, reduction)
        self.margin = margin

    def forward(self, input1, target):
        criterion = MultiLabelSoftMarginLoss()
        result = Softmax(criterion(input1, target))
        return result
