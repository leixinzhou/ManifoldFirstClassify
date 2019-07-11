import torch
import torch.nn.functional as F

class DoubleCELoss(object):
    def __call__(self, input1, input2, target, size_average=True):
        loss = F.cross_entropy(input1, target, size_average=size_average) + \
                    F.cross_entropy(input2, target, size_average=size_average)
        return loss