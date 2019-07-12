import torch
import torch.nn.functional as F

class DoubleCELoss(object):
    def __call__(self, input, target, size_average=True):
        # print(len(input), input[0].size())
        loss = torch.mean(torch.stack(
                    [F.cross_entropy(input[:, i, :], target, size_average=size_average) for i in range(input.shape[1])]))

        return loss