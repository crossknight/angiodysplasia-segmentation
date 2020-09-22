from torch import nn
from torch.nn import functional as F
import torch

def soft_jaccard(outputs, targets):
    eps = 1e-15
    jaccard_target = (targets == 1).float()
    jaccard_output = torch.sigmoid(outputs)

    intersection = (jaccard_output * jaccard_target).sum()
    union = jaccard_output.sum() + jaccard_target.sum()
    return intersection / (union - intersection + eps)


class LossBinary:
    """
    Loss defined as BCE - log(soft_jaccard)

    Vladimir Iglovikov, Sergey Mushinskiy, Vladimir Osin,
    Satellite Imagery Feature Detection using Deep Convolutional Neural Network: A Kaggle Competition
    arXiv:1706.06169
    """

    def __init__(self, jaccard_weight=0):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            loss += self.jaccard_weight * (1 - soft_jaccard(outputs, targets))
        return loss

class MultiLossBinary:
    """
    Loss defined as BCE - log(soft_jaccard)

    Vladimir Iglovikov, Sergey Mushinskiy, Vladimir Osin,
    Satellite Imagery Feature Detection using Deep Convolutional Neural Network: A Kaggle Competition
    arXiv:1706.06169
    """

    def __init__(self, jaccard_weight=0, root=None):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight
        pos_weight = torch.Tensor([0.5]).to('cuda')
        self.weighted_nll_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def __call__(self, outputs, targets, coutputs, ctargets):
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)
        closs = self.nll_loss(coutputs, ctargets)

        if self.jaccard_weight:
            loss += self.jaccard_weight * (1 - soft_jaccard(outputs, targets))
        return (loss, closs)
