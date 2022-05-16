import torch.nn.functional as F


def nll_loss(output, target):
    # 负对数似然损失
    return F.nll_loss(output, target)

def cross_entropyLoss(output, target):
    return F.cross_entropy(output, target)