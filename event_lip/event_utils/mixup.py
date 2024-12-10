import torch
import numpy as np


# -- mixup data augmentation
# from https://github.com/hongyi-zhang/mixup/blob/master/cifar/utils.py
def mixup_data(x, y, alpha=1.0, use_cuda=False, lam=None):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''

    if not lam:

        if alpha > 0.:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.

    batch_size = y.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    # print(index)
    for key in x:
        x[key] = lam * x[key] + (1 - lam) * x[key][index, :]

    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
