
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class LossAverage(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = round(self.sum / self.count, 4)
        # print(self.val)

class DiceAverage(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self,class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.val = np.asarray([0]*self.class_num, dtype='float64')
        self.avg = np.asarray([0]*self.class_num, dtype='float64')
        self.sum = np.asarray([0]*self.class_num, dtype='float64')
        self.count = 0

    def update(self, logits, targets):
        self.val = DiceAverage.get_dices(logits, targets)
        self.sum += self.val
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)##########np.around()输入矩阵self.sum/self.count,保留小数点右边4位小数
        # print(self.avg)

    @staticmethod
    def get_dices(logits, targets):
        dices = []
        logits=torch.tensor(logits)
        targets=torch.tensor(targets)
        inter = torch.sum(logits[ :, :, :] * targets[ :, :, :])
        union = torch.sum(logits[ :, :, :]) + torch.sum(targets[ :, :, :])#tensor(142804., dtype=torch.float64)
        dice = (2. * inter + 1) / (union + 1)
        dices.append(dice.item())
        print(np.asarray(dices))
        return np.asarray(dices)
