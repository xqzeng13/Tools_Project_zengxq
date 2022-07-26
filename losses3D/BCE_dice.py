import torch.nn as nn
import torch
from lib.losses3D.dice import DiceLoss
from lib.losses3D.basic import expand_as_one_hot
from sklearn.metrics import accuracy_score,auc,roc_curve
# Code was adapted and mofified from https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/losses.py


class BCEDiceLoss(nn.Module):
    """Linear combination of BCE and Dice losses3D"""

    def __init__(self, alpha=1, beta=1, classes=4):
        super(BCEDiceLoss, self).__init__()
        # self.alpha = alpha#####平衡因子，用于平衡正负比例
        # self.bce = nn.BCEWithLogitsLoss()
        # self.beta = beta
        # self.dice = DiceLoss(classes=classes)
        # self.classes=classes
        self.alpha = 1  #####平衡因子，用于平衡正负比例

        self.bce = nn.BCEWithLogitsLoss()
        self.beta = 1
        self.dice = DiceLoss(classes=2)
        self.classes = 2


    def forward(self, input, target):
        # target_expanded = expand_as_one_hot(target.long(), self.classes)
        target_expanded_BCE = target.unsqueeze(1)
        assert input.size() == target_expanded_BCE.size(), "'input' and 'target' must have the same shape"
        device = torch.device('cuda:0')

        input.to(device)
        target_expanded_BCE.to(device)
        eposion = 1e-10
        # sigmoid_pred = torch.sigmoid(input)
        count_pos = torch.sum(target_expanded_BCE) * 1.0 + eposion#计算一个batch有多少正样本
        count_neg = torch.sum(1. - target_expanded_BCE) * 1.0#计算一个batch有多少负样本
        beta = count_neg / count_pos###计算负样本是正样本的倍数,tensor类型
        beta_back = count_pos / (count_pos + count_neg)

        # bce1 = nn.BCEWithLogitsLoss(pos_weight=beta)####随每次的batch变化而变化
        bce1 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([beta]).to(device))####随每次的batch变化而变化
        # bce1 = nn.BCEWithLogitsLoss()####随每次的batch变化而变化

        # print(input.is_cuda, target_expanded.is_cuda)
        loss_1 =  bce1(input, target_expanded_BCE)

        loss_2, channel_score = self.beta * self.dice(input, target)

        return  (loss_1) , channel_score
        #
        # return (loss_1 )


