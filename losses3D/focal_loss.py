# from keras import backend as K
# import tensorflow as tf
#
# # Compatible with tensorflow backend
#
# def focal_loss(gamma=2., alpha=.25):
# 	def focal_loss_fixed(y_true, y_pred):
# 		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
# 		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
# 		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1+K.epsilon())) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
# 	return focal_loss_fixed
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# class FocalLoss(nn.Module):
#     def __init__(self, gamma=0, alpha=None, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
#         if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
#         self.size_average = size_average
#
#     def forward(self, input, target):
#         if input.dim()==4:
#             input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
#             input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
#         elif input.dim()==5:
#             input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
#             input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
#         target = target.view(-1,1)
#
#         logpt = F.log_softmax(input)
#         logpt = logpt.gather(1,target)
#         logpt = logpt.view(-1)
#         pt = Variable(logpt.data.exp())
#
#         if self.alpha is not None:
#             if self.alpha.type()!=input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
#             at = self.alpha.gather(0,target.data.view(-1))
#             logpt = logpt * Variable(at)
#
#         loss = -1 * (1-pt)**self.gamma * logpt
#         if self.size_average: return loss.mean()
#         else: return loss.sum()
# 针对多分类任务的 CELoss 和　Focal Loss
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


class CELoss(nn.Module):
    def __init__(self, class_num, alpha=None, use_alpha=False, size_average=True):
        super(CELoss, self).__init__()
        self.class_num = class_num
        self.alpha = alpha
        if use_alpha:
            self.alpha = torch.tensor(alpha).cuda()

        self.softmax = nn.Softmax(dim=1)
        self.use_alpha = use_alpha
        self.size_average = size_average

    def forward(self, pred, target):
        prob = self.softmax(pred.view(-1, self.class_num))
        prob = prob.clamp(min=0.0001, max=1.0)

        target_ = torch.zeros(target.size(0), self.class_num).cuda()
        target_.scatter_(1, target.view(-1, 1).long(), 1.)

        if self.use_alpha:
            batch_loss = - self.alpha.double() * prob.log().double() * target_.double()
        else:
            batch_loss = - prob.log().double() * target_.double()

        batch_loss = batch_loss.sum(dim=1)

        # print(prob[0],target[0],target_[0],batch_loss[0])
        # print('--')

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss


class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, use_alpha=False, size_average=True):
        super(FocalLoss, self).__init__()
        self.class_num = class_num
        self.alpha = alpha
        self.gamma = gamma
        if use_alpha:
            self.alpha = torch.tensor(alpha).cuda()

        self.softmax = nn.Softmax(dim=1)
        self.use_alpha = use_alpha
        self.size_average = size_average

    def forward(self, pred, target):
        # device = torch.device('cuda:5')
        # device = torch.device('cuda:5')
		#
        # pred.to(device)
        # target.to(device)
        prob = self.softmax(pred.view(-1, self.class_num))
        prob = prob.clamp(min=0.0001, max=1.0)

        target_ = torch.zeros(target.size(0), self.class_num).cuda()
        target_.scatter_(1, target.view(-1, 1).long(), 1.)

        if self.use_alpha:
            batch_loss = - self.alpha.double() * torch.pow(1 - prob,self.gamma).double() * prob.log().double() * target_.double()
        else:
            batch_loss = - torch.pow(1 - prob, self.gamma).double() * prob.log().double() * target_.double()

        batch_loss = batch_loss.sum(dim=1)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss


if __name__ == '__main__':
    device = torch.device('cuda:0')


    pre=torch.randn(16,1,64,64,32)
    size=(16,64,64,32)
    target=torch.ones(size=size,dtype=torch.int64)
    pre.to(device)
    target.to(device)
    # target=torch.one()
    # target=target.int64()
    loss=FocalLoss(2, alpha=0.25, gamma=2, use_alpha=True, size_average=True)
    loss_score=loss(pre,target)