import numpy as np
import torch
import random
from torch import sigmoid
from lib.losses3D.basic import expand_as_one_hot
from lib.losses3D.basic import flatten
def data_transform(output_list, target_list):#（8，2，16，16，16）
    # input=input.sigmoid()#归一化
    # score_arr = np.array(input)
    # target_arr = np.array(target_expanded)
    #将append的list增加维度转化为新的tensor，
    # output_list=output_list.sigmoid()

    #list转为tensor
    final_output = torch.stack(output_list)
    final_target = torch.stack(target_list)
    ###将上述转化tensor值变成一维，然后再转化为数组格式用于计算roc
    input=flatten(final_output)
    target_expanded=flatten(final_target)


    TP = (final_output * final_target).sum()
    FP = final_output.sum() - TP
    FN = final_target.sum() - TP
    TN=(final_output.size(0)*final_output.size(1)*final_output.size(2)*4096)-TP-FP-FN

    Precision=TP/(TP+FP)
    Accuracy=(TP+TN)/(TP+TN+FP+FN)
    Recall =TP/(TP+FN)
    F1=(2*Precision*Recall)/(Precision+Recall)
    print("Precision =",str(Precision),"\n Accuracy =",str(Accuracy),"\n Recall =",str(Recall),"\n F1 =",str(F1))
    #转为数组形式
    score_arr = input.cpu().detach().numpy()
    target_arr = target_expanded.cpu().detach().numpy()


    # print("score_array:", score_arr.shape)  # (batchsize, classnum)
    print("target_array:", target_arr.shape)  # torch.Size([batchsize, classnum])
    # target_arr=target_expanded.cpu().detach().numpy()
    # assert input_arr.shape==target_expanded.shape
    #
    # score_list.extend(score_tmp.detach().cpu().numpy())
    # label_list.extend(labels.cpu().numpy())

    # from sklearn.utils.multiclass import type_of_target
    # print(type_of_target(score_arr),type_of_target(target_arr[1]))#判断类型是否是binary
    return score_arr,target_arr