import glob
import os
# import torchio
import numpy as np
import torch
import SimpleITK as sitk
from matplotlib import pyplot as plt
from scipy import ndimage
def main (datapath):

    predname='Unet_ra_new'
    pre_list= sorted(glob.glob(os.path.join(datapath,predname+'\\'+ '*.nii.gz')))
    label_list=sorted(glob.glob(os.path.join(datapath, 'target\\'+'*.nii.gz')))

    Numbers = len(pre_list)
    print('Total numbers of samples is :', Numbers)
    #TODO 获取图像坐标SPCING信息
    dataimage=r'D:\Work\Datasets\GoldNormaldatas20\label\Normal074-MRA.nii.gz'
    data = sitk.ReadImage(dataimage)

    dicescore_list=[]
    iouscore_list=[]
    prescore_list=[]
    recallscore_list=[]
    f1score_list=[]
    for preimage ,labelimage,i in zip (pre_list,label_list,range(Numbers)):
    ##############数据读取######
        print(preimage, " | {}/{}".format(i + 1, Numbers))
        print(labelimage, " | {}/{}".format(i + 1, Numbers))

        pre = sitk.ReadImage(preimage)
        pre_arr = sitk.GetArrayFromImage(pre)
        label = sitk.ReadImage(labelimage)
        label_arr = sitk.GetArrayFromImage(label)

        intersection_00=(1-pre_arr)*(1-label_arr)#####预测，标签都是0时TN
        intersection=pre_arr*label_arr#对应位置相乘：都是1时全为1；其他情况都是0TP
        intersection_01=(1-pre_arr)*label_arr####预测为0标签为1的占比---漏判FN
        intersection_10=pre_arr*(1-label_arr)####预测为1，标签为0的占比---误判FP

        result00 = intersection_00.sum() /( 1-label_arr).sum()
        result11 = intersection.sum() / label_arr.sum()
        result01= intersection_01.sum() / label_arr.sum()
        result10 = intersection_10.sum() / label_arr.sum()


        IOU_Score=(pre_arr*label_arr).sum()/(pre_arr.sum()+label_arr.sum())
        DICE_Score=(2*intersection.sum())/(intersection_10.sum()+intersection_01.sum()+2*intersection.sum())
        Pre_Score=intersection.sum()/(intersection.sum()+intersection_10.sum())
        Recall_Score=intersection.sum()/(intersection.sum()+intersection_01.sum())
        F1_Score=2*Pre_Score*Recall_Score/(Pre_Score+Recall_Score)

        dicescore_list.append(DICE_Score)
        iouscore_list.append(IOU_Score)
        prescore_list.append(Pre_Score)
        recallscore_list.append(Recall_Score)
        f1score_list.append(F1_Score)
        print('\n 预测背景正确——result00 ,', result00)
        print('\n 预测血管正确——result11 ,', result11)
        print('\n 漏判——result01 ,', result01)
        print('\n 误判——result10 ,', result10)
        print('\n IOU_Score ,', IOU_Score)
        print('\n DICE_Score ,', DICE_Score)
        print('\n Pre_Score ,', Pre_Score)
        print('\n Recall_Score ,', Recall_Score)
        print('\n F1_Score ,', F1_Score)

    print('Finish!!!')
    dice_ave=np.asarray(dicescore_list)
    iou_ave=np.asarray(iouscore_list)
    pre_ave=np.asarray(prescore_list)
    recall_ave=np.asarray(recallscore_list)
    f1_ave=np.asarray(f1score_list)
    fp = open(r'D:\Work\paper_result\result_sum.txt.txt', "a+")
    print('\n Net name is ,',predname,file=fp)
    print('\n Pre_Score ,', pre_ave.mean(),pre_ave.std(),file=fp)
    print('\n Recall_Score ,', recall_ave.mean(),recall_ave.std(),file=fp)
    print('\n F1_Score ,', f1_ave.mean(),f1_ave.std(),file=fp)
    print('\n DICE_Score ,', dice_ave.mean(),dice_ave.std(),file=fp)
    print('\n IOU_Score ,', iou_ave.mean(),iou_ave.std(),file=fp)


    # # false_positive=pre_arr-intersection
    #     # false_negtive=label_arr-intersection
    #     # XOR=np.bitwise_xor(pre_arr,label_arr)#异或
    #     # Andr=np.bitwise_and(pre_arr,label_arr)
    #     # xor_rate=XOR.sum()/label_arr.sum()
    #     # Andr_rate = Andr.sum() / label_arr.sum()
    #     #
    #     # true_positiverate = intersection.sum() / label_arr.sum()
    #     # false_positiverate = false_positive.sum() / label_arr.sum()
    #     # false_negtiverate = false_negtive.sum() / label_arr.sum()
    #     #
    #     # print('\n true_positiverate ,', true_positiverate)
    #     # print('\n false_positiverate ,', false_positiverate)
    #     # print('\n false_negtiverate ,', false_negtiverate)
    #     # print('\n xor_rate ,', xor_rate)
    #     # print('\n Andr_rate ,', Andr_rate)
    #
    #
    # ##TODO 赋予原图的图像信息：方向，原坐标，层间距
    #     intersection = sitk.GetImageFromArray(np.squeeze(np.array(intersection, dtype='uint8')))
    #     intersection.SetDirection(data.GetDirection())  ###########图像方向不变
    #     intersection.SetOrigin(data.GetOrigin())  ###########图像原点不变
    #     intersection.SetSpacing((data.GetSpacing()[0], data.GetSpacing()[1], data.GetSpacing()[2]))
    #     sitk.WriteImage(intersection, os.path.join(datapath,'intersection\\'+ preimage.split('pre\\')[-1] ))
    #
    #
    #     XOR = sitk.GetImageFromArray(np.squeeze(np.array(XOR, dtype='uint8')))
    #     XOR.SetDirection(data.GetDirection())  ###########图像方向不变
    #     XOR.SetOrigin(data.GetOrigin())  ###########图像原点不变
    #     XOR.SetSpacing((data.GetSpacing()[0], data.GetSpacing()[1], data.GetSpacing()[2]))
    #     sitk.WriteImage(XOR, os.path.join(datapath, 'intersection\\' +'XOR'+ preimage.split('pre\\')[-1]))

if __name__ == '__main__':

    datapath=r'D:\Work\paper_result\\'

    main(datapath)
    print("test is over!!!")