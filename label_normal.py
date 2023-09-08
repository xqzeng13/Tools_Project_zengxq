'''
多标签归一化处理（4分类转2分类）
'''
''''
function1：从文件夹中读取对应模态(标签)进行分割
function2：顺序裁取patches
function3:随机裁取patches

找到对应模态读取对应文件
归一化处理，

'''
import numpy
import csv
import torch
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import pandas as pd
import glob
import gc
import sys
# import psutil
import os


# import pandas as pd
# from lib.medloaders import medical_image_process as img_loader
# from lib.visual3D_temp import *
##########可用
def label_normal(csvname,pathname):

    input_df = pd.read_csv(csvname)
    print("\n sum of train patches is :", input_df.shape[0])
    for i in range(input_df.shape[0]):
        print('\r[ %d / %d]' % (i, input_df.shape[0]), end='')
        dir_name = pathname + input_df.iloc[i].at['BraTS_2020_subject_ID']
        dataname = dir_name + '\\' + input_df.iloc[i].at['BraTS_2020_subject_ID'] + '_t1ce.nii.gz'
        labelname = dir_name + '\\' + input_df.iloc[i].at['BraTS_2020_subject_ID'] + '_seg.nii.gz'
        new_label=dir_name + '\\' + input_df.iloc[i].at['BraTS_2020_subject_ID'] + '_01seg.nii.gz'
        # 讲print的信息保存到txt文件
        # fp = open(r'D:\Work\Datasets\samples\recode.txt', "a+")  # a+ 如果文件不存在就创建。存在就在文件内容的后面继续追加
        # print("N is :",N)
        print("image is :", input_df.iloc[i].at['BraTS_2020_subject_ID'])
        img_nii = nib.load(dataname)
        label_nii = nib.load(labelname)
        ###标签处理，多类合为2类
        label_arr=label_nii.get_fdata()
        affine=img_nii.affine
        label_arr[label_arr>0]=1
        label_nii=nib.Nifti1Image(label_arr,affine)

        nib.save(label_nii,new_label)

if __name__ == '__main__':
    print('start crop patches !!!')
    ###TODO 参数
    csvname = r'E:\MICCAI_BraTS2020_TrainingData~\MICCAI_BraTS2020_TrainingData\name_mapping.csv'  # train_hessian_debug.csv Work\Datasets\new_samples\train_only_vessel_new.csv
    pathname = r'E:\MICCAI_BraTS2020_TrainingData~\MICCAI_BraTS2020_TrainingData\\'
    print("############====label_normal======#############")
    label_normal(csvname,pathname)

