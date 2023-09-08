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



if __name__ == '__main__':
    print('start rename !!!')
    ori_path=r'E:\MRA_public\final_datasets\ori_data\semi_train\\'
    # txt_savepath=r'E:\MRA_public\final_datasets\ori_data\semi_train\\'

    # #TODO 重命名
    # image_list = sorted(glob.glob(os.path.join(ori_path,  '*.nii.gz')))
    # for i in range(len(image_list)):
    #     dataname=image_list[i]
    #     new_name=savepath+dataname.split('nn_pred_mask\\')[-1].split('.nii.gz')[0]+'_0000'+'.nii.gz'
    #     img_nii = nib.load(dataname)
    #     nib.save(img_nii, new_name)
    #     print('\r[ %d / %d]' % (i, len(image_list)), end='')


#read file number
    namelist=[]
    txt_name=r'E:\MRA_public\final_datasets\ori_data\semi_train\\'
    image_list = sorted(glob.glob(os.path.join(ori_path,  'data\\'+'*.nii.gz')))
    fp = open(txt_name+'train.txt', "a+")  # a+ 如果文件不存在就创建。存在就在文件内容的后面继续追加
    for i in range(len(image_list)):
        dataname=image_list[i]
        new_name=dataname.split('data\\')[-1].split('.nii.gz')[0]
        print(new_name,file=fp)
        print('\r[ %d / %d]' % (i, len(image_list)), end='')
