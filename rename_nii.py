''''
time: 2022.8.25
author： zengxq
function1：rename
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
import cv2


if __name__ == '__main__':
    print('start rename !!!')
    ori_path=r'D:\SEED\test\resize_1024\\'
    savepath=r'D:\SEED\test\resize_1024\\'

    #TODO 重命名
    image_list = sorted(glob.glob(os.path.join(ori_path,  '*.png')))
    for i in range(0,len(image_list),1):
        datapath=image_list[i]
        name=datapath.split('resize_1024\\')[-1].split('_')[-1].replace('.png','')
        num=datapath.split('resize_1024\\')[-1].split('_')[0]
        new_name=name+'_Annotation'+num+'.png'
        image = cv2.imread(datapath)
        cv2.imwrite(savepath + new_name, image)
        print('\r[ %d / %d]' % (i, len(image_list)), end='')


