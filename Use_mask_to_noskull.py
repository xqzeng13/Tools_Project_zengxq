''''
function：利用s

'''

import glob
import os
# import torchio
import numpy as np
import torch
import torchvision
import SimpleITK as sitk
from monai.transforms import AdjustContrast
from matplotlib import pyplot as plt
from scipy import ndimage
import nibabel as nib

from sklearn.metrics import precision_score, recall_score, f1_score
def main (datapath,maskpath,savepath):

    image_list=sorted(glob.glob(os.path.join(datapath, '*.nii')))
    mask_list=sorted(glob.glob(os.path.join(maskpath, '*.nii.gz')))

    Numbers = len(mask_list)

    # TODO 获取图像坐标SPCING信息
    # dataimage=r'D:\Work\Datasets\GoldNormaldatas20\label\Normal074-MRA.nii.gz'
    # data = sitk.ReadImage(dataimage)
    for i in range(Numbers):
    ##############数据读取######

        # print(image, " | {}/{}".format(i + 1, Numbers))
        #
        # image1 = sitk.ReadImage(image_list[i])
        # arr=sitk.GetArrayFromImage(image1)
        print('\n img is :',image_list[i])
        print('\n mask  is :', mask_list[i])

        img=nib.load(image_list[i])
        img_arr=img.get_fdata()

        mask = nib.load(mask_list[i])
        msk_arr = mask.get_fdata()

        new_image_arr=img_arr*msk_arr


        new_image = nib.Nifti1Image(new_image_arr, img.affine)
        nib.save(new_image, savepath+mask_list[i].split('mask_0.3516\\')[-1])

        # print(label_arr.shape, vol_sum)

if __name__ == '__main__':

    datapath=r'E:\Improtant file\patient\mra_test\\'
    maskpath=r'E:\Improtant file\patient\mask_0.3516\\'

    savepath=r'E:\Improtant file\patient\noskull\\'
    main(datapath,maskpath,savepath)
    print("test is over!!!")