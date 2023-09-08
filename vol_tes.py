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
def main (datapath,savepath):

    image_list=sorted(glob.glob(os.path.join(datapath, '*.nii.gz')))

    Numbers = len(image_list)

    # TODO 获取图像坐标SPCING信息
    # dataimage=r'D:\Work\Datasets\GoldNormaldatas20\label\Normal074-MRA.nii.gz'
    # data = sitk.ReadImage(dataimage)
    for i in range(Numbers):
    ##############数据读取######

        # print(image, " | {}/{}".format(i + 1, Numbers))
        #
        # image1 = sitk.ReadImage(image_list[i])
        # arr=sitk.GetArrayFromImage(image1)
        name=image_list[i].split('data\\')[-1]
        img=nib.load(image_list[i])
        arr=img.get_fdata()
        Contrast=AdjustContrast(5)
        arr_1=Contrast(arr)
        arr_1=np.array(arr_1,dtype='float32')
        new_image = nib.Nifti1Image(arr_1, img.affine)
        nib.save(new_image, savepath+'ac_'+name)

        # print(label_arr.shape, vol_sum)

if __name__ == '__main__':

    datapath=r'H:\Datasets\ixi45\data\\'
    savepath=r'H:\Datasets\ixi45\adjust_contrast\\'
    main(datapath,savepath)
    print("test is over!!!")