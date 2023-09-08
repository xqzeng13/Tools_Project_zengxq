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

#%%
#
# filepath=r'I:\ATLAS_R2.0\ATLAS_2\Testing\\'
# savepath=r'I:\ATLAS_R2.0_new\test\\'
# savedata=os.path.join(savepath, 'image\\')
# # savelabel=os.path.join(savepath, 'mask\\')
# file_list = sorted(glob.glob(os.path.join(filepath,  '*')))#R001
# num=0
# for i in range (len(file_list)):
#     dir=file_list[i]
#
#     print("dir:",dir)
#     file1_list = sorted(glob.glob(os.path.join(dir,  '*')))#sub-r001s001
#     for j in range (len(file1_list)):
#         filepath=file1_list[j]+'\\ses-1\\anat\\'
#         print('filepath',filepath)
#         # num = dir.split('R')[-1]
#         # mask_list = sorted(glob.glob(os.path.join(filepath,  '*mask.nii.gz')))#
#         image_list=sorted(glob.glob(os.path.join(filepath,  '*T1w.nii.gz')))#
#         for k in range(len(image_list)):
#             image=image_list[k]
#             # mask=mask_list[k]
#             img_nii = nib.load(image)
#             # label_nii = nib.load(mask)
#             file_data=savedata+'image_'+str(num)+'.nii.gz'
#             # file_label=savelabel+'mask_'+str(num)+'.nii.gz'
#
#             nib.save(img_nii,file_data)
#             # nib.save(label_nii,file_label)
#             num=int(num)+1
#

datapath=r'I:\dataset-ISLES22^public^unzipped^version-20220811T105258Z-001\dataset-ISLES22^public^unzipped^version\rawdata\\'
maskpath=r'I:\dataset-ISLES22^public^unzipped^version-20220811T105258Z-001\dataset-ISLES22^public^unzipped^version\derivatives\\'
savepath=r'I:\ISLES22_new\\'
savedata=os.path.join(savepath, 'image\\')
savelabel=os.path.join(savepath, 'mask\\')
data_list = sorted(glob.glob(os.path.join(datapath,  '*')))#R001
mask_list = sorted(glob.glob(os.path.join(maskpath,  '*')))#R001

num=0
for i in range (len(mask_list)):
    data_dir=data_list[i]
    mask_dir=mask_list[i]
    dirname=data_dir.split('rawdata\\')[-1]
    assert data_dir.split('rawdata\\')[-1]==mask_dir.split('derivatives\\')[-1]
    print("dir:",data_dir.split('rawdata\\')[-1])
    data1_list = sorted(glob.glob(os.path.join(data_dir,  '*')))#sub-r001s001
    label1_list=sorted(glob.glob(os.path.join(mask_dir,  '*')))#sub-r001s001
    for j in range (len(data1_list)):
        filepath=data1_list[j]+'\\'
        labelpath=label1_list[j]+'\\'
        print('filepath',filepath)
        # num = dir.split('R')[-1]
        adc_list = sorted(glob.glob(os.path.join(filepath,  '*adc.nii.gz')))#
        dwi_list = sorted(glob.glob(os.path.join(filepath,  '*dwi.nii.gz')))#
        flair_list = sorted(glob.glob(os.path.join(filepath,  '*flair.nii.gz')))#
        mask2_list=sorted(glob.glob(os.path.join(labelpath,  '*msk.nii.gz')))
        # image_list=sorted(glob.glob(os.path.join(filepath,  '*T1w.nii.gz')))#
        for k in range(len(mask2_list)):
            adc=adc_list[k]
            dwi=dwi_list[k]
            flair=flair_list[k]
            mask=mask2_list[k]
            # mask=mask_list[k]
            adc_nii = nib.load(adc)
            dwi_nii = nib.load(dwi)
            flair_nii = nib.load(flair)
            mask_nii=nib.load(mask)

            # label_nii = nib.load(mask)
            savepath=savedata+"sub-strokecase"+str(num)+'\\'
            if not os.path.exists(savepath):
                os.mkdir(savepath)

            file_adc=savepath+'image_'+'_adc.nii.gz'
            file_dwi=savepath+'image_'+'_dwi.nii.gz'
            file_flair=savepath+'image_'+'_flair.nii.gz'
            flie_mask=savepath+'mask_'+'_msk.nii.gz'


            nib.save(adc_nii,file_adc)
            nib.save(dwi_nii,file_dwi)
            nib.save(flair_nii,file_flair)
            nib.save(mask_nii,flie_mask)

            num=int(num)+1