import glob
import os
# import torchio
import numpy as np
import torch
import SimpleITK as sitk
import nibabel as nib

from matplotlib import pyplot as plt
from scipy import ndimage




# if __name__ == '__main__':
#     ##############路径设置#########
#     datapath=r"D:\result_fusion\\"
#     savepath=r'D:\result_fusion\Result_fusion\\'
#
#
#     unet_data_list = sorted(glob.glob(os.path.join(datapath,'UNET\\'+ '*.nii.gz')))
#     unet_attention_data_list = sorted(glob.glob(os.path.join(datapath,'UNET_A\\'+ '*.nii.gz')))
#     unet_Rattention_data_list = sorted(glob.glob(os.path.join(datapath,'UNET_RA\\'+ '*.nii.gz')))
#
#     # if not os.path.exists(savepath):
#     # if  not os.path.exists(savepath):  # 创建保存目录
#     #     os.makedirs(savepath + 'data')
#     #     os.makedirs(savepath + 'label')
#     N=len(unet_data_list)
#     for i in range(N):
#         name='Fusion'+'Nor'+unet_data_list[i].split('Nor')[-1]
#         unet_data = nib.load(unet_data_list[i]).get_fdata()###(dtype=np.float32)?????????????
#         unet_attention_data = nib.load(unet_attention_data_list[i]).get_fdata()###(dtype=np.float32)?????????????
#         unet_Rattention_data = nib.load(unet_Rattention_data_list[i]).get_fdata()###(dtype=np.float32)?????????????
#         fusion_data=unet_data+unet_attention_data+unet_Rattention_data
#         fusion_data[fusion_data<2]=0
#         fusion_data[fusion_data>=2]=1
#
#         print('fusion_data shape is :',fusion_data.shape)
#         fusion_image=nib.Nifti1Image(fusion_data,np.eye(4))
#         nib.save(fusion_image,savepath+name)
#         print('\r[ %d / %d]' % (i, N), end='')

if __name__ == '__main__':
    ##############路径设置#########
    # datapath=r"D:\result_fusion\\"
    # savepath=r'D:\result_fusion\Result_fusion\\'

    datafilePath = r'D:\Work\Datasets\Challenge\test\pred_result\result\\'
    dataoutput = r'D:\Work\Datasets\Challenge\test\pred_result\\'

    unet_dataList = sorted(glob.glob(os.path.join(datafilePath, 'ranojuchi/*.nii.gz')))

    unet_raList = sorted(glob.glob(os.path.join(datafilePath, 'fusion/*.nii.gz')))
    unet_noresList = sorted(glob.glob(os.path.join(datafilePath, 'nores/*.nii.gz')))
    dataimage = r'D:\Work\Datasets\GoldNormaldatas20\label\Normal074-MRA.nii.gz'
    # TODO 获取图像坐标SPCING信息
    data = sitk.ReadImage(dataimage)
    # unet_data = nib.load(r'D:\result_fusion\0_0.nii.gz').get_fdata()  ###(dtype=np.float32)?????????????
    # unet_attention_data = nib.load(r'D:\result_fusion\A_0.nii.gz').get_fdata()  ###(dtype=np.float32)?????????????
    # unet_Rattention_data = nib.load(r'D:\result_fusion\RA_0.nii.gz').get_fdata()  ###(dtype=np.float32)?????????????
    for N in range (len(unet_dataList)):
        name="predt"+str(unet_dataList[N]).split('predt')[1]
        unet_data = nib.load(unet_dataList[N]).get_fdata()
        unet_ra = nib.load(unet_raList[N]).get_fdata()
        unet_nores = nib.load(unet_noresList[N]).get_fdata()

        fusion_data = unet_data + unet_ra + unet_nores
        fusion_data[fusion_data < 2] = 0
        fusion_data[fusion_data >= 2] = 1
        # unet_data1 = nib.load(r'D:\result_fusion\0_1.nii.gz').get_fdata()  ###(dtype=np.float32)?????????????
        # unet_attention_data1 = nib.load(r'D:\result_fusion\A_1.nii.gz').get_fdata()  ###(dtype=np.float32)?????????????
        # unet_Rattention_data1 = nib.load(r'D:\result_fusion\RA_1.nii.gz').get_fdata()  ###(dtype=np.float32)?????????????
        # fusion_data1 = unet_data1 + unet_attention_data1 + unet_Rattention_data1
        # fusion_data1[fusion_data1 < 1.5] = 0.2
        # fusion_data1[fusion_data1 >= 1.5] = 0.8
        # import torch
        ##TODO 针对通道为2的概率图谱
        # fusion_data=torch.from_numpy(fusion_data)
        # fusion_data=fusion_data.unsqueeze(0)
        # fusion_data1=torch.from_numpy(fusion_data1)
        # fusion_data1=fusion_data1.unsqueeze(0)
        # fusion_result=torch.cat([fusion_data,fusion_data1],dim=0)
        # pred_img = torch.argmax(fusion_result, dim=0)
        # pred_img=pred_img.numpy()
        # print(pred_img.shape)
        # savepath=r'D:\result_fusion\\'
        pred_img = fusion_data.swapaxes(0, 2)  ###交换x,z  调整方向
        pred_img = sitk.GetImageFromArray(np.squeeze(np.array(pred_img, dtype='uint8')))
        ######################################################先转为numpy(),再转为数组array，再减少维数送入sitk中处理
        ##TODO 赋予原图的图像信息：方向，原坐标，层间距
        pred_img.SetDirection(data.GetDirection())  ###########图像方向不变
        pred_img.SetOrigin(data.GetOrigin())  ###########图像原点不变
        pred_img.SetSpacing((data.GetSpacing()[0], data.GetSpacing()[1], data.GetSpacing()[2]))
        sitk.WriteImage(pred_img, os.path.join(dataoutput,name))