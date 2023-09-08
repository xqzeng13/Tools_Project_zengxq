# function 1: transfrom data size by ndimage.zoom()
# function 2: data load to test
# import cv2
import glob
import os
# import torchio
import numpy as np
# import torch
import SimpleITK as sitk
from matplotlib import pyplot as plt
from scipy import ndimage

# from torchio.transforms import (
#     RandomFlip,
#     RandomAffine,
#     RandomElasticDeformation,
#     RandomNoise,
#     RandomMotion,
#     RandomBiasField,
#     RescaleIntensity,
#     Resample,
#     ToCanonical,
#     ZNormalization,
#     CropOrPad,
#     HistogramStandardization,
#     OneOf,
#     Compose,
# )



def main ():
    ##############路径设置#########
    datapath=r"/data4/zengxq/ATM22/test/"
    savepath=r'/data4/zengxq/ATM22/test/data_resample/'

    # if not os.path.exists(savepath):
    # if  not os.path.exists(savepath):  # 创建保存目录
    #     os.makedirs(savepath + 'data')
    #     os.makedirs(savepath + 'label')
    # # cut_param=[64,64,64]
    ############resize的大小#######
    s = 128
    h = 448
    w = 448
    # cut_param = {'patch_s': 32,
    #              'patch_h': 32,
    #              'patch_w': 32,
    #              'stride_s': 64,
    #              'stride_h': 64,
    #              'stride_w': 64}
    ######数据加载#############
    data_list= sorted(glob.glob(os.path.join(datapath,'data/'+ '*.nii.gz')))
    label_list=sorted(glob.glob(os.path.join(datapath, 'label/'+'*.nii.gz')))

    Numbers = len(data_list)
    print('Total numbers of samples is :', Numbers)


    for dataimage ,labelimage,i in zip (data_list,label_list,range(Numbers)):
    ##############数据读取######
        print(dataimage, " | {}/{}".format(i + 1, Numbers))
        print(labelimage, " | {}/{}".format(i + 1, Numbers))
    #######读取data######
        # s, h, w = dataimage.shape
        # dataimage=r'D:\Work\Datasets\GoldNormaldatas20\patch_label_result-411.nii.gz'
        # data = sitk.ReadImage(dataimage)
        # data_np = sitk.GetArrayFromImage(data)######3
    # ##转化为数组##########
        s1,h1,w1= (1,1,1)
        # print("s,h,w", s1, h1, w1)
        data = sitk.ReadImage(dataimage)
        Spacing = data.GetSpacing()
        s,h,w=Spacing
        data_np = sitk.GetArrayFromImage(data)######3

        # data_array=ndimage.zoom(data_np,(s/s1,h/h1,w/w1),order=1)###########插值法下采样

        #######读取label################
        label = sitk.ReadImage(labelimage)####data:order=3////label:order=0#####
        label_np = sitk.GetArrayFromImage(label)  ########转化为数组##########
        # label_array = ndimage.zoom(label_np, (s/s1,h/h1,w/w1), order=0)  ###########插值法下采样
    ###################################根据所需尺寸更改，但得注意data/label需要同时同尺度变化##############
         ######## Bilinear interpolation （双线性插值） order = 1,    Nearest （最邻近插值） order = 0,

    ###############文件保存#########################
            ####data#####
        new_data = sitk.GetImageFromArray(data_np)
        new_data.SetDirection(data.GetDirection())  ###########图像方向不变
        new_data.SetOrigin(data.GetOrigin())  ###########图像原点不变
        # new_data.SetSpacing((data.GetSpacing()[0] ,data.GetSpacing()[1] ,data.GetSpacing()[2]))#####层间距变为()
        new_data.SetSpacing((1, 1, 1))  #####层间距变为()

    # print(data.GetSpacing()[0],data.GetSpacing()[1],data.GetSpacing()[2])
            ########label#####
        new_label = sitk.GetImageFromArray(label_np)
        new_label.SetDirection(label.GetDirection())  ###########图像方向不变
        new_label.SetOrigin(label.GetOrigin())  ###########图像原点不变
        # new_label.SetSpacing((label.GetSpacing()[0], label.GetSpacing()[1], s/s1))  #####层间距变为()
        # new_label.SetSpacing((label.GetSpacing()[0], label.GetSpacing()[1], label.GetSpacing()[2]))
        new_label.SetSpacing((1, 1, 1))

    # sitk.WriteImage(new_data, os.path.join(savepath ,'data\\', data))
            #######保存######################
        sitk.WriteImage(new_data,savepath+'data/'+dataimage.split('data/')[-1])#保存格式（保存文件，路径+文件名+文件格式）
        sitk.WriteImage(new_label,savepath + 'label/' +dataimage.split('data/')[-1] )  # 保存格式（保存文件，路径+文件名+文件格式）

    # ** ERROR(nifti_image_write_hdr_img2): cannot    open    output    file    '.\output\newdatas\\data\ormal097-MRA_brain.nii.gz'
    # ** ERROR(nifti_image_write_hdr_img2): cannot    open    output    file    '.\output\newdatas\\label\ormal097-MRA.nii.gz'

def main_sub():

        ##############路径设置#########
        datapath = r"E:\Improtant file\patient\mask\\"
        savepath = r'E:\Improtant file\patient\mask_0.3516\\'

        # if not os.path.exists(savepath):
        # if  not os.path.exists(savepath):  # 创建保存目录
        #     os.makedirs(savepath + 'data')
        #     os.makedirs(savepath + 'label')
        # # cut_param=[64,64,64]
        ############resize的大小#######
        # s = 128
        # h = 448
        # w = 448
        # cut_param = {'patch_s': 32,
        #              'patch_h': 32,
        #              'patch_w': 32,
        #              'stride_s': 64,
        #              'stride_h': 64,
        #              'stride_w': 64}
        ######数据加载#############
        data_list = sorted(glob.glob(os.path.join(datapath,  '*.nii.gz')))
        # label_list = sorted(glob.glob(os.path.join(datapath, 'target\\' + '*.nii.gz')))

        Numbers = len(data_list)
        print('Total numbers of samples is :', Numbers)

        for dataimage, i in zip(data_list, range(Numbers)):
            ##############数据读取######
            print(dataimage, " | {}/{}".format(i + 1, Numbers))

            # s1, h1, w1 = (128, 448, 448)
            # print("s,h,w", s1, h1, w1)
            data = sitk.ReadImage(dataimage)
            data_np = sitk.GetArrayFromImage(data)  ######3

            data_array = ndimage.zoom(data_np, (1, 1,1), order=1)  ###########插值法下采样

            # #######读取label################
            # label = sitk.ReadImage(labelimage)  ####data:order=3////label:order=0#####
            # label_np = sitk.GetArrayFromImage(label)  ########转化为数组##########
            # label_array = ndimage.zoom(label_np, (s / s1, h / h1, w / w1), order=0)  ###########插值法下采样
            ###################################根据所需尺寸更改，但得注意data/label需要同时同尺度变化##############
            ######## Bilinear interpolation （双线性插值） order = 1,    Nearest （最邻近插值） order = 0,

            ###############文件保存#########################
            ####data#####
            new_data = sitk.GetImageFromArray(data_array)
            new_data.SetDirection(data.GetDirection())  ###########图像方向不变
            new_data.SetOrigin(data.GetOrigin())  ###########图像原点不变
            # new_data.SetSpacing((data.GetSpacing()[0] ,data.GetSpacing()[1] ,data.GetSpacing()[2]))#####层间距变为()
            new_data.SetSpacing((0.3516, 0.3516, 0.7))  #####层间距变为()

            # print(data.GetSpacing()[0],data.GetSpacing()[1],data.GetSpacing()[2])
            ########label#####
            # new_label = sitk.GetImageFromArray(label_array)
            # new_label.SetDirection(label.GetDirection())  ###########图像方向不变
            # new_label.SetOrigin(label.GetOrigin())  ###########图像原点不变
            # new_label.SetSpacing((label.GetSpacing()[0], label.GetSpacing()[1], s/s1))  #####层间距变为()
            # new_label.SetSpacing((label.GetSpacing()[0], label.GetSpacing()[1], label.GetSpacing()[2]))
            # new_label.SetSpacing((1, 1, 1))

            # sitk.WriteImage(new_data, os.path.join(savepath ,'data\\', data))
            #######保存######################
            sitk.WriteImage(new_data,
                            savepath  + dataimage.split('mask\\')[-1])  # 保存格式（保存文件，路径+文件名+文件格式）
            # sitk.WriteImage(new_label,
            #                 savepath + 'target\\' + dataimage.split('dataimage\\')[-1])  # 保存格式（保存文件，路径+文件名+文件格式）

        # ** ERROR(nifti_image_write_hdr_img2): cannot    open    output    file    '.\output\newdatas\\data\ormal097-MRA_brain.nii.gz'
        # ** ERROR(nifti_image_write_hdr_img2): cannot    open    output    file    '.\output\newdatas\\label\ormal097-MRA.nii.gz'
def padding_img( img, C):
    assert (len(img.shape) == 3)  # 3D array
    img_s, img_h, img_w = img.shape
    leftover_s = (img_s - C['patch_s']) % C['stride_s']  ##############   %求余
    leftover_h = (img_h - C['patch_h']) % C['stride_h']
    leftover_w = (img_w - C['patch_w']) % C['stride_w']
    if (leftover_s != 0):
        s = img_s + (C['stride_s'] - leftover_s)
    else:
        s = img_s

    if (leftover_h != 0):
        h = img_h + (C['stride_h'] - leftover_h)
    else:
        h = img_h

    if (leftover_w != 0):
        w = img_w + (C['stride_w'] - leftover_w)
    else:
        w = img_w

    tmp_full_imgs = np.zeros((s, h, w))############给定这个形状，数据填充0
    tmp_full_imgs[:img_s, :img_h, 0:img_w] = img
    print("Padded images shape: " + str(img.shape))
    return tmp_full_imgs







        # print(data_np)
    # print(data_list)


    # img = cv2.imread("cube.png", -1);
    # h, w, c = img.shape;
    #
    # img_n = cv2.resize(img, (w//2, h//2), interpolation = cv2.INTER_NEAREST);
    # img_n = cv2.resize(img_n, (w, h), interpolation = cv2.INTER_NEAREST);

# def transform():
#
#     if  mode == '3d':
#         training_transform = Compose([
#         # ToCanonical(),
#         CropOrPad((crop_or_pad_size, crop_or_pad_size, crop_or_pad_size), padding_mode='reflect'),
#         RandomMotion(),
#         RandomBiasField(),
#         ZNormalization(),
#         RandomNoise(),
#         RandomFlip(axes=(0,)),
#         OneOf({
#             RandomAffine(): 0.8,
#             RandomElasticDeformation(): 0.2,
#         }),])
#     # elif hp.mode == '2d':
#     #     training_transform = Compose([
#     #     CropOrPad((hp.crop_or_pad_size, hp.crop_or_pad_size,1), padding_mode='reflect'),
#     #     RandomMotion(),
#     #     RandomBiasField(),
#     #     ZNormalization(),
#     #     RandomNoise(),
#     #     RandomFlip(axes=(0,)),
#     #     OneOf({
#     #         RandomAffine(): 0.8,
#     #         RandomElasticDeformation(): 0.2,
#     #     }),])
#     else:
#         raise Exception('no such kind of mode!')
#
#
#     return training_transform





if __name__ == '__main__':
    # mode='3d'
    # crop_or_pad_size=64
    # main_sub()
    main()
    print("test is over!!!")