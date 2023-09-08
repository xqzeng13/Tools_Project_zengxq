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
def image_norm(ori_images,new_images,normalization):
    datanameList = sorted(glob.glob(os.path.join(ori_images, '*nii.gz')))
    # labelnameList = sorted(glob.glob(os.path.join(Exsit_training_path, 'label\\'+'*nii.gz')))
    # fp = open(r'E:\MRA_public\final_datasets\data_spilt.txt', "a+")  # a+ 如果文件不存在就创建。存在就在文件内容的后面继续追加
    # print('datanameList is ,',datanameList,file=fp)
    print("\n sum of train patches is :", len(datanameList))
    for i in range(len(datanameList)):
        print('\r[ %d / %d]' % (i, len(datanameList)), end='')
        # dir_name = pathname + input_df.iloc[i].at['BraTS_2020_subject_ID']
        datanamenew = new_images+datanameList[i].split('adjust_contrast\\')[1]
        dataname = datanameList[i]

        # labelname = labelnameList[i]

        # 讲print的信息保存到txt文件
        # fp = open(r'D:\Work\Datasets\samples\recode.txt', "a+")  # a+ 如果文件不存在就创建。存在就在文件内容的后面继续追加
        # print("N is :",N)
        print("image is :", datanameList[i])
        img_nii = nib.load(dataname)
        affine=img_nii.affine
        img_tensor = load_crop_to_mage(img_nii, type="T1", normalization=normalization)
        img_np=np.array(img_tensor)
        # arr1 = np.flip(arr, axis=1)###图像左右调换
        new_image = nib.Nifti1Image(img_np, affine)
        nib.save(new_image, datanamenew)

        # nib.save(img_nii,datanamenew)
        # label_nii = nib.load(labelname)
        ###标签处理，多类合为2类
        # label_arr=label_nii.get_fdata()
        # label_arr[label_arr>0]=1
        # full_vol_dim=label_nii.shape
        # for N in range(samples):
        #     print('\r[ %d / %d]' % (N, samples), end='')
        #
        #     ###随机裁取选择合适占比
        #     while True:
        #         crop = find_random_crop_dim(full_vol_dim, crop_size)
        #         full_segmentation_map = load_crop_to_mage(label_nii, viz3d=True, type='label',crop_size=crop_size,crop=crop)
        #         full_segmentation_map = fix_seg_map(full_segmentation_map, 'Vessel')#标签归一
        #         #
        #         segmentation_map = load_crop_to_mage(label_nii, type='label', crop_size=crop_size,crop=crop)
        #         # segmentation_map = fix_seg_map(segmentation_map, 'Vessel')
        #         img_tensor = load_crop_to_mage(img_nii, type="T1",normalization=normalization,crop_size=crop_size, crop=crop)




def padding_img(img, C,affine):  ##C=64 64 32
    if len(img.shape) == 3:
        assert (len(img.shape) == 3)  # 3D array
        img_h, img_w, img_s = img.shape
        leftover_h = (img_h) % C[0]  ##############%求余
        leftover_w = (img_w) % C[1]
        leftover_s = (img_s) % C[2]

        if (leftover_h != 0):
            h = img_h + (C[0] - leftover_h)
        else:
            h = img_h

        if (leftover_w != 0):
            w = img_w + (C[1] - leftover_w)
        else:
            w = img_w

        if (leftover_s != 0):
            s = img_s + (C[2] - leftover_s)
        else:
            s = img_s

        tmp_full_imgs = np.zeros((h, w, s))
        tmp_full_imgs[:img_h, :img_w, 0:img_s] = img
        # print("Padded images shape: " + str(img.shape))
        # tmp_full_imgs=tmp_full_imgs.swapaxes(0,1)
        new_imag=nib.Nifti1Image(tmp_full_imgs,affine)
        # print("Padded images shape: " + str(img.shape))
        return new_imag
        # return tmp_full_imgs
    if len(img.shape) == 4:
        assert (len(img.shape) == 4)  # 3D array
        bachsize, img_h, img_w, img_s = img.shape
        leftover_h = (img_h) % C[0]  ##############%求余
        leftover_w = (img_w) % C[1]
        leftover_s = (img_s) % C[2]

        if (leftover_h != 0):
            h = img_h + (C[0] - leftover_h)
        else:
            h = img_h

        if (leftover_w != 0):
            w = img_w + (C[1] - leftover_w)
        else:
            w = img_w

        if (leftover_s != 0):
            s = img_s + (C[2] - leftover_s)
        else:
            s = img_s

        tmp_full_imgs = np.zeros((bachsize, h, w, s))
        tmp_full_imgs[:, img_h, :img_w, 0:img_s] = img
        print("Padded images shape: " + str(img.shape))
def find_random_crop_dim(full_vol_dim, crop_size):
    assert full_vol_dim[0] >= crop_size[0], "crop size is too big"
    assert full_vol_dim[1] >= crop_size[1], "crop size is too big"
    assert full_vol_dim[2] >= crop_size[2], "crop size is too big"

    if full_vol_dim[0] == crop_size[0]:
        slices = crop_size[0]
    else:
        slices = np.random.randint(full_vol_dim[0] - crop_size[0])

    if full_vol_dim[1] == crop_size[1]:
        w_crop = crop_size[1]
    else:
        w_crop = np.random.randint(full_vol_dim[1] - crop_size[1])

    if full_vol_dim[2] == crop_size[2]:
        h_crop = crop_size[2]
    else:
        h_crop = np.random.randint(full_vol_dim[2] - crop_size[2])

    return (slices, w_crop, h_crop)
def load_crop_to_mage(data_label, type=None,  viz3d=False,  normalization='full_volume_mean',
                       clip_intenisty=True,  ):
    # img_nii = nib.load(path)
    img_np = np.squeeze(data_label.get_fdata(dtype=np.float32))

    if viz3d:
        return torch.from_numpy(img_np)#######当viz3d为true时，下面的则不执行了，执行return torch。from_numpy(img_np)

    # 1. Intensity outlier clipping
    if clip_intenisty and type != "label":
        img_np = percentile_clip(img_np)

    # 3. intensity normalization
    img_tensor = torch.from_numpy(img_np)

    MEAN, STD, MAX, MIN = 0., 1., 1., 0.
    if type != 'label':
        MEAN, STD = img_tensor.mean(), img_tensor.std()
        MAX, MIN = img_tensor.max(), img_tensor.min()
    if type != "label":
        img_tensor = normalize_intensity(img_tensor, normalization=normalization, norm_values=(MEAN, STD, MAX, MIN))
    return img_tensor
    # img_tensor = crop_img(img_tensor, crop_size, crop)
    # TODO nib保存（img_np应该是数组）
    # arr1 = np.flip(arr, axis=1)###图像左右调换
    # new_image = nib.Nifti1Image(img_np, np.eye(4))
    # nib.save(new_image, r'D:\Work\Datasets\GoldNormaldatas20\nibdata0411.nii.gz')

    # return img_tensor
def normalize_intensity(img_tensor, normalization="full_volume_mean", norm_values=(0, 1, 1, 0)):
    """
    Accepts an image tensor and normalizes it
    :param normalization: choices = "max", "mean" , type=str
    """
    if normalization == "mean":
        mask = img_tensor.ne(0.0)
        desired = img_tensor[mask]
        mean_val, std_val = desired.mean(), desired.std()
        img_tensor = (img_tensor - mean_val) / std_val
    elif normalization == "max":
        max_val= torch.max(img_tensor)
        img_tensor = img_tensor / max_val
    elif normalization == 'brats':
        # print(norm_values)
        normalized_tensor = (img_tensor.clone() - norm_values[0]) / norm_values[1]
        final_tensor = torch.where(img_tensor == 0., img_tensor, normalized_tensor)
        final_tensor = 100.0 * ((final_tensor.clone() - norm_values[3]) / (norm_values[2] - norm_values[3])) + 10.0
        x = torch.where(img_tensor == 0., img_tensor, final_tensor)
        return x

    elif normalization == 'full_volume_mean':
        img_tensor = (img_tensor.clone() - norm_values[0]) / norm_values[1]

    elif normalization == 'max_min':
        img_tensor = (img_tensor - norm_values[3]) / ((norm_values[2] - norm_values[3]))

    elif normalization == 'MAX':
        img_tensor = img_tensor/norm_values[2]
    return img_tensor
def percentile_clip(img_numpy, min_val=0.1, max_val=99.8):
    """
    Intensity normalization based on percentile
    Clips the range based on the quarile values.
    :param min_val: should be in the range [0,100]
    :param max_val: should be in the range [0,100]
    :return: intesity normalized image
    """
    low = np.percentile(img_numpy, min_val)
    high = np.percentile(img_numpy, max_val)

    img_numpy[img_numpy < low] = low
    img_numpy[img_numpy > high] = high
    return img_numpy
def crop_img(img_tensor, crop_size, crop):
    if crop_size[0] == 0:
        return img_tensor
    w_crop, h_crop ,slices_crop,= crop
    dim1, dim2, dim3 = crop_size
    inp_img_dim = img_tensor.dim()
    assert inp_img_dim >= 3
    if img_tensor.dim() == 3:
        full_dim1, full_dim2, full_dim3 = img_tensor.shape
    elif img_tensor.dim() == 4:
        _, full_dim1, full_dim2, full_dim3 = img_tensor.shape
        img_tensor = img_tensor[0, ...]

    if full_dim1 == dim1:
        img_tensor = img_tensor[:, w_crop:w_crop + dim2,
                     h_crop:h_crop + dim3]
    elif full_dim2 == dim2:
        img_tensor = img_tensor[slices_crop:slices_crop + dim1, :,
                     h_crop:h_crop + dim3]
    elif full_dim3 == dim3:
        img_tensor = img_tensor[slices_crop:slices_crop + dim1, w_crop:w_crop + dim2, :]
    else:
        img_tensor = img_tensor[w_crop:w_crop + dim1, h_crop:h_crop + dim2,
                     slices_crop:slices_crop + dim3]

    if inp_img_dim == 4:
        return img_tensor.unsqueeze(0)

    return img_tensor
def fix_seg_map(segmentation_map, dataset="iseg2017"):
    if dataset == "iseg2017" or dataset == "iseg2019":
        label_values = [0, 10, 150, 250]
        for c, j in enumerate(label_values):
            segmentation_map[segmentation_map == j] = c

    elif dataset == "brats2018" or dataset == "brats2019" or dataset == "brats2020":
        ED = 2
        NCR = 1
        NET_NCR = 1
        ET = 3
        # print('dsdsdsdsd')
        segmentation_map[segmentation_map == 1] = NET_NCR
        segmentation_map[segmentation_map == 2] = ED
        segmentation_map[segmentation_map == 3] = 3
        segmentation_map[segmentation_map == 4] = 3
        segmentation_map[segmentation_map >= 4] = 3
    elif dataset == "mrbrains4":
        GM = 1
        WM = 2
        CSF = 3
        segmentation_map[segmentation_map == 1] = GM
        segmentation_map[segmentation_map == 2] = GM
        segmentation_map[segmentation_map == 3] = WM
        segmentation_map[segmentation_map == 4] = WM
        segmentation_map[segmentation_map == 5] = CSF
        segmentation_map[segmentation_map == 6] = CSF

    elif dataset == "Vessel":
        segmentation_map[segmentation_map > 0] = 1

    return segmentation_map
def find_non_zero_labels_mask(segmentation_map, th_percent, crop_size, crop):
    d1, d2, d3 = segmentation_map.shape
    # segmentation_map[segmentation_map > 0] = 1
    total_voxel_labels = crop_size[0]*crop_size[1]*crop_size[2]

    # cropped_segm_map = crop_img(segmentation_map, crop_size, crop)
    crop_voxel_labels = segmentation_map.sum()

    label_percentage = crop_voxel_labels / total_voxel_labels###计算每个块中的目标区域占比
    # print(label_percentage,total_voxel_labels,crop_voxel_labels)
    if label_percentage >= th_percent:
        return True
    else:
        return False
if __name__ == '__main__':
    print('start norm patches !!!')
    ###TODO 参数

    normalization='full_volume_mean'
    ori_image=r'H:\Datasets\ixi45\adjust_contrast\\'
    new_image=r'H:\Datasets\ixi45\adjust_contrast\nomalize\\'

    # Exsit_training_path=r'E:\MRA_public\final_datasets\\'
    # pathname = r'E:\MICCAI_BraTS2020_TrainingData~\MICCAI_BraTS2020_TrainingData\\'
    print("############====Images Norm======#############")
    image_norm(ori_image,new_image,normalization)
    # print("############====Crop Order======#############")
    # create_order_train_direct_volumes(crop_size, normalization, part_vol_path)

