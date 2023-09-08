import SimpleITK as sitk
import skimage.io as io
import glob
import os
import numpy as np
import nibabel as nib
import torch
from utils.data_intensity_normalization import load_crop_to_mage


# def read_img(path):
#     img = sitk.ReadImage(path)
#     # 查看图片深度
#     print(img.GetDepth())
#     # 查看Size
#     print(img.GetSize())
#     data = sitk.GetArrayFromImage(img)
#     return data
#

# 显示一个系列图
def datasets(data,pathname):
    # if model=='data':
        for s in range(data.shape[2]):#data.shape[2]
            # io.imshow(data[64:384, 64:420,i], cmap='gray')#层*高*宽
            data_npys=data[:, :,s]#64:384, 64:420,s
            filenames=pathname+'\s_file\\'+'s'+str(s)+'__'+str(N)
            f_datas=filenames + str(0)+'.npy'
            np.save(f_datas, data_npys)


        for h in range(data.shape[1]):#(80,352,1)
            # data_npy=data[:, :,i]#层*高*宽

            # io.imshow(data[i,:,:], cmap='gray')  # 层*高*宽
            data_npyh=data[h,:,:]
            filenameh=pathname+'\h_file\\'+'h'+str(h)+'__'+str(N)
            f_datah=filenameh + str(0)+'.npy'
            np.save(f_datah, data_npyh)



        for w in range(data.shape[0]):#(48,416,1)
            # io.imshow(data[80:368, w,:], cmap='gray')#层*高*宽
            data_npyw= data[:, w,:]#80:368, w,:
            filenamew= pathname+'\w_file\\' + 'w'+str(w) + '__' + str(N)
            f_dataw = filenamew+ str(0) + '.npy'
            np.save(f_dataw, data_npyw)

        print("crop data is over")

def labelsets(label,pathname):
    # if model=='data':
        for s in range(label.shape[2]):
            # io.imshow(data[64:384, 64:420,i], cmap='gray')#层*高*宽
            label_npys=label[:, :,s]#64:384, 64:420,s
            filenames=pathname+'\s_file\\'+'s'+str(s)+'__'+str(N)
            f_segs=filenames + str(0)+'seg.npy'
            np.save(f_segs, label_npys)
    #
        for h in range(label.shape[1]):#80,352,1
            # data_npy=data[:, :,i]#层*高*宽

            # io.imshow(data[i,:,:], cmap='gray')  # 层*高*宽
            label_npyh=label[h,:,:]
            filenameh=pathname+'\h_file\\'+'h'+str(h)+'__'+str(N)
            f_segh=filenameh + str(0)+'seg.npy'
            np.save(f_segh, label_npyh)


        for w in range(label.shape[0]):#48,416,1
            # io.imshow(label[80:368, w,:], cmap='gray')#层*高*宽
            label_npyw= label[:, w,:]#80:368, w,:
            filenamew= pathname+'\w_file\\' + 'w' +str(w)+ '__' + str(N)
            f_segw = filenamew+ str(0) + 'seg.npy'
            np.save(f_segw, label_npyw)
    #   ######深(0-128)
    # # elif model == 'label':
#
# def show_img_s(data,pathname):
#     for i in range(data.shape[2]):
#         # io.imshow(data[64:384, 64:420,i], cmap='gray')#层*高*宽
#         data_npys=data[64:384, 64:420,i]
#         filenames=pathname+'train_s'+'__'+str(N)
#         f_datas=filenames + str(0)+'.npy'
#         np.save(f_datas, data_npys)
#         # np.save()
#         # print(i)
#         # io.show()
#
# ########高(80-352)
# def show_img_h(data,pathname):
#     for i in range(80,352,1):
#         # data_npy=data[:, :,i]#层*高*宽
#
#         # io.imshow(data[i,:,:], cmap='gray')  # 层*高*宽
#         data_npyh=data[i,:,:]
#         filenameh=pathname+'train_h'+'__'+str(N)
#         f_datah=filenameh + str(0)+'.npy'
#         np.save(f_datah, data_npyh)
#         # print(i)
#         # # np.save()
#         # # img_tensor=torch.from_numpy(data_npy)
#         # # pred_img = sitk.GetImageFromArray((np.array(img_tensor.numpy(), dtype='float32')))
#         # # ######################################################先转为numpy(),再转为数组array，再减少维数送入sitk中处理
#         # # sitk.WriteImage(pred_img, os.path.join(r'D:\Work\\', 'patch_result-' + '.nii.gz'))
#         # io.show()
# #####宽(48-416)
# def show_img_w(data,pathname):
#     for i in range(48,416,1):
#         io.imshow(data[80:368, i,:], cmap='gray')#层*高*宽
#         data_npyw= data[80:368, i,:]
#         filenamew= pathname + 'train_h' + '__' + str(N)
#         f_dataw = filenamew+ str(0) + '.npy'
#         np.save(f_dataw, data_npyw)
# #         print(i)
# #         io.show()
# #
# # 单张显示
def show_img1(ori_img):
    io.imshow(ori_img[50], cmap='gray')
    io.show()

if __name__ == '__main__':
    # 文件夹路径
    trainpath = r'D:\Unet2D_vessel_segmentation\Unet-main\save_path\data_normalization\train\\'###手动修改train/val
    # valdatapath = r'D:\Work\Datasets\Data_augmentation\val\data\\'

    data_list = sorted(glob.glob(os.path.join(trainpath, 'data\\' + '*.nii.gz')))
    label_list = sorted(glob.glob(os.path.join(trainpath, 'label\\' + '*.nii.gz')))

    save_path=r'D:\Unet2D_vessel_segmentation\Unet-main\save_path\\'
    # save_path1= r'D:\Work\Unet2D_vessel_segmentation\Unet-main\save_path\\'

    for N in range (len(data_list)):

        print(str(data_list[N]),str(label_list[N]))
        # img = sitk.ReadImage(data_list[N])
        # data = sitk.GetArrayFromImage(img)  # (128, 448, 448)
        # show_img_h(data)

        label_nii = nib.load(label_list[N])#(448, 448, 128)
        img_nii = nib.load(data_list[N])#(448, 448, 128)
        normalization = "full_volume_mean"  # ('max_min', 'full_volume_mean', 'brats', 'max', 'mean')
        # 图像强度增强和归一化
        img_tensor = load_crop_to_mage(img_nii, type="T1",viz3d=True ,normalization=None)
        label_tensor = load_crop_to_mage(label_nii, type="label",viz3d=True, normalization=None)

        # show_img_s(img_tensor)
        # show_img_h(img_tensor)
        # show_img_w(img_tensor)
        datasets(img_tensor,save_path)
        labelsets(label_tensor,save_path)#sys.getsizeof() 占空间大小
        print('Now N is =',N)
    print("dataset is over!")

