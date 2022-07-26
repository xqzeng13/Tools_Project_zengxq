
import glob
import os

import SimpleITK as sitk
from matplotlib import pyplot as plt
from scipy import ndimage
import torchio as tio
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Each instance of tio.Subject is passed arbitrary keyword arguments.
# Typically, these arguments will be instances of tio.Image

def main ():
    ##############路径设置#########
    datapath=r"D:\Work\Datasets\GoldNormaldatas20\\"

    save_image_path=r'D:\Work\Datasets\Data_augmentation\random_datasets\train\data\\'
    save_label_path = r'D:\Work\Datasets\Data_augmentation\random_datasets\train\label\\'

    # if not os.path.exists(savepath):
    if  not os.path.exists(save_image_path):  # 创建保存目录
        os.makedirs(save_image_path )
        os.makedirs(save_label_path )

    data_list = sorted(glob.glob(os.path.join(datapath, 'data\\' + '*.nii.gz')))
    label_list = sorted(glob.glob(os.path.join(datapath, 'label\\' + '*.nii.gz')))
    assert len(data_list) == len(label_list)

    Numbers = len(data_list)
    print('Total numbers of image/label is :', Numbers)

    # for dataimage, labelimage, i in zip(data_list, label_list, range(Numbers)):
    #     ##############数据读取######
    #     print(dataimage, " | {}/{}".format(i + 1, Numbers))
    #     print(labelimage, " | {}/{}".format(i + 1, Numbers))
    #     #######读取data######
    #
    #     # s, h, w = dataimage.shape
    #     data = sitk.ReadImage(dataimage)
    #     data_np = sitk.GetArrayFromImage(data)  ########转化为数组##########
    #
    #
    #     #######读取label################
    #     label = sitk.ReadImage(labelimage)  ####data:order=3////label:order=0#####
    #     label_np = sitk.GetArrayFromImage(label)  ########转化为数组##########
    #
    #     ###################################根据所需尺寸更改，但得注意data/label需要同时同尺度变化##############
    #     ######## Bilinear interpolation （双线性插值） order = 1,    Nearest （最邻近插值） order = 0,
    #     ###############文件保存#########################
    #     ####data#####
    #     new_data = sitk.GetImageFromArray(data_array)
    #     new_data.SetDirection(data.GetDirection())  ###########图像方向不变
    #     new_data.SetOrigin(data.GetOrigin())  ###########图像原点不变
    #     new_data.SetSpacing((data.GetSpacing()[0], data.GetSpacing()[1], data.GetSpacing()[2]))  #####层间距变为()
    #     print(data.GetSpacing()[0], data.GetSpacing()[1], data.GetSpacing()[2])
    #     ########label#####
    #     new_label = sitk.GetImageFromArray(label_array)
    #     new_label.SetDirection(label.GetDirection())  ###########图像方向不变
    #     new_label.SetOrigin(label.GetOrigin())  ###########图像原点不变
    #     # new_label.SetSpacing((label.GetSpacing()[0], label.GetSpacing()[1], s/s1))  #####层间距变为()
    #     new_label.SetSpacing((label.GetSpacing()[0], label.GetSpacing()[1], label.GetSpacing()[2]))
    #     # sitk.WriteImage(new_data, os.path.join(savepath ,'data\\', data))
    #     #######保存######################
    #     sitk.WriteImage(new_data,
    #                     r'.\output\newdatas\\' + 'data\\' + dataimage.split('N')[-1] + '.nii')  # 保存格式（保存文件，路径+文件名+文件格式）
    #     sitk.WriteImage(new_label, r'.\output\newdatas\\' + 'label\\' + labelimage.split('N')[
    #         -1] + '.nii')  # 保存格式（保存文件，路径+文件名+文件格式）
    #     ########################################################################################3

    subjects = []
    Name_data=[]####讲读取的data名字存入Name_data[]中
    for (image_path, label_path,i) in zip(data_list, label_list,range(Numbers)):
        print(image_path, " | {}/{}".format(i + 1, Numbers))
        print(label_path, " | {}/{}".format(i + 1, Numbers))

        name=image_path.split('N')[-1]
        Name_data.append(name)
        # print(Name_data)

        subject = tio.Subject(
            mri=tio.ScalarImage(image_path),
            label=tio.LabelMap(label_path),
        )
        subjects.append(subject)###拼接到一起
    # landmarks = tio.HistogramStandardization.train(self.image_paths)
    ###数据处理变换
    # training_transform=transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
    training_transform = tio.Compose([
        # tio.ToCanonical(),##对原图重排列使之转变为RAS方向轴排布（x轴-左右， y轴-前后，z轴-上下），若需要，通常在变换操作的第一位使用
        # tio.Resample(4),#重采样，改变图像像素的物理尺度，需要指定各个维度的像素长度。允许在重采样之前load一个仿射变换矩阵（作为预先的仿射处理），
                        # 允许指定差值的方法，允许只对图像本身操作（如分类任务），也可以同时对图像和mask操作（分割任务）
        #####空间变换#############
        # tio.CropOrPad((48, 60, 48)),###裁剪/扩充

        #######数据增强#############
        # tio.HistogramStandardization({'mri': landmarks}),
        #同样是基于单张图像的归一化操作，即计算标准差和均值，将像素值强度的分布转化成高斯分布
        # tio.RandomMotion(),##添加动态模糊，mri图像的噪声来源一部分是由于被测试者在采集时的动作导致的，通过当前增强来模拟该场景
        # tio.RandomFlip(),#随机翻转
        tio.RandomAffine(),
        tio.RandomNoise(), # 随机噪声
        ######强度变换
        # tio.ZNormalization(masking_method=tio.ZNormalization.mean),
        # tio.RandomGhosting(),###鬼影
        # tio.ElasticDeformation((0, 0.25),interpolation=2, p=0.1)
        # tio.RandomBiasField(),
        # tio.RandomBlur(),


        # tio.RandomAffine(),#随机仿射变换
        # tio.RandomBiasField(p=0.3),#添加随机偏置场伪影，通常由于mri成像设备的磁场不均匀导致的低频强度变化（可以理解为一侧亮度偏低）

        # tio.OneOf({
        #     tio.RandomAffine(): 0.8,
        #     tio.RandomElasticDeformation(): 0.2,###从给定增强变换序列中随机选择一项执行，允许设置每一项执行的概率
        # }),
        # tio.OneHot(),####转为onehot编码
        ])

    ###参考该链接：https://zhuanlan.zhihu.com/p/391313749
    # print(str(training_transform))
    #########
    subjects_dataset = tio.SubjectsDataset(subjects, transform=training_transform)
    # dataset = tio.SubjectsDataset(subjects)
    training_loader = DataLoader(subjects_dataset,batch_size=8)
    print('Dataset size:', len(subjects_dataset), 'subjects')
    ########plot显示################################################
    # one_subject = subjects_dataset[0]
    # one_subject.plot()
    # print(one_subject)
    # print(one_subject.mri)
    # print(one_subject.label)



# Training epoch
    for i in range (len(data_list)) :
        print("transform", " | {}/{}".format(i + 1, Numbers))
        for subjects_batch in training_loader:
            ##image
            inputs = subjects_batch['mri'][tio.DATA]
            array1= inputs[0][0]#####shape:[352,448,176]
            array1 = array1.swapaxes(0, 2)###############(x,y,z)---->(z,y,x)变换
            ##label
            target =subjects_batch['label'][tio.DATA]
            array2 =target[0][0]
            array2=array2.swapaxes(0,2)
            #arry是x，y，z三个方向的大小，而sitk.GetImageFromArray是z，y，x
            dataimage = r'D:\Work\Datasets\Data_augmentation\new_datasets\test\data\Normal074-MRA_brain.nii.gz'
            # TODO 获取图像坐标SPCING信息
            data = sitk.ReadImage(dataimage)

            image = sitk.GetImageFromArray(array1)####从数组回到图像Size: [176, 448, 352]
            label =sitk.GetImageFromArray(array2)
            # array2=image.GetSize()
            # array2=torch.tensor(array2)
            # image2=sitk.GetImageFromArray(array2)
            # print(image)
            ##TODO 赋予原图的图像信息：方向，原坐标，层间距
            image.SetDirection(data.GetDirection())  ###########图像方向不变
            image.SetOrigin(data.GetOrigin())  ###########图像原点不变
            image.SetSpacing((data.GetSpacing()[0], data.GetSpacing()[1], data.GetSpacing()[2]))

            label.SetDirection(data.GetDirection())  ###########图像方向不变
            label.SetOrigin(data.GetOrigin())  ###########图像原点不变
            label.SetSpacing((data.GetSpacing()[0], data.GetSpacing()[1], data.GetSpacing()[2]))


            if  len(training_transform)==1:
                sitk.WriteImage(image, save_image_path+str(training_transform).split('[')[-1].split('(')[0]+"_N"+ Name_data[i])  # 保存格式（保存文件，路径+文件名+文件格式）
                sitk.WriteImage(label,save_label_path+str(training_transform).split('[')[-1].split('(')[0]+"_N"+ Name_data[i])  # 保存格式（保存文件，路径+文件名+文件格式）
            else:
                sitk.WriteImage(image, save_image_path + "RandomAffine" + "_N" + Name_data[i])  # 保存格式（保存文件，路径+文件名+文件格式）
                sitk.WriteImage(label, save_label_path + "RandomAffine" + "_N" + Name_data[i])  # 保存格式（保存文件，路径+文件名+文件格式）
            # print(inputs)

if __name__ == '__main__':
    print("start>>>>>>")
    main()
    print("transforme is over !")

# import torch
# import torchio as tio
# from torch.utils.data import DataLoader
#
# # Each instance of tio.Subject is passed arbitrary keyword arguments.
# # Typically, these arguments will be instances of tio.Image
# subject_a = tio.Subject(
#     t1=tio.ScalarImage('subject_a.nii.gz'),
#     label=tio.LabelMap('subject_a.nii'),
#     diagnosis='positive',
# )
#
# # Image files can be in any format supported by SimpleITK or NiBabel, including DICOM
# subject_b = tio.Subject(
#     t1=tio.ScalarImage('subject_b_dicom_folder'),
#     label=tio.LabelMap('subject_b_seg.nrrd'),
#     diagnosis='negative',
# )
#
# # Images may also be created using PyTorch tensors or NumPy arrays
# tensor_4d = torch.rand(4, 100, 100, 100)
# subject_c = tio.Subject(
#     t1=tio.ScalarImage(tensor=tensor_4d),
#     label=tio.LabelMap(tensor=(tensor_4d > 0.5)),
#     diagnosis='negative',
# )
#
# subjects_list = [subject_a, subject_b, subject_c]
#
# # Let's use one preprocessing transform and one augmentation transform
# # This transform will be applied only to scalar images:
# rescale = tio.RescaleIntensity(out_min_max=(0, 1))
#
# # As RandomAffine is faster then RandomElasticDeformation, we choose to
# # apply RandomAffine 80% of the times and RandomElasticDeformation the rest
# # Also, there is a 25% chance that none of them will be applied
# spatial = tio.OneOf({
#         tio.RandomAffine(): 0.8,
#         tio.RandomElasticDeformation(): 0.2,
#     },
#     p=0.75,
# )
#
# # Transforms can be composed as in torchvision.transforms
# transforms = [rescale, spatial]
# transform = tio.Compose(transforms)
#
# # SubjectsDataset is a subclass of torch.data.utils.Dataset
# subjects_dataset = tio.SubjectsDataset(subjects_list, transform=transform)
#
# # Images are processed in parallel thanks to a PyTorch DataLoader
# training_loader = DataLoader(subjects_dataset, batch_size=4, num_workers=4)
#
# # Training epoch
# for subjects_batch in training_loader:
#     inputs = subjects_batch['t1'][tio.DATA]
#     target = subjects_batch['label'][tio.DATA]