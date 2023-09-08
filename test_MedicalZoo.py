# """
# 在测试集目录中进行测试，给出性能评价指标和可视化结果
# """
# from torch.utils.data import DataLoader
# import torch
# from tqdm import tqdm
# from scipy import ndimage
# import config
# from utils import logger, weights_init, metrics,common
# from dataset.dataset_lits_test import Test_Datasets,to_one_hot_3d,Recompone_tool
# import SimpleITK as sitk
# import os
# import numpy as np
# from models.UNet import UNet3D
# from utils.common import load_file_name_list
# from utils.metrics import DiceAverage
# from collections import OrderedDict
# def test(model ,image_dataset,args):
#     dateloader=DataLoader()
#     model.eval()##############eval() 函数用来执行一个字符串表达式，并返回表达式的值。
#     test_data_path = 'D:\\Work\\1114Unet\\3DUNet-Pytorch-master_V0\\3DUNet-Pytorch-master\\brain_vessel\\test\\data\\'
#     label_test_path = 'D:\\Work\\1114Unet\\3DUNet-Pytorch-master_V0\\3DUNet-Pytorch-master\\brain_vessel\\test\\label\\'
#     result_save_path = './output/{}/result'.format(args.save)
#     save_tool = Recompone_tool(image_dataset.ori_shape, image_dataset.new_shape, args.n_labels, image_dataset.cut_param)
#     target = torch.from_numpy(np.expand_dims(image_dataset.label_np, axis=0)).long()
#     target = to_one_hot_3d(target, args.n_labels)
#
# if __name__ == '__main__':
#     # test_data_path = 'D:\\Work\\1114Unet\\3DUNet-Pytorch-master_V0\\3DUNet-Pytorch-master\\brain_vessel\\test\\data\\'
#     # label_test_path = 'D:\\Work\\1114Unet\\3DUNet-Pytorch-master_V0\\3DUNet-Pytorch-master\\brain_vessel\\test\\label\\'
#     # result_save_path = './output/{}/result'.format(args.save)
#

"""
在测试集目录中进行测试，给出性能评价指标和可视化结果
"""
#TODO 根据实际情况调整,但整体结构不变
# from torch.utils.data import DataLoader
# import torch
# from tqdm import tqdm
# import torch.nn as nn
# from scipy import ndimage
# import config
# from utils import logger, weights_init, metrics, common
# from dataset.new_test_datasets import Test_Datasets, to_one_hot_3d, Recompone_tool
# import SimpleITK as sitk
# import os
# import numpy as np
# from models.UNet import UNet3D
# from utils.common import load_file_name_list
# from utils.metrics import DiceAverage
# from losses3D import DiceLoss
# from collections import OrderedDict
# from models import unet3d
# from models import new_UNet3D

def test(model, img_dataset, args):
    n_labels = args.n_labels
    dataloader = DataLoader(dataset=img_dataset, batch_size=8, num_workers=0, shuffle=False)#
    ##将裁成16，16，16大小的path----data使用DataLoader装载
    model.eval()
    # test_dice = DiceAverage(n_labels)
    print(img_dataset.nomalize_image_shape, img_dataset.patches_shape, n_labels, img_dataset.cut_param)
    ####通过recompone_tool工具得到重建的图像数组信息
    save_tool = Recompone_tool(img_dataset.nomalize_image_shape,img_dataset.nomalize_image_shape, n_labels, img_dataset.cut_param)

    ####将label的值赋给target进行dice准确度的计算#############
    test_dice = DiceAverage(n_labels)
    # test_dice=DiceLoss(n_labels)
    fp = open(r'D:\Work\Datasets\16x16x16_7%\test\nii\test\p.txt', "a+")  # a+ 如果文件不存在就创建。存在就在文件内容的后面继续追加
    target = torch.from_numpy(np.expand_dims(img_dataset.label_np, axis=0)).long()
    target1 = to_one_hot_3d(target, n_labels)
    with torch.no_grad():#######dataloader（1，16，16，16）-->(1,1,16,16,16)最前面的是batchsize1
        for data in tqdm(dataloader, total=len(dataloader)):
            data = data.unsqueeze(1)#由1,176,128,128 增加一维为1,1,176,128,128(列扩充)
            data = data.float().to(device)
            output = model(data)###将data送入到model中得到output
            # =====================================================
            # output_z=output[0]#print(8,2,16,16,16)
            # output0=output_z[0,:,:,:]#channal 0
            # output1=output_z[1,:,:,:]#channal 1
            #
            #
            # output0_img = output0.swapaxes(0, 2)  ###交换x,z  调整方向
            # output0_img = output0_img.cpu()
            # output0_img = sitk.GetImageFromArray(np.squeeze(np.array(output0_img.numpy(), dtype='float32')))
            # ######################################################先转为numpy(),再转为数组array，再减少维数送入sitk中处理
            # tes_result_save_path = r'D:\Work\Datasets\16x16x16_7%\test\nii\test\result\\'
            #
            # dataimage = r'D:\Work\Datasets\Data_augmentation\new_datasets\test\data\Normal074-MRA_brain.nii.gz'
            # # TODO 获取图像坐标SPCING信息
            # data = sitk.ReadImage(dataimage)
            # ##TODO 赋予原图的图像信息：方向，原坐标，层间距
            # output0_img.SetDirection(data.GetDirection())  ###########图像方向不变
            # output0_img.SetOrigin(data.GetOrigin())  ###########图像原点不变
            # output0_img.SetSpacing((data.GetSpacing()[0], data.GetSpacing()[1], data.GetSpacing()[2]))
            #
            # # sitk.WriteImage(output0_img, os.path.join(tes_result_save_path, 'best result-' + file_idx))
            # sitk.WriteImage(output0_img, os.path.join(tes_result_save_path,
            #                                        'new normal_output0_1img' +  '.nii.gz'))
            #
            # # =========
            # output1_img = output1.swapaxes(0, 2)  ###交换x,z  调整方向
            # output1_img = output1_img.cpu()
            # output1_img = sitk.GetImageFromArray(np.squeeze(np.array(output1_img.numpy(), dtype='float32')))
            # ######################################################先转为numpy(),再转为数组array，再减少维数送入sitk中处理
            #
            # ##TODO 赋予原图的图像信息：方向，原坐标，层间距
            # output1_img.SetDirection(data.GetDirection())  ###########图像方向不变
            # output1_img.SetOrigin(data.GetOrigin())  ###########图像原点不变
            # output1_img.SetSpacing((data.GetSpacing()[0], data.GetSpacing()[1], data.GetSpacing()[2]))
            #
            # # sitk.WriteImage(output0_img, os.path.join(tes_result_save_path, 'best result-' + file_idx))
            # sitk.WriteImage(output1_img, os.path.join(tes_result_save_path,
            #                                           'new normal_output1_1img' + '.nii.gz'))
            #
            #
            # # ========
            # output_z_img = output_z.cpu()
            # output_z_img = torch.argmax(output_z_img, dim=0)  ##返回dim维度上张量最大值的索引
            # output_z_img = output_z_img.swapaxes(0,2)  ###交换x,z  调整方向
            #
            # output_z_img = sitk.GetImageFromArray(np.squeeze(np.array(output_z_img.numpy(), dtype='float32')))
            # ######################################################先转为numpy(),再转为数组array，再减少维数送入sitk中处理
            # tes_result_save_path = r'D:\Work\Datasets\16x16x16_7%\test\nii\test\result\\'
            #
            # dataimage = r'D:\Work\Datasets\Data_augmentation\new_datasets\test\data\Normal074-MRA_brain.nii.gz'
            # # TODO 获取图像坐标SPCING信息
            # data = sitk.ReadImage(dataimage)
            # ##TODO 赋予原图的图像信息：方向，原坐标，层间距
            # output_z_img.SetDirection(data.GetDirection())  ###########图像方向不变
            # output_z_img.SetOrigin(data.GetOrigin())  ###########图像原点不变
            # output_z_img.SetSpacing((data.GetSpacing()[0], data.GetSpacing()[1], data.GetSpacing()[2]))
            #
            # # sitk.WriteImage(output0_img, os.path.join(tes_result_save_path, 'best result-' + file_idx))
            # sitk.WriteImage(output_z_img, os.path.join(tes_result_save_path,
            #                                           'new normal_output_z_1img' + '.nii.gz'))
            # # ===========================================================




            # print("output shape is ",output.shape)
            # # ===============PR,ACC,RECALL,F1==============================
            # output1=output
            # output1[output1>0.5]=1
            # output1[output1<0.5]=0
            # target1=target
            # # A=output1&target1
            # TP = (output1 * target1).sum()
            # FP = output1.sum() - TP
            # FN = target1.sum() - TP
            # TN = (output1.size(0) * output1.size(1) * output1.size(2) * 4096) - TP - FP - FN
            # print("\nTP is :",TP,"\nFP is :",FP,"\nFN is :",FN,"\nTN is :",TN)
            # exp=0.00001
            # Precision = TP / (TP + FP+exp)
            # Accuracy = (TP + TN) / (TP + TN + FP + FN+exp)
            # Recall = TP / (TP + FN+exp)
            # F1 = (2 * Precision * Recall) / (Precision + Recall+exp)
            # IOU = TP / (TP + FN + FP)
            # Dice = 2 * TP / ((TP + FN) + (TP + FP))
            # print("Precision =", str(Precision), "\n Accuracy =", str(Accuracy), "\n Recall =", str(Recall), "\n F1 =",
            #       str(F1),
            #       "\n IOU =", str(IOU), "\n DICE =", str(Dice))
            # =========================================================================================#

            # print(full_prob,file=fp)
            # fp.close()
            save_tool.add_result(output.detach().cpu())##将output的数据参数添加新的save_tool中去
          ### detach假设有模型A和模型B，我们需要将A的输出作为B的输入，但训练时我们只训练模型B.那么可以这样做：
            # 例如：input_B = output_A.detach()
            #执行完循环之后data=result为：torch.size(34,2,176,128,128)

    pred = save_tool.recompone_overlap()#########将save_tool中的数据拼接起来返回一个image值给pred

    # pred = torch.nn.functional.interpolate(data, scale_factor=(1//img_dataset.slice_down_scale,1//img_dataset.xy_down_scale,1//img_dataset.xy_down_scale), \
    #                                  mode='trilinear', align_corners=False) # 空间分辨率恢复到原始size
    pred = torch.unsqueeze(pred, dim=0)#行扩充
    # # ===============PR,ACC,RECALL,F1==============================
    # pred1 = pred
    # pred1[pred1 > 0.5] = 1
    # pred1[pred1 < 0.5] = 0
    # target1 = target
    # TP = (pred1 * target1).sum()
    # FP = pred1.sum() - TP
    # FN = target1.sum() - TP
    # TN = (pred1.size(0) * pred1.size(1) * pred1.size(2) * 4096) - TP - FP - FN
    # print("\nTP is :", TP, "\nFP is :", FP, "\nFN is :", FN, "\nTN is :", TN)
    # exp = 0.0000001
    # Precision = TP / (TP + FP + exp)
    # Accuracy = (TP + TN) / (TP + TN + FP + FN + exp)
    # Recall = TP / (TP + FN + exp)
    # F1 = (2 * Precision * Recall) / (Precision + Recall)
    # print("Precision =", str(Precision), "\n Accuracy =", str(Accuracy), "\n Recall =", str(Recall), "\n F1 =",
    #       str(F1))
    # # =========================================================================================#

###############此时target，pred的维度均为5维
    print("pred shape is ",pred.shape)
    test_dice.update(pred, target1)
    print("test_dice is :",test_dice )
#########################################调用self值格式：class名.self名
    test_dice= OrderedDict({ 'Test_BackGround dice0': test_dice.avg[0]*100,'Test_Vessel dice1': test_dice.avg[1]*100})
    # test_dice= OrderedDict({'Test dice': test_dice.avg[1]})

    # else:
    #     test_dice = OrderedDict(
    #         {'Test dice0': test_dice.avg[0], 'Test dice1': test_dice.avg[1], 'Test dice2': test_dice.avg[2]})

    pred_img = torch.argmax(pred, dim=1)##返回dim维度上张量最大值的索引
    # save_tool.save(filename)
    return  pred_img,test_dice

if __name__ == '__main__':
    args = config.args
    cuda="True"
    device = torch.device('cuda')
    #path                                                                           #D:\Work\Datasets\Data_augmentation\new_datasets\test\
    test_data_path = r'D:\Work\Datasets\Data_augmentation\new_datasets\test\\'          #val D:\Work\Datasets\Data_augmentation\new_datasets\val
    result_save_path = './output/{}/result'.format(args.save)
    tes_result_save_path = r'D:\Work\Datasets\16x16x16_7%\test\nii\test\result\\'
    dataimage=r'D:\Work\Datasets\Data_augmentation\new_datasets\test\data\Normal074-MRA_brain.nii.gz'
    #TODO 获取图像坐标SPCING信息
    data = sitk.ReadImage(dataimage)

    if not os.path.exists(result_save_path):
        os.mkdir(result_save_path)
    model = unet3d.UNet(in_channels=1, n_classes=2, base_n_filter=8)  ####2
    #选择是否gpu
    if cuda:
        model = model.cuda()
        print("Model transferred in GPU.....")

    #D:\Work\Datasets\16x16x16_7%\test\nii\test       D:\Work\Datasets\16x16x16_7%\test\nii\test\\
    #########加载路径##########

    # D:\Work\Datasets\GoldNormaldatas20\test               D:\Work\Datasets\16x16x16_7%\test\nii\test\

    # test_data_path=r'D:\Work\Datasets\samples\16x16x16_7%\test\nii\test\\'
    #r'D:\Work\Datasets\samples\16x16x16_7%\test\nii\test\\'#D:\Work\Datasets\samples\16x16x16_7%\test\nii\test\
    # test_data_path =r'D:\Work\Datasets\samples\16x16x16_7%\test\nii\test_none\nii\\'#D:\Work\Datasets\samples\16x16x16_7%\test_10%patch\test_vol_all_nii\\(完整一个文件)



    ####################加载模型#########################
    # model info
    # model = UNet3D(in_channels=1, filter_num_list=[16, 32, 48, 64, 96], class_num=args.n_labels).to(device)

    # ckpt = torch.load('./output/{}/best_model.pth'.format(args.save))
    # model.load_state_dict(ckpt['net'])
    # model = new_UNet3D.new_UNet3D(in_channels=1, n_classes=2, base_n_filter=8)
    # ckpt = torch.load('./output/{}/UNET3D_21_03___13_52_vessel__BEST.pth'.format(args.save))#UNET3D_22_03___12_57_vessel__BEST.pth#$
    #UNET3D_12_04___03_55_vessel__last_epoch.pth        58.19 UNET3D_09_04___15_09_vessel__BEST.pth
    ckpt = torch.load('./output/{}/UNET3D_15_04___05_25_vessel__BEST.pth'.format(args.save))#UNET3D_13_04___01_19_vessel__BEST.pth  70.8
###UNET3D_13_04___16_10_vessel__BEST.pth    ---55.7     UNET3D_14_04___08_12_vessel__BEST.pth ----71.6
   # UNET3D_15_04___05_25_vessel__BEST.pth----0.8 for (74)
    model.load_state_dict(ckpt['model_state_dict'])
    ########保存log函数###########################################
    test_log = logger.Test_Logger(result_save_path, "test_log")

    # data info###144，224，176#########
    # #当patch-s/h/w=stride-s/c/w时，则是按顺序取，不重合，若patch>stride则会重复取，若patch<stride则是等间隔抽取，会遗漏信息
    # cut_param = {'patch_s': 16, 'patch_h': 16, 'patch_w': 16,############注意不能超过原图想范围
    #              'stride_s': 8, 'stride_h':8, 'stride_w':8}###############DICE=0.4524
    datasets = Test_Datasets(test_data_path,  args=args)###得到Test_Datasets所有参数---
    for img_dataset, file_idx in datasets:#########img_dateset遍历datasets,包括由Test_Datasets返回的Img_datasets所返回的所有参数：可截图记录方便对比验证

        # file_idx表示str,文件路径+文件名

        pred_img,test_dice = test(model, img_dataset, args)
        # test_log.update(file_idx)
        test_log.update(file_idx, test_dice)
        pred_img=pred_img.swapaxes(1,3)###交换x,z  调整方向
        pred_img = sitk.GetImageFromArray(np.squeeze(np.array(pred_img.numpy() ,dtype='uint8'), axis=0))
        ######################################################先转为numpy(),再转为数组array，再减少维数送入sitk中处理
        ##TODO 赋予原图的图像信息：方向，原坐标，层间距
        pred_img.SetDirection(data.GetDirection())  ###########图像方向不变
        pred_img.SetOrigin(data.GetOrigin())  ###########图像原点不变
        pred_img.SetSpacing((data.GetSpacing()[0], data.GetSpacing()[1], data.GetSpacing()[2]))

        # sitk.WriteImage(pred_img, os.path.join(tes_result_save_path, 'best result-' + file_idx))
        sitk.WriteImage(pred_img, os.path.join(tes_result_save_path, 'new normal_result-415_val' + img_dataset.data_path.split('\\')[-1]+'.nii.gz'))


#########test log############
# OrderedDict([('Test Loss', array([0.9991, 0.297 ])), ('Test dice0', 0.9991), ('Test dice1', 0.297)])
#######20220311-13：30   result 001--->0.4578
                        # result 081--->