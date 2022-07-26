import os
import glob
import numpy as np
import nibabel as nib

from utils.dice_coefficient import DiceAverage
def dice_evalute(pred, target):
    predList = sorted(glob.glob(os.path.join(pred, '*.nii.gz')))
    labelList = sorted(glob.glob(os.path.join(target, '*.nii.gz')))
    dice = DiceAverage(2)
    # assert  len(predList)==len(labelList)
    total=len(predList)
    for i in range (total):
        print(predList[i],labelList[0])
        preddata=predList[i]
        labeldata = labelList[i]
        ##TODO TEST 单个文件测试时用
        # preddata =r'D:\other\pre\connect_processNormal001-MRA_brain.nii.gz'
        # labeldata =r'D:\other\label\Normal001-MRA_brain.nii.gz'
        pred_arr=nib.load(preddata).get_fdata()
        label_arr=nib.load(labeldata).get_fdata()
        dice.update(pred_arr,label_arr)
        print("dice is :",dice )
#
# def update(self, logits, targets):
#         self.val = DiceAverage.get_dices(logits, targets)
#         self.sum += self.val
#         self.count += 1
#         self.avg = np.around(self.sum / self.count, 4)##########np.around()输入矩阵self.sum/self.count,保留小数点右边4位小数
#         # print(self.avg)
#
#     @staticmethod
#     def get_dices(logits, targets):
#         dices = []
#         for class_index in range(targets.size()[1]):#########取target(1,2,16,16,16)中的2是代表几分类?????????
#             inter = torch.sum(logits[:, class_index, :, :, :] * targets[:, class_index, :, :, :])
#             union = torch.sum(logits[:, class_index, :, :, :]) + torch.sum(targets[:, class_index, :, :, :])
#             dice = (2. * inter + 1) / (union + 1)
#             dices.append(dice.item())
#         return np.asarray(dices)


if __name__ == '__main__':
    # data = 'output/'  # 分割结果地址，图像为nii.gz
    # pred = r'D:\Work\Tools_Project_zengxq\output\\'  # 移除假阳性后保存地址D:\Work\Tools_Project_zengxq\output\result_test  D:\Work\Tools_Project_zengxq\output_remove\\
    # pred = r'D:\Unet2D_vessel_segmentation\result\\'#D:\Unet2D_vessel_segmentation\connect_result\
    pred = r'D:\other\pre\\'#D:\Unet2D_vessel_segmentation\connect_result\

    target=r'D:\other\label\\'
    dice_evalute(pred, target)