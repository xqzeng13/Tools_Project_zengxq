'''
后处理：使用连通域进行后处理
'''

import cc3d
import nibabel as nib
from pathlib2 import Path
from tqdm import tqdm
import numpy as np
import os
import glob
import cv2
import SimpleITK as sitk

from scipy.ndimage.morphology import binary_fill_holes

from skimage.measure import label
import matplotlib.pyplot as plt


def connec_main(data, output, target):
    data = Path(data).resolve()
    output = Path(output).resolve()  ##把一个路径解析为绝对路径
    target = Path(target).resolve()

    # assert data != output, f'postprocess data will replace original data, use another output path'
    dataimage = r'D:\Work\Datasets\GoldNormaldatas20\label\Normal074-MRA.nii.gz'
    # TODO 获取图像坐标SPCING信息
    data1 = sitk.ReadImage(dataimage)

    if not output.exists():
        output.mkdir(parents=True)
    # data=r'D:\Work\Tools_Project_zengxq\output\new normal_result-422_x16Normal074-MRA_brain.nii'
    # predictions = sorted(data.glob('*.nii.gz'))
    predictions_list = sorted(glob.glob(os.path.join(data, '*.nii.gz')))
    # predictions = sorted(glob.glob(os.path.join(data, '*.nii.gz')))
    target_list = sorted(glob.glob(os.path.join(target, '*.nii.gz')))

    # target=sorted(target.glob('*.nii.gz'))

    # for pred in tqdm(predictions):
    for N in range(len(predictions_list)):
        # if not pred.name.startswith('.'):
        name=predictions_list[N].split('Fusion')[1]
        pred = predictions_list[N]
        target = target_list[N]

        vol_nii = nib.load(pred)
        target_nii = nib.load(target)

        ########-----------------------测试
        # vol_nii = nib.load(r'D:\Work\Tools_Project_zengxq\output\result\patch_result-dilate_1.nii.gz')
        #######----------------------------------
        affine = vol_nii.affine
        vol = vol_nii.get_fdata()
        target = target_nii.get_fdata()

        vol = post_processing(vol, target)
        vol = vol.swapaxes(0, 2)
        pred_img = sitk.GetImageFromArray((np.array(vol, dtype='uint8')))
        pred_img.SetDirection(data1.GetDirection())  ###########图像方向不变
        pred_img.SetOrigin(data1.GetOrigin())  ###########图像原点不变
        pred_img.SetSpacing((data1.GetSpacing()[0], data1.GetSpacing()[1], data1.GetSpacing()[2]))
        sitk.WriteImage(pred_img, os.path.join(r'D:\other\pre\\',
                                               'connect_process' +name))

        # vol_hole=hole_filling(vol, 0, 100, fill_2d=True)
        #
        #
        # vol_hole=vol_hole.swapaxes(0,2)
        # pred_img = sitk.GetImageFromArray((np.array(vol_hole, dtype='uint8')))
        # pred_img.SetDirection(data1.GetDirection())  ###########图像方向不变
        # pred_img.SetOrigin(data1.GetOrigin())  ###########图像原点不变
        # pred_img.SetSpacing((data1.GetSpacing()[0], data1.GetSpacing()[1], data1.GetSpacing()[2]))
        # sitk.WriteImage(pred_img, os.path.join(r'D:\Work\Tools_Project_zengxq\output\\',
        #                                        'patch_result-' + '.nii.gz'))

        # vol = post_processing(vol)
        # vol_nii = nib.Nifti1Image(vol_hole, affine)
        #
        # vol_nii_filename = output /"fill"+ pred.name
        # vol_nii.to_filename(str(vol_nii_filename))


def post_processing(vol, target_arr):
    vol_ = vol.copy()###值传递     vol_1 = vol#址传递
    vol_1 = vol.copy()

    vol_[vol_ > 0] = 1
    vol_ = vol_.astype(np.int64)
    vol_cc = cc3d.connected_components(vol_, connectivity=26)  ####得到所有的区域赋予不同灰度值（同一连通域相同灰度值）
    # ###-----
    # dataimage = r'D:\Work\Datasets\GoldNormaldatas20\label\Normal074-MRA.nii.gz'
    # # TODO 获取图像坐标SPCING信息
    # data1 = sitk.ReadImage(dataimage)
    # # out_arr_dilate = out_arr_dilate.swapaxes(0, 2)
    # pred_img1 = sitk.GetImageFromArray((np.array(vol_cc, dtype='uint8')))
    # pred_img1.SetDirection(data1.GetDirection())  ###########图像方向不变
    # pred_img1.SetOrigin(data1.GetOrigin())  ###########图像原点不变
    # pred_img1.SetSpacing((data1.GetSpacing()[0], data1.GetSpacing()[1], data1.GetSpacing()[2]))
    # sitk.WriteImage(pred_img1, os.path.join(r'D:\Work\Tools_Project_zengxq\output\result\\',
    #                                         'test_result' + '.nii.gz'))
    # #------------------------------------------

    ##cc_sum 表示所有的连通域，i=1表示灰度值为1，vol_cc[vol_cc == i].shape[0]表示该连通域i=1有多少个
    cc_sum = [(i, vol_cc[vol_cc == i].shape[0]) for i in range(vol_cc.max() + 1)]
    cc_sum.sort(key=lambda x: x[1], reverse=True)  ###排序，按照连通域的数量
    cc_sum.pop(0)  # remove background
    ##筛选出需要移除的连通域
    reduce_cc = [cc_sum[i][0] for i in range(1, len(cc_sum)) if cc_sum[i][1] < cc_sum[0][1] * 0.1]
###TODO1
    for i in reduce_cc:
        vol[vol_cc == i] = 0  #######得到最大连通域
   # TODO ##将需要剔除的连通域值赋为10
#     post_cc = vol.copy()
#     for j in reduce_cc:
#         vol_1[vol_cc == j] = 10  #######得到
# ##TODO2
#     res = vol_1.copy()##
#     res[res == 1] = 0
#     res[res > 2] = 1
#
#     # for j in reduce_cc:
#     #     vol_1[vol_cc!=j] =0
#     #     vol_1[vol_cc==j] =1
#     ##TODO3
#     vol_and = res * target_arr#移除的连通域与target相交，保留相交部分
#     #TODO4
#     vol_final = vol_and + post_cc  ####
#     return vol_final
    return vol

# def hole_filling(bw, hole_min, hole_max, fill_2d=True):
#     bw = bw > 0
#     if len(bw.shape) == 2:
#         background_lab = label(~bw, connectivity=1)
#         fill_out = np.copy(background_lab)
#         component_sizes = np.bincount(background_lab.ravel())
#         too_big = component_sizes > hole_max
#         too_big_mask = too_big[background_lab]
#         fill_out[too_big_mask] = 0
#         too_small = component_sizes < hole_min
#         too_small_mask = too_small[background_lab]
#         fill_out[too_small_mask] = 0
#     elif len(bw.shape) == 3:
#         if fill_2d:
#             fill_out = np.zeros_like(bw)
#             for zz in range(bw.shape[1]):
#                 background_lab = label(~bw[:, zz, :], connectivity=1)  # 1表示4连通， ~bw[zz, :, :]1变为0， 0变为1
#                 # 标记背景和孔洞， target区域标记为0
#                 out = np.copy(background_lab)
#                 # plt.imshow(bw[:, :, 87])
#                 # plt.show()
#                 component_sizes = np.bincount(background_lab.ravel())  # ravel()方法将数组维度拉成一维数组
#                 # 求各个类别的个数
#                 too_big = component_sizes > hole_max
#                 too_big_mask = too_big[background_lab]
#
#                 out[too_big_mask] = 0
#
#                 too_small = component_sizes < hole_min
#                 too_small_mask = too_small[background_lab]
#                 out[too_small_mask] = 0
#                 # 大于最大孔洞和小于最小孔洞的都标记为0， 所以背景部分被标记为0了。只剩下符合规则的孔洞
#                 fill_out[:, zz, :] = out
#                 # 只有符合规则的孔洞区域是1， 背景及target都是0
#         else:
#             background_lab = label(~bw, connectivity=1)
#             fill_out = np.copy(background_lab)
#             component_sizes = np.bincount(background_lab.ravel())
#             too_big = component_sizes > hole_max
#             too_big_mask = too_big[background_lab]
#             fill_out[too_big_mask] = 0
#             too_small = component_sizes < hole_min
#             too_small_mask = too_small[background_lab]
#             fill_out[too_small_mask] = 0
#     else:
#         print('error')
#         return
#
#     return np.logical_or(bw, fill_out)  # 或运算，孔洞的地方是1，原来target的地方也是1


#
# class Point(object):
#  def __init__(self,x,y):
#   self.x = x
#   self.y = y
#
#  def getX(self):
#   return self.x
#  def getY(self):
#   return self.y
#
# def getGrayDiff(img,currentPoint,tmpPoint):
#  return abs(int(img[currentPoint.x,currentPoint.y]) - int(img[tmpPoint.x,tmpPoint.y]))
#
# def selectConnects(p):
#  if p != 0:
#   connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), \
#      Point(0, 1), Point(-1, 1), Point(-1, 0)]
#  else:
#   connects = [ Point(0, -1), Point(1, 0),Point(0, 1), Point(-1, 0)]
#  return connects
#
# def regionGrow(img,seeds,thresh,p = 1):
#  height, weight = img.shape
#  seedMark = np.zeros(img.shape)
#  seedList = []
#  for seed in seeds:
#   seedList.append(seed)
#  label = 1
#  connects = selectConnects(p)
#  while(len(seedList)>0):
#   currentPoint = seedList.pop(0)
#
#   seedMark[currentPoint.x,currentPoint.y] = label
#   for i in range(8):
#    tmpX = currentPoint.x + connects[i].x
#    tmpY = currentPoint.y + connects[i].y
#    if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
#     continue
#    grayDiff = getGrayDiff(img,currentPoint,Point(tmpX,tmpY))
#    if grayDiff < thresh and seedMark[tmpX,tmpY] == 0:
#     seedMark[tmpX,tmpY] = label
#     seedList.append(Point(tmpX,tmpY))
#  return seedMark
#
#
# img = cv2.imread('lean.png',0)
# seeds = [Point(10,10),Point(82,150),Point(20,300)]
# binaryImg = regionGrow(img,seeds,10)
# cv2.imshow(' ',binaryImg)
# cv2.waitKey(0)


if __name__ == '__main__':
    data = r'D:\other\pre\\'  # 分割结果地址，图像为nii.gz
    target = r'D:\Unet2D_vessel_segmentation\target\\'
    output = r'D:\other\pre\\'  # 移除假阳性后保存地址
    if not os.path.exists(output):
        os.makedirs(output)
    connec_main(data, output, target)
    # hole_filling(arr, 0, 100, fill_2d=True)
