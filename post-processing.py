import nibabel as nib
import numpy as np

data_path = r'/home/fsl/image_2_label.png'
src = gdal.Open(data_path).ReadAsArray()

n_class0 = np.sum(np.where(src==0))
n_class1 = np.sum(np.where(src==1))
n_class2 = np.sum(np.where(src==2))
n_class3 = np.sum(np.where(src==3))
n_class4 = np.sum(np.where(src==4))
sum = src.shape[0]*src.shape[1]


print("背景：{}，第一类：{}，第二类：{}，第三类：{}，第四类：{}".format(n_class0/sum ,n_class1/sum ,n_class2/sum ,n_class3/sum ,n_class4/sum ))

#
# def area_connection(result, n_class, area_threshold, )
#     """
#     result:预测影像
#     area_threshold：最小连通尺寸，小于该尺寸的都删掉
#     """
#     result = to_categorical(result, num_classes=n_class, dtype='uint8')  # 转为one-hot
#     for i in tqdm(range(n_class)):
#         # 去除小物体
#         result[:, :, i] = skimage.morphology.remove_small_objects(result[:, :, i] == 1, min_size=area_threshold,
#                                                                   connectivity=1, in_place=True)
#         # 去除孔洞
#         result[:, :, i] = skimage.morphology.remove_small_holes(result[:, :, i] == 1, area_threshold=area_threshold,
#                                                                 connectivity=1, in_place=True)
#     # 获取最终label
#     result = np.argmax(result, axis=2).astype(np.uint8)
#
#     return result
