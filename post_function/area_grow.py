# import numpy as np
#
#
# def grow(img, seed, t):
#     """
#     img: ndarray, ndim=3
#         An image volume.
#
#     seed: tuple, len=3
#         Region growing starts from this point.
#     t: int
#         The image neighborhood radius for the inclusion criteria.
#     """
#     seg = np.zeros(img.shape, dtype=np.bool)
#     checked = np.zeros_like(seg)
#
#     seg[seed] = True
#     checked[seed] = True
#     needs_check = get_nbhd(seed, checked, img.shape)
#
#     while len(needs_check) > 0:
#         pt = needs_check.pop()
#
#         # Its possible that the point was already checked and was
#         # put in the needs_check stack multiple times.
#         if checked[pt]: continue
#
#         checked[pt] = True
#
#         # Handle borders.
#         imin = max(pt[0] - t, 0)
#         imax = min(pt[0] + t, img.shape[0] - 1)
#         jmin = max(pt[1] - t, 0)
#         jmax = min(pt[1] + t, img.shape[1] - 1)
#         kmin = max(pt[2] - t, 0)
#         kmax = min(pt[2] + t, img.shape[2] - 1)
#
#         if img[pt] >= img[imin:imax + 1, jmin:jmax + 1, kmin:kmax + 1].mean():
#             # Include the voxel in the segmentation and
#             # add its neighbors to be checked.
#             seg[pt] = True
#             needs_check += get_nbhd(pt, checked, img.shape)
#
#     return seg

# -*- coding:utf-8 -*-
# import cv2
# import numpy as np
# import gdalTools
# from scipy import ndimage as ndi
# from skimage.morphology import remove_small_holes, closing, square, opening, remove_small_objects, watershed
#
# import matplotlib.pyplot as plt
# ####################################################################################
#
#
# #######################################################################################
# class Point(object):
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y
#
#     def getX(self):
#         return self.x
#
#     def getY(self):
#         return self.y
#
#
# connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), Point(0, 1), Point(-1, 1),
#             Point(-1, 0)]
#
#
# #####################################################################################
# # 计算两个点间的欧式距离
# def get_dist(image, seed_location1, seed_location2):
#     l1 = image[seed_location1.x, seed_location1.y]
#     l2 = image[seed_location2.x, seed_location2.y]
#     count = np.sqrt(np.sum(np.square(l1 - l2)))
#     return count
#
#
# ########################################################################################
# def reginGrow(image, mask):
#     im_shape = image.shape
#     height = im_shape[0]
#     width = im_shape[1]
#
#     markers = ndi.label(mask, output=np.uint32)[0]
#     unis = np.unique(markers)
#     # 获取种子点
#     seed_list = []
#
#     for uni in unis:
#         if uni == 0:
#             continue
#         pointsX, pointsY = np.where(markers == uni)
#         num_point = len(pointsX) // 4
#         for i in [0, num_point * 1, num_point * 2, num_point * 3]:
#             pointX, pointY = pointsX[i], pointsY[i]
#             seed_list.append(Point(pointX, pointY))
#
#     # 标记，判断种子是否已经生长
#     img_mark = np.zeros([height, width])
#
#     T = 7.5  # 阈值
#     class_k = 1  # 类别
#     # 生长一个类
#     while (len(seed_list) > 0):
#         seed_tmp = seed_list[0]
#         # 将以生长的点从一个类的种子点列表中删除
#         seed_list.pop(0)
#
#         img_mark[seed_tmp.x, seed_tmp.y] = class_k
#
#         # 遍历8邻域
#         for i in range(8):
#             tmpX = seed_tmp.x + connects[i].x
#             tmpY = seed_tmp.y + connects[i].y
#
#             if (tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= width):
#                 continue
#             dist = get_dist(image, seed_tmp, Point(tmpX, tmpY))
#             # 在种子集合中满足条件的点进行生长
#             if (dist < T and img_mark[tmpX, tmpY] == 0):
#                 img_mark[tmpX, tmpY] = class_k
#                 seed_list.append(Point(tmpX, tmpY))
#     img_mark = img_mark + mask
#     img_mark = remove_small_holes(img_mark.astype(np.uint8), 100)
#     return np.where(img_mark > 0, 1, 0)
#
#
# if __name__ == '__main__':
#
#     #import Image
#     im_proj, im_geotrans, im_width, im_height, im = gdalTools.read_img('data/image.tif')
#     im = im.transpose((1, 2, 0))
#     image = im.copy()
#     _, _, _, _, mask = gdalTools.read_img('data/seed.tif')
#     img_mark = reginGrow(image, mask)
import numpy as np
import cv2

# class Point(object):
#     def __init__(self,x,y):
#         self.x = x
#         self.y = y
#
#     def getX(self):
#         return self.x
#     def getY(self):
#         return self.y
#
# def getGrayDiff(img,currentPoint,tmpPoint):
#      return abs(int(img[currentPoint.x,currentPoint.y]) - int(img[tmpPoint.x,tmpPoint.y]))
#
# def selectConnects(p):
#      if p != 0:
#         connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1),Point(0, 1), Point(-1, 1), Point(-1, 0)]
#      else:
#         connects = [ Point(0, -1), Point(1, 0),Point(0, 1), Point(-1, 0)]
#      return connects
#
# def regionGrow(img,seeds,thresh,p = 1):
#      height, weight ,CH = img.shape
#      seedMark = np.zeros(img.shape)
#      seedList = []
#      for seed in seeds:
#       seedList.append(seed)
#      label = 1
#      connects = selectConnects(p)
#      while(len(seedList)>0):
#       currentPoint = seedList.pop(0)
#
#       seedMark[currentPoint.x,currentPoint.y] = label
#       for i in range(8):
#        tmpX = currentPoint.x + connects[i].x
#        tmpY = currentPoint.y + connects[i].y
#        if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
#         continue
#        grayDiff = getGrayDiff(img,currentPoint,Point(tmpX,tmpY))
#        if grayDiff < thresh and seedMark[tmpX,tmpY] == 0:
#         seedMark[tmpX,tmpY] = label
#         seedList.append(Point(tmpX,tmpY))
#      return seedMark
#
# if __name__ == '__main__':
#     img = cv2.imread(r'D:\Work\Datasets\test.png')
#     seeds = [Point(10,10),Point(82,150),Point(20,300)]
#     binaryImg = regionGrow(img,seeds,10)
#     cv2.imshow(' ',binaryImg)
#     cv2.waitKey(0)
# 区域生长 programmed by changhao
from PIL import Image
import matplotlib.pyplot as plt  # plt 用于显示图片
# import numpy as np

im = Image.open(r'D:\Work\Datasets\test.png') # 读取图片
# im.show()

im_array = np.array(im)

# print(im_array)
[m, n] = im_array.shape

a = np.zeros((m, n))  # 建立等大小空矩阵
a[70, 70] = 1  # 设立种子点
k = 40  # 设立区域判断生长阈值

flag = 1  # 设立是否判断的小红旗
while flag == 1:
    flag = 0
    lim = (np.cumsum(im_array * a)[-1]) / (np.cumsum(a)[-1])
    for i in range(2, m):
        for j in range(2, n):
            if a[i, j] == 1:
                for x in range(-1, 2):
                    for y in range(-1, 2):
                        if a[i + x, j + y] == 0:
                            if (abs(im_array[i + x, j + y] - lim) <= k):
                                flag = 1
                                a[i + x, j + y] = 1

data = im_array * a  # 矩阵相乘获取生长图像的矩阵
new_im = Image.fromarray(data)  # data矩阵转化为二维图片

# if new_im.mode == 'F':
#    new_im = new_im.convert('RGB')
# new_im.save('new_001.png') #保存PIL图片

# 画图展示
plt.subplot(1, 2, 1)
plt.imshow(im, cmap='gray')
plt.axis('off')  # 不显示坐标轴
plt.show()

plt.subplot(1, 2, 2)
plt.imshow(new_im, cmap='gray')
plt.axis('off')  # 不显示坐标轴
plt.show()
