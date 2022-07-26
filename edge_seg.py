import nibabel as nib
import numpy as np
from skimage.feature import canny
import SimpleITK as sitk
def canny_3d(image_arr):
    cannyop=sitk.CannyEdgeDetectionImageFilter()
    cannyop.SetUpperThreshold(0.8)#image_arr.max()
    cannyop.SetLowerThreshold(0.8)
    cannyop.SetVariance(1)
    cannyop.SetMaximumError(0.1)
    canny_sitk = cannyop.Execute(image_arr)
    canny_sitk = sitk.Cast(canny_sitk, sitk.sitkInt16)
    return canny_sitk

def canny_edges_3d(grayImage):
    MAX_CANNY_THRESHOLD = grayImage.max()

    MIN_CANNY_THRESHOLD = 0.7*MAX_CANNY_THRESHOLD

    dim = np.shape(grayImage)

    edges_x = np.zeros(grayImage.shape, dtype=bool)
    edges_y = np.zeros(grayImage.shape, dtype=bool)
    edges_z = np.zeros(grayImage.shape, dtype=bool)
    edges = np.zeros(grayImage.shape, dtype=bool)

    # print(np.shape(edges))

    for i in range(dim[0]):
        edges_x[i, :, :] = canny(grayImage[i, :, :], low_threshold=MIN_CANNY_THRESHOLD,
                                 high_threshold=MAX_CANNY_THRESHOLD, sigma=0)

    for j in range(dim[1]):
        edges_y[:, j, :] = canny(grayImage[:, j, :], low_threshold=MIN_CANNY_THRESHOLD,
                                 high_threshold=MAX_CANNY_THRESHOLD, sigma=0)

    for k in range(dim[2]):
        edges_z[:, :, k] = canny(grayImage[:, :, k], low_threshold=MIN_CANNY_THRESHOLD,
                                 high_threshold=MAX_CANNY_THRESHOLD, sigma=0)

    edges = canny(grayImage, low_threshold=MIN_CANNY_THRESHOLD, high_threshold=MAX_CANNY_THRESHOLD)
    for i in range(dim[0]):
        for j in range(dim[1]):
            for k in range(dim[2]):
                edges[i, j, k] = (edges_x[i, j, k] and edges_y[i, j, k]) or (edges_x[i, j, k] and edges_z[i, j, k]) or (
                            edges_y[i, j, k] and edges_z[i, j, k])
                # edges[i,j,k] = (edges_x[i,j,k]) or (edges_y[i,j,k]) or (edges_z[i,j,k])

    return edges
def sobel_3d(image_arr):
    data=sitk.Cast(image_arr,sitk.sitkFloat32)
    # 提取梯度图
    gra_filter = sitk.SobelEdgeDetectionImageFilter()
    sobel_data = gra_filter.Execute(data)
    sobel_data = sitk.Cast(sobel_data, sitk.sitkInt16)
    return sobel_data
def load_nii(filename):
    Image = nib.load(filename)
    img_arr = Image.get_fdata()
    name = filename.split('/')[-1]
    return img_arr.astype(np.float32), img_arr.shape, name, Image.affine, Image.header


def save_nii(data, save_name, affine, header):

    new_img = nib.Nifti1Image(data.astype(np.int16), affine, header)

    nib.save(new_img, save_name)

# img = sitk.ReadImage(r'D:\other\x1patch.nii.gz')
img_arr=np.load(r'D:\Work\Datasets\64xsamples\train\train_1%part(29, 153, 53)__280.npy')
img=sitk.GetImageFromArray(img_arr)
img_ori=img
sitk.WriteImage(img_ori, 'D:\\other\\627_ori.nii.gz')
# img, shape, name, affine, header = load_nii(r'D:\Work\Datasets\97\ori_data\data\Normal097-MRA_brain.nii.gz')

# edges = canny_edges_3d(img)
canny_sitk=canny_3d(img)
# sobel_img=sobel_3d(img)
#归一化
# sobel_img_nor=sobel_img/sobel_img.max()
sobel_arr=sitk.GetArrayFromImage(canny_sitk)
#
# sobel_arr=sitk.GetArrayFromImage(sobel_img)
sobel_min=sobel_arr.min()
sobel_max=sobel_arr.max()
sobel_nor=(sobel_arr-sobel_min)/(sobel_max-sobel_min)
# sobel_nor[sobel_nor<0.5]=0
# sobel_nor[sobel_nor>0.5]=1

sobel_img=sitk.GetImageFromArray(sobel_nor)
sitk.WriteImage(sobel_img, 'D:\\other\\627canny_edge.nii.gz')
# save_nii(edges, "D:\other\edge.nii.gz",affine, header)
###todo
# import os
# import numpy as np
# import SimpleITK as sitk
#
# data_dir = r'D:\Work\Datasets\97\ori_data\data\Normal097-MRA_brain.nii.gz'
# data_nii = sitk.ReadImage(data_dir)
# origin = data_nii.GetOrigin()
# spacing = data_nii.GetSpacing()
# direction = data_nii.GetDirection()
#
# # change data type before edge detection
# data_float_nii = sitk.Cast(data_nii, sitk.sitkFloat32)
#
# canny_op = sitk.CannyEdgeDetectionImageFilter()
# canny_op.SetLowerThreshold(100)
# canny_op.SetUpperThreshold(200)
# canny_op.SetVariance(1)
# canny_op.SetMaximumError(0.5)
# canny_sitk = canny_op.Execute(data_float_nii)
# canny_sitk = sitk.Cast(canny_sitk, sitk.sitkInt16)
#
# canny_sitk.SetOrigin(origin)
# canny_sitk.SetSpacing(spacing)
# canny_sitk.SetDirection(direction)
# sitk.WriteImage(canny_sitk,'D:\other\edge.nii.gz')
