# import numpy as np
import numpy as np
import SimpleITK as sitk
from glob import glob

# def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):
#
#     resampler = sitk.ResampleImageFilter()
#     originSize = itkimage.GetSize()  # 原来的体素块尺寸
#     originSpacing = itkimage.GetSpacing()
#     newSize = np.array(newSize,float)
#     factor = originSize / newSize
#     newSpacing = originSpacing * factor
#     newSize = newSize.astype(np.int) #spacing肯定不能是整数
#     resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
#     resampler.SetSize(newSize.tolist())
#     resampler.SetOutputSpacing(newSpacing.tolist())
#     resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
#     resampler.SetInterpolator(resamplemethod)
#     itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
#     return itkimgResampled
#
#
# image_path = r'E:\MRA_public\resize_data\\'
# save_path   = r'E:\MRA_public\resize_data\resized\\'
# image_file = glob(image_path + 'data\\'+'*.nii.gz')
# label_file = glob(image_path +'label\\'+'*.nii.gz')
# for i in range(len(image_file)):
#     # itkimage = sitk.ReadImage(image_file[i])
#     # itkimgResampled = resize_image_itk(itkimage, (448,448,512),resamplemethod= sitk.sitkNearestNeighbor) ## Near/Linear#这里要注意：mask用最近邻插值，CT图像用线性插值
#     # sitk.WriteImage(itkimgResampled, save_path +'data\\'+"512N-"+ image_file[i][len(image_path)+5:])
#
#     itklabel = sitk.ReadImage(label_file[i])
#     itklabelResampled = resize_image_itk(itklabel, (448,448,640),resamplemethod= sitk.sitkNearestNeighbor) ## Near/Linear#这里要注意：mask用最近邻插值，CT图像用线性插值
#     sitk.WriteImage(itklabelResampled, save_path +'label\\'+"640-"+ image_file[i][len(image_path)+5:])



def respacing_image_itk(itkimage, newSpacing, resamplemethod=sitk.sitkNearestNeighbor):

    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  # 原来的体素块尺寸
    originSpacing = itkimage.GetSpacing()
    # newSpacing=newSpacing
    newSpacing = np.array(newSpacing, float)
    factor=newSpacing/originSpacing
    newSize=originSize/factor

    # newSize = np.array(newSize,float)
    # factor = originSize / newSize
    # newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int) #spacing肯定不能是整数
    resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
    return itkimgResampled


image_path = r'E:\MRA_public\resize_data\\'
save_path   = r'E:\MRA_public\resize_data\resized\\'
image_file = glob(image_path + 'data\\'+'*.nii.gz')
label_file = glob(image_path +'label\\'+'*.nii.gz')
for i in range(len(image_file)):
    itkimage = sitk.ReadImage(image_file[i])
    itkimgResampled = respacing_image_itk(itkimage, (1,1,1),resamplemethod= sitk.sitkLinear) ## Near/Linear#这里要注意：mask用最近邻插值，CT图像用线性插值
    sitk.WriteImage(itkimgResampled, save_path +'data\\'+"1111-"+ image_file[i][len(image_path)+5:])

    itklabel = sitk.ReadImage(label_file[i])
    itklabelResampled = respacing_image_itk(itklabel, (1,1,1),resamplemethod=sitk.sitkNearestNeighbor) ## Near/Linear#这里要注意：mask用最近邻插值，CT图像用线性插值
    sitk.WriteImage(itklabelResampled, save_path +'label\\'+"11111-"+ image_file[i][len(image_path)+5:])