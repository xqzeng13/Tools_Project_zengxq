import cv2 as cv
import SimpleITK as sitk
import cv2
import itk
import os
import numpy as np
from scipy import ndimage
if __name__ == "__main__":
    sigma = [0.5, 1, 1.5, 2, 2.5]
    tau = 2
    path = r"D:\Work\Datasets\Normaldatas89\dataimage"
    result_path = r"D:\Work\Datasets\Normaldatas89\hessian_mul\\"
    path_list = os.listdir(path)
    for i in path_list:
        image_i_path = os.path.join(path,i)
        img = sitk.ReadImage(image_i_path)
        name=image_i_path.split('dataimage\\')[-1]
        # result = Hessian3D(img,sigma,tau)
        result=vessleSegment(image_i_path,name,result_path)
        # sitk.WriteImage(result,os.path.join(result_path,i))
        print(i + " is OK!")