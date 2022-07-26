import nibabel as nib
import numpy as np
import os
import glob
import SimpleITK as sitk
import SimpleITK as STK
datafilePath=r'D:\Work\Datasets\Challenge\test\data\\'
# labelfilePath=r'D:\Work\Datasets\Challenge\train\label\\'

# labelfilePath=r'D:\Work\ZoomNet\MedicalZooPytorch-master\datasets\vessel\generated1%\test_vol_16x16x16\\'
dataoutput=r'D:\Work\Datasets\Challenge\test\data\\'
# labeloutput=r'D:\Work\Datasets\Challenge\train\label\\'
datanameList = sorted(glob.glob(os.path.join(datafilePath, '*.mhd')))
# labelnameList = sorted(glob.glob(os.path.join(labelfilePath, '*.mhd')))

for i in range(len(datanameList)):
    # img=STK.ReadImage(datanameList[i])
    # img=sitk.ReadImage(datanameList[i])
    dataname=str(datanameList[i]).split('data\\')[-1].split('.mhd')[0]
    img = sitk.ReadImage(datanameList[i])
    sitk.WriteImage(img, os.path.join(dataoutput,dataname + '.nii.gz'))
    #
    # labelname=str(labelnameList[i]).split('label\\')[-1].split('.mhd')[0]
    # label = sitk.ReadImage(labelnameList[i])
    # sitk.WriteImage(label, os.path.join(labeloutput,labelname + '.nii.gz'))
    # new_image.set_data_dtype(np.my_dtype)
    print('\r[ %d / %d]' % (i, len(datanameList)), end='>>>')

print("transform is over!")#####