import nibabel as nib
import numpy as np
import os
import glob
import SimpleITK as sitk

datafilePath=r'D:\Work\Datasets\Challenge\test\pred_result\\'
dataoutput=r'D:\Work\Datasets\Challenge\test\pred_mhd\\'
datanameList = sorted(glob.glob(os.path.join(datafilePath, '*.nii.gz')))


for i in range(len(datanameList)):
    # img=STK.ReadImage(datanameList[i])
    # img=sitk.ReadImage(datanameList[i])
    dataname='pred_test'+str(datanameList[i]).split('predtest')[-1].split('.nii')[0]
    nii_data = sitk.ReadImage(datanameList[i])
    # print(nii_data)
    # print(nii_data.GetSpacing())

    Direction=nii_data.GetDirection()
    Spacing = nii_data.GetSpacing()
    Origin = nii_data.GetOrigin()
    # Origin = referencect.ImagePositionPatient
    nii_arr = sitk.GetArrayFromImage(nii_data)  # z y x
    raw_arr=nii_arr
    raw_arr.astype(np.double)  # 可以转换成自己需要的类型
    raw_arr.tofile(dataoutput+dataname+".raw")
    mhd_image = sitk.GetImageFromArray(nii_arr)  # z y x
    mhd_image.SetSpacing(Spacing)
    mhd_image.SetOrigin(Origin)
    mhd_image.SetDirection(Direction)

    sitk.WriteImage(mhd_image, dataoutput+dataname+".mhd")
    print('\r[ %d / %d]' % (i, len(datanameList)), end='>>>')

print("transform is over!")#####