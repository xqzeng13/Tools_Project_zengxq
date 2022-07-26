import nibabel as nib
import numpy as np
import os
import glob
datafilePath=r'D:\Work\Datasets\samples\16x16x16_7%\test_10%patch\test_vol_all_16x16x16\\'
# labelfilePath=r'D:\Work\ZoomNet\MedicalZooPytorch-master\datasets\vessel\generated1%\test_vol_16x16x16\\'
dataoutput=r'D:\Work\Datasets\samples\16x16x16_7%\test_10%patch\test_vol_all_nii\data\\'
labeloutput=r'D:\Work\Datasets\samples\16x16x16_7%\test_10%patch\test_vol_all_nii\label\\'
datanameList = sorted(glob.glob(os.path.join(datafilePath, '*0.nii.gz.npy')))
labelnameList = sorted(glob.glob(os.path.join(datafilePath, '*seg.nii.gz.npy')))

for i in range(len(datanameList)):
    data_arr=np.load(datanameList[i])
    new_image = nib.Nifti1Image(data_arr, np.eye(4))
    nib.save(new_image, 'nifti.nii.gz')
# new_image.set_data_dtype(np.my_dtype)
    dataname=str(datanameList[i]).split('_0')[0]
    print(dataname)
    print(datanameList[i],i)
    # print(str(datanameList[i]).split('_0')[0])
    nib.save(new_image, os.path.join(dataoutput,dataname+'.nii.gz'))


for j in range(len(labelnameList)):
    label_arr=np.load(labelnameList[j])
    new_label = nib.Nifti1Image(label_arr, np.eye(4))
    nib.save(new_label, 'nifti.nii.gz')
# new_image.set_data_dtype(np.my_dtype)
    labelname=str(labelnameList[j]).split('_0')[0]
    print(labelname)
    print(labelnameList[j],j)#############split('_0')[-1]表示‘-0’前面的保留，[1]后面的保留
    nib.save(new_label, os.path.join(labeloutput,labelname+'seg.nii.gz'))
print("transform is over!")#####