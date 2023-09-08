import numpy as np
import os
import nibabel as nib
from glob import glob
import torch
####load brain mask
# root_path=r'E:\Datasets_public\IXI_MRA45\\'
# mask_path=sorted(glob(os.path.join(root_path,'noskull_data\\'+'*mask.nii.gz')))
# label_path=sorted(glob(os.path.join(root_path,'label\\'+'*GT.nii.gz')))
# save_path=r'E:\Datasets_public\IXI_MRA45\noskull_label\\'
# for i in range(len(mask_path)):
#     mask_arr=nib.load(mask_path[i]).get_fdata()
#     name=label_path[i].split('label\\')[-1]
#     label_arr=nib.load(label_path[i]).get_fdata()
#     new_arr=mask_arr*label_arr
#     new_arr=np.array(new_arr,dtype='uint8')
#     img=nib.Nifti1Image(new_arr,nib.load(mask_path[i]).affine)
#     nib.save(img,save_path+name)
#     print('\r[ %d / %d]' % (i, len(mask_path)), end='')
import monai.transforms as transforms

# Load Nifti image
nifti_img = nib.load(r"C:\Users\hello\Desktop\SSL\flare23\data\Normal001-MRA_brain.nii.gz")
nifti_label = nib.load(r"C:\Users\hello\Desktop\SSL\flare23\data\Normal001-MRA_brain.nii.gz")

image = nifti_img.get_fdata()
label = nifti_label.get_fdata()

# Convert image to PyTorch tensor
image = torch.Tensor(image)
label = torch.Tensor(label)

# Create MONAI transforms
cropper = transforms.CropForegroundd(keys=['image', 'label'], source_key='image',meta_keys=["end_coord"],margin=0)

# Create data dictionary
data_dict = {
    "image": image,
    "label":label
}

# Apply transform to data
data_dict = cropper(data_dict)

# Get end coordinate
# end_coord = data_dict["end_coord"]

start_coord = data_dict['foreground_start_coord'][0]
end_coord=data_dict['foreground_end_coord'][0]
print("Start \n End Coordinate: ",start_coord, end_coord)
