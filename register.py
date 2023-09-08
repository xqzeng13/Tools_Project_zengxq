##DWI-->FLAIR, ADC-.mat --> FLAIR

from glob import glob
import time
import os

flair_path = r'/home/sparrow/Medical_Image/DWI_FLAIR_Mismatch/Data_open/original/FLAIR'
dwi_path = r'/home/sparrow/Medical_Image/DWI_FLAIR_Mismatch/Data_open/original/DWI'
mat_path = r'/home/sparrow/Medical_Image/DWI_FLAIR_Mismatch/Data_open/2flair/mat'

flair_slices = glob(os.path.join(flair_path, '*'))
dwi_slices = glob(os.path.join(dwi_path, '*'))
print(len(flair_slices))

save_path = r'/home/sparrow/Medical_Image/DWI_FLAIR_Mismatch/Data_open/2flair/dwi_re'

for i, slice in enumerate(dwi_slices):
    basename = os.path.basename(slice)
    matname = basename + '.mat'
    invol = slice
    refvol = os.path.join(flair_path, basename)
    outvol = os.path.join(save_path, basename)
    matvol = os.path.join(mat_path, matname)
    os.system('flirt -in ' + invol + ' -ref ' + refvol + ' -out ' + outvol + ' -omat ' + matvol + ' \
    -bins 256 -cost corratio -searchrx 0 0 -searchry 0 0 -searchrz 0 0 -dof 6 -schedule \
    /usr/local/fsl/etc/flirtsch/sch3Dtrans_3dof  -interp trilinear')

    # with open(r't1w.txt','a+') as f:
    # f.write(str(i)+'  ['+time.ctime()+']   '+basename+'.....done!\n')
# for i,slice in enumerate(t2w_slices):
#     basename = os.path.basename(slice)
#     input_file = slice
#     output_file = os.path.join(t2w_save_path,basename)
#     os.system('bet '+input_file+' '+output_file+' -R')
#     # print('bet '+input_file+' '+output_file+' -R')
#     with open(r't2w.txt','a+') as f:
#         f.write(str(i)+'  ['+time.ctime()+']   '+basename+'.....done!\n')