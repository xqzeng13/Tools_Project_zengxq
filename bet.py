from glob import glob
import time
import os
t1w_path = r'/data/mxq/code/patient_process/t1w'
t1w_save_path = r'/data/mxq/code/patient_process/t1w_bet'
t1w_slices = glob(os.path.join(t1w_path,'*'))
t2w_path = r'/data/mxq/patient/t2w'
t2w_save_path = r'/data/mxq/patient/t2w_bet'
t2w_slices = glob(os.path.join(t2w_path,'*'))
for i,slice in enumerate(t1w_slices):
    basename = os.path.basename(slice)
    input_file = slice
    output_file = os.path.join(t1w_save_path,basename)
    os.system('bet '+input_file+' '+output_file+' -R')
    # print('bet '+input_file+' '+output_file+' -R')
    with open(r't1w.txt','a+') as f:
        f.write(str(i)+'  ['+time.ctime()+']   '+basename+'.....done!\n')
# for i,slice in enumerate(t2w_slices):
#     basename = os.path.basename(slice)
#     input_file = slice
#     output_file = os.path.join(t2w_save_path,basename)
#     os.system('bet '+input_file+' '+output_file+' -R')
#     # print('bet '+input_file+' '+output_file+' -R')
#     with open(r't2w.txt','a+') as f:
#         f.write(str(i)+'  ['+time.ctime()+']   '+basename+'.....done!\n')