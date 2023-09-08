# '''
# position enconder  of patch
# patch的位置编码
# '''
# import numpy as np
# import nibabel as nib
# import os
# import torch
# import pandas as pd
# def position_encoder(pos_site,patchsize):
#     # fre=np.array([[pos / np.power(10000, 2 * (j // 2) / dim[0]) for j in range(dim[0])] for pos in range(128)])##power 幂指数；// 取整舍弃余数
#     # num=np.array([[[c for c in range(128, 128 + 64, 1)] for k in range(128, 128 + 64, 1)] for d in range(32, 32 + 32, 1)])
#     # num = np.array([[[(c, k, d) for c in range(128, 128 + 64, 1)] for k in range(256, 256 + 64, 1)]for d in range (32 ,3+32,1)])
#     # out.requires_grad = False
#     flist = []
#     position=[]
#     pos_site=pos_site
#     patchsize=patchsize
#     for c in range(pos_site[0][0], pos_site[0][0] + patchsize[0], 1):
#         for k in range(pos_site[0][1], pos_site[0][1] + patchsize[1], 1):
#             for d in range(pos_site[0][2], pos_site[0][2] +patchsize[2], 1):
#                 f = (c, k, d)
#                 flist.append(f)
#
#     flist_arr=np.array(flist)/(10000)
#     for i in range(len(flist_arr[:,0])):
#         pos=flist_arr[i]
#         if (i%2)==0:#####偶数
#             # pos=(np.sin(pos[0])+np.sin(pos[1])+np.sin(pos[2]))/3
#             pos=(np.sin(pos[0])+2*np.cos(pos[1])+3*np.sin(0.5*pos[2]))/3
#
#             position.append(pos)
#         else :#奇数
#             # pos = (np.cos(pos[0]) + np.cos(pos[1]) + np.cos(pos[2])) / 3
#             pos=(np.cos(pos[0])+3*np.sin(pos[1])+3*np.cos(0.5*pos[2]))/3
#
#             position.append(pos)
#     pos_arr=np.array(position)
#     a = np.unique(pos_arr)
#
#
#     if len(pos_arr)==len(a):
#         print(pos_site,'ok!!')
#     else:
#
#         print(pos_site,'error!!')
#
#
#     pos_arr=pos_arr.view().reshape(patchsize)
#     return pos_arr
#
# # np.unique()
#
# def get_pos_number():
#     csvname = r'G:\lung\nii\val_32nor.csv'  # train_hessian_debug.csv Work\Datasets\new_samples\train_only_vessel_new.csv
#     # pathname = r'/data4/zengxq/lung/32x32x32/train_val/'
#     # edgepath=r'/data4/zengxq/64Xsamples/64xsamples/edge/'
#     # hessianpath=r'/data4/zengxq/64Xsamples/64xsamples/hessian/'
#
#     input_df = pd.read_csv(csvname)
#     print("\n sum of val patches is :", input_df.shape[0])
#     for i in range(input_df.shape[0]):
#         labelname = input_df.iloc[i].at['filename']
#         num_str=labelname.split('(')[1].split(')')[0]
#         pos_num=(int(num_str.split(',')[0]),int(num_str.split(',')[1]),int(num_str.split(',')[2]))
#         return pos_num
#         # dataname =  str(input_df.iloc[i].at['filename']).split('seg')[0] + '0.npy'
#
#
# # pos_num=get_pos_number()
# # pos_num={(0,0,32),(0,0,64),(0,0,96),(0,0,128),(0,0,128),(0,0,128),(0,0,128),(0,0,128),(0,0,128),(0,0,128),(0,0,128),(0,0,128),(0,0,128),(0,0,128),(0,0,128),}
# pos_numlist=[]
# for s in range(0,128,32):
#     for h in range(0,448,64):
#         for w in range(0,448,64):
#             w1=w
#             h1=h
#             s1=s
#             pos1=(h1,w1,s1)
#             pos_numlist.append(pos1)
# patchsize=(64,64,32)
#
# for pos_num in zip (pos_numlist):
#     possition_arr=position_encoder(pos_num,patchsize)
#     # print(possition_arr,'\n position_shape is : \n',possition_arr.shape)

#todo
'''
position enconder  of patch
patch的位置编码
'''
import numpy as np
import nibabel as nib
import os
import torch
import pandas as pd
def position_encoder(pos_site,patchsize):
    # fre=np.array([[pos / np.power(10000, 2 * (j // 2) / dim[0]) for j in range(dim[0])] for pos in range(128)])##power 幂指数；// 取整舍弃余数
    # num=np.array([[[c for c in range(128, 128 + 64, 1)] for k in range(128, 128 + 64, 1)] for d in range(32, 32 + 32, 1)])
    # num = np.array([[[(c, k, d) for c in range(128, 128 + 64, 1)] for k in range(256, 256 + 64, 1)]for d in range (32 ,3+32,1)])
    # out.requires_grad = False
    flist = []
    position=[]
    pos_site=pos_site
    patchsize=patchsize
    for c in range(pos_site[0][0], pos_site[0][0] + patchsize[0], 1):
        for k in range(pos_site[0][1], pos_site[0][1] + patchsize[1], 1):
            for d in range(pos_site[0][2], pos_site[0][2] +patchsize[2], 1):
                f = (c, k, d)
                # f=c+k+d
                flist.append(f)


    # flist_arr=np.array(flist)/(10000*np.pi)
    flist_arr=np.array(flist)

    for i in range(len(flist_arr[:,0])):
    # for i in range(len(flist_arr)):

        pos=flist_arr[i,:]

        if (i%2)==0:#####偶数
            pos=(np.sin(pos[0])+np.cos(pos[1])+np.sin(0.5*pos[2]))/3
            # pos=(np.sin(pos[0]/(np.power(10000, 2 * (i // 2))))+np.cos(pos[1]/(np.power(10000, 2 * (i // 2))))+np.sin(pos[2]/(np.power(10000, 2 * (i // 2)))))

            position.append(pos)
        else :#奇数
            pos = (np.cos(pos[0]) + np.sin(pos[1]) + np.cos(0.5*pos[2])) / 3
            # pos=(np.cos(pos))
            # pos = (np.cos(pos[0] / (np.power(10000, 2 * (i // 2)))) + np.sin(
            #     pos[1] / (np.power(10000, 2 * (i // 2)))) + np.cos(pos[2] / (np.power(10000, 2 * (i // 2)))))

            position.append(pos)
    pos_arr=np.array(position)
    a = np.unique(pos_arr)


    if len(pos_arr)==len(a):
        print(pos_site,'ok!!')
    else:

        print(pos_site,'error!!')


    pos_arr=pos_arr.view().reshape(patchsize)
    return pos_arr

# np.unique()

def get_pos_number():
    csvname = r'G:\lung\nii\val_32nor.csv'  # train_hessian_debug.csv Work\Datasets\new_samples\train_only_vessel_new.csv
    # pathname = r'/data4/zengxq/lung/32x32x32/train_val/'
    # edgepath=r'/data4/zengxq/64Xsamples/64xsamples/edge/'
    # hessianpath=r'/data4/zengxq/64Xsamples/64xsamples/hessian/'

    input_df = pd.read_csv(csvname)
    print("\n sum of val patches is :", input_df.shape[0])
    for i in range(input_df.shape[0]):
        labelname = input_df.iloc[i].at['filename']
        num_str=labelname.split('(')[1].split(')')[0]
        pos_num=(int(num_str.split(',')[0]),int(num_str.split(',')[1]),int(num_str.split(',')[2]))
        return pos_num
        # dataname =  str(input_df.iloc[i].at['filename']).split('seg')[0] + '0.npy'


# pos_num=get_pos_number()
# pos_num={(0,0,32),(0,0,64),(0,0,96),(0,0,128),(0,0,128),(0,0,128),(0,0,128),(0,0,128),(0,0,128),(0,0,128),(0,0,128),(0,0,128),(0,0,128),(0,0,128),(0,0,128),}
pos_numlist=[]
for s in range(0,128,32):
    for h in range(0,448,64):
        for w in range(0,448,64):
            w1=w
            h1=h
            s1=s
            pos1=(h1,w1,s1)
            pos_numlist.append(pos1)
patchsize=(64,64,32)

for pos_num in zip (pos_numlist):
    possition_arr=position_encoder(pos_num,patchsize)
    # print(possition_arr,'\n position_shape is : \n',possition_arr.shape)