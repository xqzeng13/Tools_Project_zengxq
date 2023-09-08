import pandas as pd
import glob
import os
import numpy as np
import openpyxl
import torch

if __name__ == '__main__':
    csvname = r'G:\lung\nii\train_64nor.csv'  # train_hessian_debug.csv Work\Datasets\new_samples\train_only_vessel_new.csv
    # pathname = r'/data4/zengxq/lung/64x64x32/train_val_nor/'
    savepath = r'G:\lung\nii\train_32nor.csv'
    filenamelist=[]
    numberlist=[]
    # edgepath=r'/data4/zengxq/64Xsamples/64xsamples/edge/'
    # hessianpath=r'/data4/zengxq/64Xsamples/64xsamples/hessian/'

    input_df = pd.read_csv(csvname)
    print("\n sum of val patches is :", input_df.shape[0])
    for i in range(input_df.shape[0]):
        labelname =  input_df.iloc[i].at['filename']
        dataname =  str(input_df.iloc[i].at['filename']).split('seg')[0] + '0.npy'
        number=input_df.iloc[i].at['filename'].split('__')[-1].split('seg')[0]

        label1=input_df.iloc[i].at['filename'].split('__')[0]+'01'+'__'+input_df.iloc[i].at['filename'].split('__')[-1]
        filenamelist.append(label1)
        numberlist.append(number)

        label2=input_df.iloc[i].at['filename'].split('__')[0]+'02'+'__'+input_df.iloc[i].at['filename'].split('__')[-1]
        filenamelist.append(label2)
        numberlist.append(number)

        label3=input_df.iloc[i].at['filename'].split('__')[0]+'03'+'__'+input_df.iloc[i].at['filename'].split('__')[-1]
        filenamelist.append(label3)
        numberlist.append(number)

        label4=input_df.iloc[i].at['filename'].split('__')[0]+'04'+'__'+input_df.iloc[i].at['filename'].split('__')[-1]
        filenamelist.append(label4)
        numberlist.append(number)

        output_excel = { 'filename': [],'number':[]}

        output_excel['filename'] = filenamelist
        output_excel['number']=numberlist
        output = pd.DataFrame(output_excel)
        output.to_csv(csvname, index=False)
        print('\r[ %d / %d]' % (i, input_df.shape[0]), end='')
