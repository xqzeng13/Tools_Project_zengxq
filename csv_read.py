import pandas as pd
import glob
import os
import numpy as np
import openpyxl
def count_person_result(input_file, output_file):
    """
    将每个病例的所有测试图像的四个等级的预测概率求平均
    :param input_file:
    :param output_file:
    :return:
    """
    input_df = pd.read_excel(input_file, sheet_name='Sheet1')
    # TODO 解决排序后结果没有写回的问题
    input_df = input_df.sort_values(by='dirs')

    output_list = []
    count = 0
    temp_row = [0.0, 0.0, 0.0, 0.0, 0, 0, 'test']
    for i in range(len(input_df['dirs'])):
        temp_row[0] = temp_row[0] + input_df['p0'][i]
        temp_row[1] = temp_row[1] + input_df['p1'][i]
        temp_row[2] = temp_row[2] + input_df['p2'][i]
        temp_row[3] = temp_row[3] + input_df['p3'][i]
        count = count + 1

        if i + 1 < len(input_df['dirs']) and input_df['dirs'][i] is not input_df['dirs'][i + 1]:
            for j in range(4):
                temp_row[j] = temp_row[j] / count
            temp_row[4] = temp_row[:4].index(max(temp_row[:4]))
            temp_row[5] = input_df['label_gt'][i]
            temp_row[6] = input_df['dirs'][i]
            output_list.append(temp_row)

            count = 0
            temp_row = [0.0, 0.0, 0.0, 0.0, 0, 0, 'test']

        if i + 1 == len(input_df['dirs']):
            # last line
            for j in range(4):
                temp_row[j] = temp_row[j] / count
            temp_row[4] = temp_row[:4].index(max(temp_row[:4]))
            temp_row[5] = input_df['label_gt'][i]
            temp_row[6] = input_df['dirs'][i]
            output_list.append(temp_row)

    df = pd.DataFrame(output_list, columns=['p0', 'p1', 'p2', 'p3', 'label-pre', 'label_gt', 'dirs'])
    df.to_excel(output_file)


if __name__ == '__main__':

    model="write"
    # model="read"
    csvname=r'E:\Improtant file\patient\mra\mra.csv'
    Exsit_training_path = r'E:\Improtant file\patient\mra\\'
#   Exsit_val_path = r'D:\Work\Datasets\samples\debug_loss\val\\'
#   Exsit_test_path = r'D:\Work\Datasets\samples\debug_loss\test\\'
    datanameList = sorted(glob.glob(os.path.join(Exsit_training_path, '*.nii')))
    # labelnameList = sorted(glob.glob(os.path.join(Exsit_training_path, '*seg.npy')))
    datanameList.sort()
    # labelnameList.sort()
    # pathnamelist=[]
    namelist=[]
    labellist=[]
    # classificationlist=[]
    # numberlist=[]
    # print(labelnameList[41056])
    # print("adafa")
    if model=="write":
        for i in range(len(datanameList)):
            name=datanameList[i].split('.nii')[0].split('mra\\')[-1]




            # pathnamelist.append(pathname)
            namelist.append(name)

            output_excel = { 'name': []}

            # output_excel['pathname'] = pathnamelist
            output_excel['name'] = namelist

            output = pd.DataFrame(output_excel)
            output.to_csv(csvname, index=False)
            print('\r[ %d / %d]' % (i, len(datanameList)), end='')
    if model =='read':

        csvname1= r'C:\Users\hello\Desktop\alllabel.csv'#train_hessian_debug.csv Work\Datasets\new_samples\train_only_vessel_new.csv
        input_df1 = pd.read_csv(csvname1)

        csvname2 = r'E:\Improtant file\patient\mra\mra.csv'  # train_hessian_debug.csv Work\Datasets\new_samples\train_only_vessel_new.csv
        input_df2= pd.read_csv(csvname2)
        print("\n sum of train patches is :", input_df2.shape[0])
        for i in range(input_df2.shape[0]):
            # labelname = input_df1.iloc[i].at['label']
            dataname = input_df2.iloc[i].at['name']
            if dataname in (input_df1.iloc[i].at['subject']):
                label=


