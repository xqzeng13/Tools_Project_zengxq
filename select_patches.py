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
    csvname=r'D:\Work\Datasets\MRA_Challenge\112x112x32\trian_ord.csv'
    Exsit_training_path = r'D:\Work\Datasets\MRA_Challenge\112x112x32\train_vol_112x112x32\\'
#   Exsit_val_path = r'D:\Work\Datasets\samples\debug_loss\val\\'
#   Exsit_test_path = r'D:\Work\Datasets\samples\debug_loss\test\\'
    datanameList = sorted(glob.glob(os.path.join(Exsit_training_path, '*0.npy')))
    labelnameList = sorted(glob.glob(os.path.join(Exsit_training_path, '*seg.npy')))
    datanameList.sort()
    labelnameList.sort()
    pathnamelist=[]
    percentlist=[]
    filenamelist=[]
    classificationlist=[]
    numberlist=[]
    # print(labelnameList[41056])
    # print("adafa")
    if model=="write":
        for i in range(len(labelnameList)):
            num=datanameList[i].split('train_vol_112x112x32\\')[1].split('train')[0]
            data_arr = np.load(datanameList[i])
            label_arr = np.load(labelnameList[i])
            #判断占比
            label_arr_sum=label_arr.sum()
            percent=(label_arr_sum/401408)*100
            if percent==0 :
                classification="BackGround"

            elif percent!=0:
                classification="vessel"

            pathname='D:\Work\Datasets\MRA_Challenge\\112x112x32\\train_val\\'
            filename=str(labelnameList[i]).split('train_vol_112x112x32\\')[-1]

            pathnamelist.append(pathname)
            filenamelist.append(filename)
            percentlist.append(percent)
            classificationlist.append(classification)
            numberlist.append(num)
            output_excel = {'pathname': [], 'filename': [], 'percent': [],'classification': [],'number':[]}

            output_excel['pathname'] = pathnamelist
            output_excel['filename'] = filenamelist
            output_excel['percent'] = percentlist
            output_excel['classification'] = classificationlist
            output_excel['number']=numberlist
            output = pd.DataFrame(output_excel)
            output.to_csv(csvname, index=False)
            print('\r[ %d / %d]' % (i, len(labelnameList)), end='')

    elif model=="read":
        input_df = pd.read_csv(csvname)
        input_df = input_df.sort_values(by='classification')
        output_list = []

        dataframe_1=input_df.loc[input_df['classification'] == 'vessel', ['pathname', 'filename']]
        dataframe_1.to_csv(r'D:\Work\Datasets\samples\train_file_vessel_1.csv')


