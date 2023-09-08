import pandas as pd
import glob
import os
import numpy as np
import openpyxl
from tqdm import tqdm



if __name__ == '__main__':

    model="read"
    # model="read"
    csv1name=r'D:\SEED\test\result448.csv'
    csv2name=r'D:\SEED\test\提交示例.csv'
    savecsv=r'D:\SEED\test\final448.csv'

    filenamelist=[]
    classificationlist=[]
    numberlist=[]
    result0l=[]
    result1l=[]
    result2l=[]
    result3l=[]
    result4l=[]
    # print(labelnameList[41056])
    # print("adafa")
    if model=="write":
        # for i in range(0,len(labelnameList),1):
        for i in tqdm(range(len(datanameList))):
            num=labelnameList[i].split('seg.npy')[0].split('__')[1]
            data_arr = np.load(datanameList[i])
            label_arr = np.load(labelnameList[i])
            #判断占比
            label_arr_sum=label_arr.sum()
            percent=(label_arr_sum/32768)*100
            if percent==0 :
                classification="BackGround"

            elif percent!=0:
                classification="vessel"

            # pathname='D:\Work\Datasets\MRA_Challenge\\112x112x32\\train_val\\'
            ##TODO
            filename=str(labelnameList[i]).split('val_patch\\')[-1]

            # pathnamelist.append(pathname)
            filenamelist.append(filename)
            percentlist.append(percent)
            classificationlist.append(classification)
            numberlist.append(num)
            output_excel = { 'filename': [], 'percent': [],'classification': [],'number':[]}

            # output_excel['pathname'] = pathnamelist
            output_excel['filename'] = filenamelist
            output_excel['percent'] = percentlist
            output_excel['classification'] = classificationlist
            output_excel['number']=numberlist
            output = pd.DataFrame(output_excel)
            output.to_csv(csvname, index=False)
            # print('\r[ %d / %d]' % (i, len(labelnameList)), end='')

    elif model=="read":
        input_df1 = pd.read_csv(csv1name)
        # input_df1 = input_df1.sort_values(by='idx')

        input_df2 = pd.read_csv(csv2name)
        # input_df2 = input_df2.sort_values(by='id')
        print("\n sum of train patches is :", input_df2.shape[0])
        for i in range(input_df2.shape[0]):
            print(input_df2.iloc[i].at['id'])
            pngname=input_df2.iloc[i].at['id'].split('_')[0]# 'N4HDTG'
            roiid=input_df2.iloc[i].at['id'].split('Annotation')[-1]#Annotation 0
            for j in range (input_df1.shape[0]):
                pngname1=input_df1.iloc[j].at['id'].split('_____')[-1].split('_')[-1]# 'N4HDTG'
                name_id=input_df1.iloc[j].at['id'].split('_____')[0]#最前面的数字
                roiid1=input_df1.iloc[j].at['id'].split('_____')[-1][0:1]#中间数字
                if (pngname==pngname1)&(roiid==roiid1):#锁定同一个文件且同一个roi 区域
                    print(input_df1.iloc[i].at['id'])

                    print("取平均")
                    if name_id=='0':
                        result0=input_df1.iloc[j].at['classification']
                        result0l.append(result0)
                        result1l.append('')
                        result2l.append('')
                        result3l.append('')
                        result4l.append('')
                        filenamelist.append(input_df2.iloc[i].at['id'])
                        # break
                    elif name_id=='1':
                        result1=input_df1.iloc[j].at['classification']
                        result1l.append(result1)
                        result0l.append('')
                        result2l.append('')
                        result3l.append('')
                        result4l.append('')
                        filenamelist.append(input_df2.iloc[i].at['id'])

                        # break

                    elif name_id == '2':
                        result2 = input_df1.iloc[j].at['classification']
                        result2l.append(result2)
                        result1l.append('')
                        result0l.append('')
                        result3l.append('')
                        result4l.append('')
                        filenamelist.append(input_df2.iloc[i].at['id'])

                        # continue

                    elif name_id == '3':
                        result3 = input_df1.iloc[j].at['classification']
                        result3l.append(result3)
                        result1l.append('')
                        result2l.append('')
                        result0l.append('')
                        result4l.append('')
                        filenamelist.append(input_df2.iloc[i].at['id'])

                        # break

                    elif name_id == '4':
                        result4 = input_df1.iloc[j].at['classification']
                        result4l.append(result4)
                        result1l.append('')
                        result2l.append('')
                        result3l.append('')
                        result0l.append('')
                        filenamelist.append(input_df2.iloc[i].at['id'])

                        # break

        output_excel = { 'filename': [], 'result0': [], 'result1': [], 'result2': [], 'result3': [], 'result4': []}

            # output_excel['pathname'] = pathnamelist
        output_excel['filename'] = filenamelist
        output_excel['result0'] = result0l
        output_excel['result1'] = result1l
        output_excel['result2'] = result2l
        output_excel['result3'] = result3l
        output_excel['result4'] = result4l

        # output_excel['classification'] = classificationlist
        # output_excel['number']=numberlist
        output = pd.DataFrame(output_excel)
        output.to_csv(savecsv, index=False)


