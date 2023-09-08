from sklearn.model_selection import StratifiedKFold
import numpy
import pandas as pd

def K_StratifiedKFold_spilt(K, X,Y):
    '''
    :param K: 要把数据集分成的份数。如十次十折取K=10
    :param fold: 要取第几折的数据。如要取第5折则 flod=5
    :param data: 需要分块的数据
    :param label: 对应的需要分块标签
    :return: 对应折的训练集、测试集和对应的标签
    '''
    split_list = []
    kf = StratifiedKFold(n_splits=K)

    for train, test in kf.split(X,Y):
        split_list.append(train.tolist())
        split_list.append(test.tolist())
    # train,test=split_list[2 * fold],split_list[2 * fold + 1]
    # return  data[train], data[test], label[train], label[test]  #已经分好块的数据集
    return split_list


seed = 7  # 随机种子
numpy.random.seed(seed)  # 生成固定的随机数
num_k = 1  # 多少折

csvfile=r'E:\hgtdata\data_new\train_data\train_val.csv'
# 整个数据集(自己定义)
input_df1 = pd.read_csv(csvfile)

X =input_df1['id']
Y =input_df1['label']

kfold = StratifiedKFold(n_splits=num_k, shuffle=True, random_state=seed)  # 分层K折，保证类别比例一致
split_list = K_StratifiedKFold_spilt(k, X, Y)

cvscores = []
for fold in range(5):
    print(str(fold) + '折')
    train, test = split_list[2 * fold], split_list[2 * fold + 1]
    print(train)
    x_train = X.iloc[train]
    print(x_train)
    x_test = X.iloc[test]
    print(x_test)
    y_train = Y.iloc[train]
    y_test = Y.iloc[test]
#     model =
# model.compile()  # 自定义
#
# # 模型训练
# model.fit(X[train], Y[train], epochs=150, batch_size=10, verbose=0)
#
# # 模型测试
# scores = model.evaluate(X[test], Y[test], verbose=0)
#
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))  # 打印出验证集准确率
#
# cvscores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))  # 输出k-fold的模型平均和标准差结果
