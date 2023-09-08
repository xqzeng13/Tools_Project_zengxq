import pandas as pd
from sklearn.model_selection import KFold

# 读取原始CSV文件
data = pd.read_csv(r"/data4/zengxq/oct/pre_oct/train_aug.csv")

# 创建五折交叉验证的KFold对象
kfold = KFold(n_splits=2, shuffle=True, random_state=42)

# 定义训练集和测试集存储列表
train_sets = []
test_sets = []

# 遍历KFold对象的划分结果，划分数据集为训练集和测试集
for train_idx, test_idx in kfold.split(data):
    train_set = data.iloc[train_idx]
    test_set = data.iloc[test_idx]
    train_sets.append(train_set)
    test_sets.append(test_set)

# 保存训练集和测试集到CSV文件
for i in range(len(train_sets)):
    train_sets[i].to_csv(f"E:\hgtdata\data_new\\train_data/dwi_train_{i+1}.csv", index=False)
    test_sets[i].to_csv(f"E:\hgtdata\data_new\\train_data\dwi_val_{i+1}.csv", index=False)
