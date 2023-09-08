import glob
import os
import re
import json
from collections import OrderedDict
#将YOUR DIR替换成你自己的目录
path_originalData = "/YOUR_DIR/datasets/Task66_liver/"

os.mkdir(path_originalData+"imagesTr/")
os.mkdir(path_originalData+"labelsTr/")
os.mkdir(path_originalData+"imagesTs/")
os.mkdir(path_originalData+"labelsTs/")

def list_sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    def tryint(s):
        try:
            return int(s)
        except:
            return s
    def alphanum_key(s):
        """ Turn a string into a list of string and number chunks.
            "z23a" -> ["z", 23, "a"]
        """
        return [ tryint(c) for c in re.split('([0-9]+)', s) ]
    l.sort(key=alphanum_key)
    return l

train_image = list_sort_nicely(glob.glob(path_originalData+"imagesTr/*"))
train_label = list_sort_nicely(glob.glob(path_originalData+"labelsTr/*"))
test_image = list_sort_nicely(glob.glob(path_originalData+"imagesTs/*"))
test_label = list_sort_nicely(glob.glob(path_originalData+"labelsTs/*"))

train_image = ["{}".format(patient_no.split('/')[-1]) for patient_no in train_image]
train_label = ["{}".format(patient_no.split('/')[-1]) for patient_no in train_label]
test_image = ["{}".format(patient_no.split('/')[-1]) for patient_no in test_image]
#输出一下目录的情况，看是否成功
print(len(train_image),len(train_label),len(test_image),len(test_label), train_image[0])

#####下面是创建json文件的内容
#可以根据你的数据集，修改里面的描述
json_dict = OrderedDict()
json_dict['name'] = "BraTS"
json_dict['description'] = " Segmentation"
json_dict['tensorImageSize'] = "3D"
json_dict['reference'] = "see challenge website"
json_dict['licence'] = "see challenge website"
json_dict['release'] = "0.0"
#这里填入模态信息，0表示只有一个模态，还可以加入“1”：“MRI”之类的描述，详情请参考官方源码给出的示例
json_dict['modality'] = {
    "1": "MRI"
}

#这里为label文件中的多个标签，比如这里有血管、胆管、结石、肿块四个标签，名字可以按需要命名
json_dict['labels'] = {
    "0": "Background",
    "1": "vessel ",#静脉血管
    "2": "bileduck",#胆管
    "3": "stone",#结石
    "4": "lump" #肿块
}

#下面部分不需要修改>>>>>>
json_dict['numTraining'] = len(train_image)
json_dict['numTest'] = len(test_image)

json_dict['training'] = []
for idx in range(len(train_image)):
    json_dict['training'].append({'image': "./imagesTr/%s" % train_image[idx], "label": "./labelsTr/%s" % train_label[idx]})

json_dict['test'] = ["./imagesTs/%s" % i for i in test_image]

with open(os.path.join(path_originalData, "dataset.json"), 'w') as f:
    json.dump(json_dict, f, indent=4, sort_keys=True)
#<<<<<<<
