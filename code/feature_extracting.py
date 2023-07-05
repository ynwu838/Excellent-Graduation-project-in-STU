import  numpy as np
import pandas as pd
import os
import sklearn as sk
import matplotlib.pyplot as plt
import json
import math
import os
from PIL import Image
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.feature_selection import RFE
from sklearn.feature_selection import VarianceThreshold
def fun(x):
    return x >> 3
def depth2xyz(depth_map, depth_cam_matrix, flatten=False, depth_scale=1000):
    fx, fy = depth_cam_matrix[0, 0], depth_cam_matrix[1, 1]
    cx, cy = depth_cam_matrix[0, 2], depth_cam_matrix[1, 2]
    h, w = np.mgrid[0:depth_map.shape[0], 0:depth_map.shape[1]]
    z = depth_map/518
    x = w
    y = h
    xyz = np.dstack((x, y, z))
    return xyz

def getdata(filename1,filename2):
    jsonname = filename1
    picurename = filename2
    picurename = picurename.replace('Colour', 'Depth')
    with open(jsonname) as f:
        keypoints = json.load(f)
    body25 = keypoints['people'][0]['pose_keypoints_2d']
    img = Image.open(picurename)
    img = np.array(img)
    img = fun(img)
    img = np.where(img > 8192, img - 8192, img)

    depth_map = img.T

    depth_cam_matrix = np.array([[518, 0, 325.5],
                                 [0, 519, 253.5],
                                 [0, 0, 1]])
    targrt = depth2xyz(depth_map, depth_cam_matrix)

    x = []
    y = []
    z = []
    for i in range(0, 74, 3):
        D = targrt[int(body25[i])-1][int(body25[i + 1])-1]
        x.append(body25[i])
        y.append(body25[i + 1])
        z.append(D[2])

    return x, y, z


def VectorCosine(x, y):
    a = 0
    b = 0
    c = 0

    for i in range(0, len(x)):
        a += x[i] * y[i]
        b += x[i] * x[i]
        c += y[i] * y[i]
        c += 0.00001
        b += 0.00001
    result = a / (math.sqrt(b) * math.sqrt(c))
    return result


def cosvalues(filename1,filename2):
    x, y, z = getdata(filename1,filename2)
    result = []
    bodys = [
        [15, 17], [15, 0], [16, 0], [16, 18], [0, 1], [2, 1], [8, 1], [5, 1],
        [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [8, 12], [9, 10], [12, 13],
        [10, 11], [13, 14], [11, 24], [14, 21], [11, 22], [14, 19], [22, 23], [19, 20]
    ]
    i = 0
    bodys1 = bodys
    lengths = []

    for body in bodys:
        for body1 in bodys1[i + 1:24:1]:
            vector1 = np.array([x[body[0]] - x[body[1]], y[body[0]] - y[body[1]], z[body[0]] - z[body[1]]])
            vector2 = np.array([x[body1[0]] - x[body1[1]], y[body1[0]] - y[body1[1]], z[body1[0]] - z[body1[1]]])
            array = np.array([np.linalg.norm(vector1), np.linalg.norm(vector2)])

            cosine = VectorCosine(vector1, vector2)
            result.append(cosine)
        i = i + 1

    return result
Root="data/feature-extracting-result"
Sports=os.listdir(Root)
print(Sports)

for Sport in Sports:
    sportpath=Root+'/'+Sport
    Actors=os.listdir(sportpath)
    for Actor in Actors:
        Actionspath=sportpath+'/'+Actor
        Actions=os.listdir(Actionspath)
        for Action in Actions:
            Finalpath = Actionspath + '/' + Action
            print(Finalpath)
            for i in np.linspace(1, 1, 1):
                i = int(i)
                jsonpath = Finalpath+ '/output/' + str(i) + '_keypoints.json'
                depthpath = Finalpath+ '/depth/' + str(i) + '.png'

                Matrix = np.array(cosvalues(jsonpath, depthpath))

            for i in np.linspace(2, 50, 49):
                i = int(i)
                jsonpath = Finalpath+ '/output/' + str(i) + '_keypoints.json'
                depthpath = Finalpath+ '/depth/' + str(i) + '.png'

                Vector = np.array(cosvalues(jsonpath, depthpath))
                Matrix = np.vstack((Matrix, Vector))
            print(Matrix.shape)
            Matrixname = Finalpath + '/' + "Matrix.csv"
            print(Matrixname)
            Matrix.tofile(Matrixname, sep=',', format='%1.5f')
Root="data/feature-extracting-result"
sports=os.listdir(Root)
print(sports)
Class=0
'''Bowling是,0，Drving是1.Fighting是2，FPS是3，Golf是4，Misc是5，Tennis是6'''
result=pd.read_csv("data/feature-extracting-result/Bowling/Actor_1/1/Matrix.csv",header=None)
result["class"]=1
data=pd.DataFrame(columns=result.columns)
print(data)
for sport in sports:
    Actors=os.listdir(Root+"/"+sport)
    for Actor in Actors:
        bianshus=os.listdir(Root+"/"+sport+"/"+Actor)
        for bianshu in bianshus:
            filename=Root + "/" + sport + "/" + Actor + "/" + bianshu+"/Matrix.csv"
            data1=pd.read_csv(filename,header = None)
            data1["class"]=Class
            data=data.append(data1)
    Class=Class+1
data.to_csv("future_extracting_result.csv")
data=pd.read_csv("future_extracting_result.csv")
target=data["class"]
traindata=data.drop(["class"],axis=1)
sel = VarianceThreshold(threshold=0.16)
###选取0.16的原因：若特征是伯努利随机变量，假设p=0.8，即二分类特征中某种分类占到80%以上的时候删除特征
#因为有80%都是相似数据，对于数据的整体的特征已经没有太多的区分意义
traindata = sel.fit_transform(traindata)
print(traindata.shape)
RFC_ = RFC(n_estimators =25,random_state=0)
traindata = RFE(RFC_,n_features_to_select=500).fit_transform(traindata,target)
selectdata=pd.DataFrame(traindata)
selectdata.to_csv("future_selection_result.csv",index=False)
