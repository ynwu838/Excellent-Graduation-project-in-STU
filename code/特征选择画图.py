import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn import svm
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier as RFC
from xgboost import XGBClassifier
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']='SimHei' #设置字体
plt.rcParams["font.size"]=10 #设置字号
plt.rcParams["axes.unicode_minus"]=False #正常显示负号
data=pd.read_csv("future_extracting_result.csv")
target=data["class"]
traindata=pd.read_csv("future_selection_result.csv")
x=pd.DataFrame(traindata.var())
x.hist(bins=100)
plt.title("")
plt.xlabel("特征的方差")#横坐标名字
plt.ylabel("数量")#纵坐标名字
plt.show()
print()