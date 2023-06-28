import numpy
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn import svm
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier as RFC
from xgboost import XGBClassifier
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']='SimHei' #设置字体
plt.rcParams["font.size"]=10 #设置字号
plt.rcParams["axes.unicode_minus"]=False #正常显示负号
data=pd.read_csv("future_extracting_result.csv")
target=data["class"]
traindata=pd.read_csv("future_selection_result.csv")
'''sel = VarianceThreshold(threshold=0.2)
traindata = sel.fit_transform(traindata)'''
X_train,X_test, y_train, y_test=train_test_split(traindata,target,test_size=0.5,random_state=2)
model1=SVC(probability=True,kernel="linear")
C=np.linspace(0.0001,1,20)
A=[]
B=[]
A1=[]
A2=[]
for i in C:
    model=SVC(probability=True,kernel="poly")
    model.fit(X_train, y_train)
    y_pred=model.predict(X_test)
    A.append(i)
    B.append(accuracy_score(y_test,y_pred))
for i in C:
    model=SVC(probability=True,kernel="rbf")
    model.fit(X_train, y_train)
    y_pred=model.predict(X_test)
    A2.append(accuracy_score(y_test,y_pred))
print(A2)
print(B)
plt.ylabel("分类准确率")#横坐标名字
plt.xlabel("C")#纵坐标名字
plt.plot(A, B,label='poly')
plt.plot(A,A2,label='rbf')
plt.legend(loc='lower right')
plt.ylim(0.9,1)
plt.show()