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
from matplotlib import font_manager
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']='SimHei' #设置字体
plt.rcParams["font.size"]=10 #设置字号
plt.rcParams["axes.unicode_minus"]=False #正常显示负号
my_font = font_manager.FontProperties(fname="C:\Windows\Fonts\MSYHL.TTC")
data=pd.read_csv("future_extracting_result.csv")
target=data["class"]
traindata=pd.read_csv("future_selection_result.csv")
X_train,X_test, y_train, y_test=train_test_split(traindata,target,test_size=0.5,random_state=2)
estimators=np.linspace(1,200,50).astype(int)
A=[]
B=[]
A1=[]
A2=[]
A3=[]
for i in estimators:
    model=XGBClassifier(n_estimators=i,booster="gbtree",random_state=2)
    model.fit(X_train, y_train)
    y_pred=model.predict(X_test)
    A.append(i)
    B.append(accuracy_score(y_test,y_pred))
for i in estimators:
    model=LGBMClassifier(n_estimators=i,random_state=2)
    model.fit(X_train, y_train)
    y_pred=model.predict(X_test)
    A2.append(accuracy_score(y_test,y_pred))
print(estimators)
print(A2)
print(B)
plt.ylabel("分类准确率")#横坐标名字
plt.xlabel("基学习器个数")#纵坐标名字
plt.plot(A, B, c='green',  label="Xgboost",markersize = 2,alpha=0.7)
plt.plot(A,A2,c='blue', linestyle='--' ,label="lightgbm",markersize = 2,alpha=0.5)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(prop=my_font, loc='lower right')
plt.show()
