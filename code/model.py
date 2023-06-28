import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']='SimHei' #设置字体
plt.rcParams["font.size"]=10 #设置字号
plt.rcParams["axes.unicode_minus"]=False #正常显示负号
data=pd.read_csv("future_extracting_result.csv")
target=data["class"]
traindata=pd.read_csv("future_selection_result.csv")
print(traindata.shape)
X_train,X_test, y_train, y_test=train_test_split(traindata,target,test_size=0.5,random_state=2)
#--------------------------------（1）logistic回归------------------------------------------
logistic=LogisticRegression(random_state=2,penalty="l2",C=0.214)
model1=AdaBoostClassifier(estimator=logistic,n_estimators=100,random_state=1)
model1.fit(X_train,y_train)
y_pred1=model1.predict(X_test)
print("构建的logistic+Adaboost分类模型的准确率为：",accuracy_score(y_test,y_pred1))
#---------------------------------(2)LGBT-----------------------------------------------
lgbt=LGBMClassifier(n_estimators=67,booster="gbtree",random_state=2)
lgbt.fit(X_train,y_train)
y_pred2=lgbt.predict(X_test)
print("构建的LGBT分类模型的准确率为：",accuracy_score(y_test,y_pred2))
#-----------------------------------(3)XGboost---------------------------------------------
xgb=XGBClassifier(n_estimators=25,booster="gbtree",random_state=2)
xgb.fit(X_train,y_train)
y_pred3=xgb.predict(X_test)
print("构建的XGboost分类模型的准确率为：",accuracy_score(y_test,y_pred3))
#-----------------------------------(4)KNN---------------------------------------------
KNN=KNeighborsClassifier(n_neighbors=3)
KNN.fit(X_train,y_train)
y_pred4=KNN.predict(X_test)
print("构建的KNN分类模型的准确率为：",accuracy_score(y_test,y_pred4))
#-----------------------------------(5)RandomForest---------------------------------------------
RF=RFC(n_estimators=142,criterion="entropy",random_state=1)
RF.fit(X_train,y_train)
y_pred5=RF.predict(X_test)
print("构建的RandomFores分类模型的准确率为：",accuracy_score(y_test,y_pred5))
#-----------------------------------（6）SVM多分类器---------------------------------------------------
svm=SVC(probability=True)
svm.fit(X_train,y_train)
y_pred6=svm.predict(X_test)
print("构建的SVM分类模型的准确率为：",accuracy_score(y_test,y_pred6))
#------------------------------Bagging集成模型一-----------------------------------------------------
voting=VotingClassifier(estimators=[('log',model1),("lgbt",lgbt),('KNN',KNN),('RF',RF),('svc',svm),('xgb',xgb)],voting='soft',weights=[1,1,1,1,1,1])
voting.fit(X_train,y_train)
y_pred=voting.predict(X_test)
print("构建的Bagging分类模型的准确率为：",accuracy_score(y_test,y_pred))
confusion = confusion_matrix(y_test, y_pred)
plt.imshow(confusion, cmap=plt.cm.Blues)
indices = range(len(confusion))

plt.colorbar()

plt.xlabel("真实标签")

plt.ylabel("预测的标签")

plt.title("混淆矩阵")
plt.xticks(indices, ['打保龄球','驾驶',"搏击",'射击','高尔夫','混合动作',"网球"])
plt.yticks(indices, ['打保龄球','驾驶',"搏击",'射击','高尔夫','混合动作',"网球"])
for first_index in range(len(confusion)):
    for second_index in range(len(confusion)):
        plt.text(first_index, second_index, confusion[first_index][second_index])

plt.show()


