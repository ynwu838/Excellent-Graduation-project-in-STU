# 论文摘要
人体动作识别是人机交互领域一个十分热门的话题，而且人体动作识别当中人体动作分类是一个比较经典的问题。在已经提出的人体动作分类方法中，基于手工提取特征的方法往往会对会受到一些噪声的影响，而基于深度学习的算法虽然没有这个问题又会耗费许多的算力。为此本文提出了一个基于人体关节夹角的人体动作识别算法，主要做了以下工作：  
（1）提出了一个可解释性强，耗费算力较少且鲁棒性较高的基于人体关节夹角的人体动作序列的特征抽取方法。  
（2）本文所使用的分类模型是一个融合了SVM，lightgbm等6个分类模型的集成学习模型，并且使用了一个结合了Boosting和Bagging的集成学习策略。  
最终本论文提出的方法在G3D数据集上的分类准确率为92.3%。该结果验证论文中方法的可行性  

# 论文提出的算法的流程图 
![image](https://github.com/ynwu838/Human-Action-Recognition-Algorithm-via-Human-Joint-Angle/blob/main/%E5%9B%BE%E7%89%87/602c46f727825cca6fbfb01fa5a40ce.png)  
本文当中所提出的人体动作分类算法如上图所示，本算法主要分为人体3D关节点坐标获取，特征工程，模型集成，模型训练4个部分:  
（1）人体关节点坐标获取部分：先使用Openpose的body25预训练模型结合训练集自带的深度图来提取出人体关键点3D坐标  
（2）特征工程部分：先抽取人体动作特征向量，具体方法为：  
       ①在每一个人体动作序列中等间距取50张图片  
       ②基于得到的人体关键点3D坐标，每一张图片我们都可以得到 24 个肢体向量。然后通过24 个肢体向量两两之间的夹角的余弦值，可以得到一个长度为276的向量  
       ③将50张图片得到的向量按照顺序拼接成长为13800的向量而后用基于方差的Filter方法和基于随机森林的RFE方法来特征选择。  
（3）模型集成部分：因为logistic回归拟合能力较弱，所以先让它进行Adaboost增强拟合能力，而后再把它和SVM，KNN，随机森林，Xgboost，lightgbm一起进行基于软投票策略的Bagging方法后输出最终的分类结果。    
（4）模型训练部分：按照训练集和测试集5:5的比例划分数据集，而后按照划分的数据集使用网格调参法调出每一个基分类模型的最佳参数，而后将其带入集成学习模型当中进行最终的模型分类预测  
# 论文方法的实验以及实验结果结果  
本文先使用网格调参法调出基分类器的最佳参数，具体调参的过程如下图所示：![image](https://github.com/ynwu838/Excellent-Graduation-project-in-STU/blob/main/%E5%9B%BE%E7%89%87/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20230628190416.png)  
根据调参的结果，各个基分类器的最佳参数如下：  
（1）logistic回归：“L2”惩罚项，C取0.214  
（2）支持向量机：核函数取“poly”，C取0.238  
（3）KNN：使用kdtree进行搜索，K值取3  
（4）随机森林：estimators的最佳参数为142，criterion选择“entropy”这个选项  
（5）Xgboost：最佳estimators的个数为25  
（6）Lightgbm：最佳estimators的个数为67  
（7）在实验当中Adaboost中的基分类器的个数设定为100  
将调参结果输入到集成分类学习模型当中以后分类准确率达到了92.3%，得到的分类结果的混淆矩阵如下图所示  
![image](https://github.com/ynwu838/Human-Action-Recognition-Algorithm-via-Human-Joint-Angle/blob/main/%E5%9B%BE%E7%89%87/%E5%9B%BE%E7%89%871.png) 

# 一些重要东西的下载
①数据集使用的是G3D数据集：http://dipersec.king.ac.uk/G3D/G3D.html    
②动作的特征向量数据： https://pan.baidu.com/s/12kwbcNEpWg86XSV-cOMKMQ  
密码为2009
# 代码运行
 ```cd ./code```   
①集成分类模型运行结果  ```python proposed_method.py```   
②SVM画图  ```python SVM.py```   
①KNN画图  ```python KNN.py```   
④随机森林画图  ```python randomforest.py```   
⑤xgboost和Lightgbm画图  ```python xgboost_and_lightgbm.py```   
⑥logistic画图  ```python logistic.py```
⑦特征抽取和特征选择：先把动作的特征向量数据下载好以后放到code/data目录下，而后运行 ```python feature_extracting.py```      
⑧如果你想直接使用论文实验的数据的话就把Experimental data 文件夹里面的数据放到code文件夹下面，这样就不用再跑一遍特征抽取了

