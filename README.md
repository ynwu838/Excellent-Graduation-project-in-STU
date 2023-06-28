# 论文摘要
人体动作识别是人机交互领域一个十分热门的话题，而且人体动作识别当中人体动作分类是一个比较经典的问题。在已经提出的人体动作分类方法中，基于手工提取特征的方法往往会对会受到一些噪声的影响，而基于深度学习的算法虽然没有这个问题又会耗费许多的算力。为此本文提出了一个基于人体关节夹角的人体动作识别算法，主要做了以下工作：  
（1）提出了一个可解释性强，耗费算力较少且鲁棒性较高的基于人体关节夹角的人体动作序列的特征抽取方法。  
（2）本文所使用的分类模型是一个融合了SVM，lightgbm等6个分类模型的集成学习模型，并且使用了一个结合了Boosting和Bagging的集成学习策略。  
最终本论文提出的方法在G3D数据集上的分类准确率为92.3%。该结果验证论文中方法的可行性  

# 论文提出的算法的流程图 
![image](https://github.com/ynwu838/Human-Action-Recognition-Algorithm-via-Human-Joint-Angle/blob/main/%E5%9B%BE%E7%89%87/602c46f727825cca6fbfb01fa5a40ce.png) 

# 论文方法的结果图
![image](https://github.com/ynwu838/Human-Action-Recognition-Algorithm-via-Human-Joint-Angle/blob/main/%E5%9B%BE%E7%89%87/%E5%9B%BE%E7%89%871.png)
# 一些重要东西的下载
①论文原文链接：https://pan.baidu.com/s/1AKD0U35UxwT87RhKgANRBA（提示：小学，lcy，同桌）  
②数据集使用的是G3D数据集，在http://dipersec.king.ac.uk/G3D/G3D.html。可以找到，如果需要中间的数据处理过程的话去另一个仓库就可以看到了
