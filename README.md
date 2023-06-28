# Human-Action-Recognition-Algorithm-via-Human-Joint-Angle
人体动作识别是人机交互领域一个十分热门的话题，而且人体动作识别当中人体动作分类是一个比较经典的问题。在已经提出的人体动作分类方法中，基于手工提取特征的方法往往会对会受到一些噪声的影响，而基于深度学习的算法虽然没有这个问题又会耗费许多的算力。为此本文提出了一个基于人体关节夹角的人体动作识别算法，主要做了以下工作：
（1）提出了一个可解释性强，耗费算力较少且鲁棒性较高的基于人体关节夹角的人体动作序列的特征抽取方法
（2）本文所使用的分类模型是一个融合了SVM，lightgbm等6个分类模型的集成学习模型，并且使用了一个结合了Boosting和Bagging的集成学习策略。
最终本论文提出的方法在G3D数据集上的分类准确率为92.3%。该结果验证论文中方法的可行性

# Flowchart concerning the work
(1)The picture of the feature extraction is as fellow:  
![image](https://github.com/ynwu838/first-paper-ACTION-SEQUENCE-SIMILARITY-CALCULATION-ALGORITHM-via-OPENPOSE/blob/main/result/flowchart.png)  
(2)The picture of the hunman action similarity calculation method is as fellow:  
![image](https://github.com/ynwu838/first-paper-ACTION-SEQUENCE-SIMILARITY-CALCULATION-ALGORITHM-via-OPENPOSE/blob/main/result/similarity.png)
# Important files load
Here are some locations of some important files:  
Password tips:the most unfogettable year(with primary lover lcy)  
①keypoint get:https://pan.baidu.com/s/1Gxquf8uLAkDUf4PDfU4Nhg  
②feature extraction result(matrixes)：https://pan.baidu.com/s/1RZoBe5edfd8zp0I7ttc_4w  
③G3D database：http://dipersec.king.ac.uk/G3D/G3D.html
