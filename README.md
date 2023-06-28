# Human-Action-Recognition-Algorithm-via-Human-Joint-Angle
Human motion recognition is a very hot topic today, and human motion recognition has
penetrated into many aspects of information technology today, such as video retrieval, identity
recognition, etc. Human action recognition has always been a very technical subject in the field
of computer vision, and human action classification is a important component of human action
recognition. With the development of machine learning, more and more related methods have been
proposed. However, the shortcomings of this kind of method is obvious: the method based on
manual feature extraction is often affected by some noise, And the method based on deep
learning will consume a lot of computing resource. Hence, we propose a human action sequence
feature extraction method based on the angle between human joints. In order to accomplish the
classification task better, the classification model that we used in the thesis is an ensemble
learning classification model based on Bagging strategy of soft voting . The method proposed in the paper combines the advantages of traditional machine learning
methods and deep learning methods: we firstly use the pre-training model of deep learning
method to extract the keypoint of human motion from the human motion data . Therefore，the
result of the method will not be affected by some noise. Then we use traditional manual design
features with strong interpretability and low computational consumption and machine learning
methods for modeling. We first use this human action sequence feature extraction method to extract feature vectors, and then apply feature selection to extracted feature vectors. Then use the feature vector which
go through feature selection for model training and model verification, and the ratio of training
set and verification set is divided into 5:5. In this paper, the G3D data set is used as the dataset, and finally the method proposed in this paper has a performance of 92.3% on G3D. This result
verifies the feasibility of the method in the paper
Keywords: G3D, Human Joint Angle, Openpose, Ensemble Learning
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
