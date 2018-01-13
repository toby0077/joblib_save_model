# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 23:13:36 2018

@author: Toby webchat:drug666123,QQ:231469242

overffiting,过渡拟合，找到最佳param，保存分类器

SVC模型
"""
from sklearn.externals import joblib
from sklearn.learning_curve import validation_curve
from sklearn.learning_curve import learning_curve
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
#导入数据预处理，包括标准化处理或正则处理
from sklearn import preprocessing
#样本平均测试，评分更加
from sklearn.cross_validation import cross_val_score
 
from sklearn import datasets
#导入knn分类器
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

#knn=KNeighborsClassifier(n_neighbors=12)

#导入数据
digits=load_digits()
x=digits.data
y=digits.target
#参数范围设定
param_range=np.logspace(-6,-2.3,5)
#得分方法
#scoring="mean_squared_error"
#scoring="accuracy"

train_loss,test_loss=validation_curve(SVC(),x,y,param_name='gamma',
    param_range=param_range, cv=10,scoring="neg_mean_squared_error") 
                                            
 
train_loss_mean=-np.mean(train_loss,axis=1)
test_loss_mean=-np.mean(test_loss,axis=1)


#绘图
plt.plot(param_range,train_loss_mean,'o-',color='r',label='Training')
plt.plot(param_range,test_loss_mean,'o-',color='g',label='Cross-validation')

plt.xlabel("gamma")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.show()


#寻找最佳param
#字典，存放train_loss_mean，test_loss_mean，param_range组合
a=train_loss_mean+test_loss_mean
b=zip(param_range,a)
dict_param_train_test_Loss=dict(b)

#利用lambda找最值选项
min_item=min(dict_param_train_test_Loss.items(), key=lambda x: x[1]) 
best_param=min_item[0]
print("best_param is:",best_param)

svc_clf=SVC(gamma=best_param)
#训练数据
svc_clf.fit(x,y)
print("Done fit the model")

#save,保存分类器
joblib.dump(svc_clf,'save/svc_clf.pkl')
print("save the model")






