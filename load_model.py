# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 21:22:44 2018

@author: Administrator
"""

from sklearn.externals import joblib
from sklearn.datasets import load_digits

#导入数据
digits=load_digits()
x=digits.data
y=digits.target

#加载分类器
svc_clf=joblib.load('save/svc_clf.pkl')
print("load the model")

#测试
print("predict test for the model")
print(svc_clf.predict(x[0:2]))