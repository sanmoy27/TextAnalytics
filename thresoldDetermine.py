# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:21:17 2019

@author: PAULSA-CONT
"""

import numpy as np
import pandas as pd
from sklearn import metrics

import os
path="D:\\clustering\\CI_Data_UnsupervisedClustering\\CI_Data_UnsupervisedClustering"
os.chdir(path)


analysisData=pd.read_csv("data/confusionMatrixAnalysisData.csv")
metrics.confusion_matrix(analysisData.Price_F, analysisData.Price_RNN_CNN, labels=[1,0])
metrics.confusion_matrix(analysisData.Product_F, analysisData.Product_RNN_CNN, labels=[1,0])
metrics.confusion_matrix(analysisData.Distribution_F, analysisData.Distribution_RNN_CNN, labels=[1,0])
metrics.confusion_matrix(analysisData.New_Revenue_F, analysisData.New_Revenue_RNN_CNN, labels=[1,0])
metrics.confusion_matrix(analysisData.Performance_F, analysisData.Performance_RNN_CNN, labels=[1,0])
metrics.confusion_matrix(analysisData.Technology_F, analysisData.Technology_RNN_CNN, labels=[1,0])
metrics.confusion_matrix(analysisData.Operations_F, analysisData.Operations_RNN_CNN, labels=[1,0])

metrics.confusion_matrix(analysisData.Price_F, analysisData.Price_LDA, labels=[1,0])
metrics.confusion_matrix(analysisData.Product_F, analysisData.Product_LDA, labels=[1,0])
metrics.confusion_matrix(analysisData.Distribution_F, analysisData.Distribution_LDA, labels=[1,0])
metrics.confusion_matrix(analysisData.New_Revenue_F, analysisData.New_Revenue_LDA, labels=[1,0])
metrics.confusion_matrix(analysisData.Performance_F, analysisData.Performance_LDA, labels=[1,0])
metrics.confusion_matrix(analysisData.Technology_F, analysisData.Technology_LDA, labels=[1,0])
metrics.confusion_matrix(analysisData.Operations_F, analysisData.Operations_LDA, labels=[1,0])

metrics.confusion_matrix(analysisData.Price_F, analysisData.Price_C, labels=[1,0])
metrics.confusion_matrix(analysisData.Product_F, analysisData.Product_C, labels=[1,0])
metrics.confusion_matrix(analysisData.Distribution_F, analysisData.Distribution_C, labels=[1,0])
metrics.confusion_matrix(analysisData.New_Revenue_F, analysisData.New_Revenue_C, labels=[1,0])
metrics.confusion_matrix(analysisData.Performance_F, analysisData.Performance_C, labels=[1,0])
metrics.confusion_matrix(analysisData.Technology_F, analysisData.Technology_C, labels=[1,0])
metrics.confusion_matrix(analysisData.Operations_F, analysisData.Operations_C, labels=[1,0])

metrics.confusion_matrix(analysisData.Price_F, analysisData.Price_DT, labels=[1,0])
metrics.confusion_matrix(analysisData.Product_F, analysisData.Product_DT, labels=[1,0])
metrics.confusion_matrix(analysisData.Distribution_F, analysisData.Distribution_DT, labels=[1,0])
metrics.confusion_matrix(analysisData.New_Revenue_F, analysisData.New_Revenue_DT, labels=[1,0])
metrics.confusion_matrix(analysisData.Performance_F, analysisData.Performance_DT, labels=[1,0])
metrics.confusion_matrix(analysisData.Technology_F, analysisData.Technology_DT, labels=[1,0])
metrics.confusion_matrix(analysisData.Operations_F, analysisData.Operations_DT, labels=[1,0])

feedback=pd.read_csv("data/feedBack_codedData.csv")
lda_prob=pd.read_csv("data/lda_thresoldData.csv")

lda_prob.loc[lda_prob["performance"] >= 0.27, "performance"]=1
lda_prob.loc[lda_prob["performance"] < 0.27, "performance"]=0
metrics.confusion_matrix(feedback.Performance, lda_prob.performance, labels=[1,0])

lda_prob.loc[lda_prob["price"] >= 0.1, "price"]=1
lda_prob.loc[lda_prob["price"] < 0.1, "price"]=0
metrics.confusion_matrix(feedback.Price, lda_prob.price, labels=[1,0])


lda_prob.loc[lda_prob["product"] >= 0.041, "product"]=1
lda_prob.loc[lda_prob["product"] < 0.041, "product"]=0
metrics.confusion_matrix(feedback.Product, lda_prob["product"], labels=[1,0])


lda_prob.loc[lda_prob["distribution"] >= 0.2152, "distribution"]=1
lda_prob.loc[lda_prob["distribution"] < 0.2152, "distribution"]=0
metrics.confusion_matrix(feedback.Distribution, lda_prob["distribution"], labels=[1,0])


lda_prob.loc[lda_prob["revenue"] >=0.0259, "revenue"]=1
lda_prob.loc[lda_prob["revenue"] < 0.0259, "revenue"]=0
metrics.confusion_matrix(feedback.New_Revenue, lda_prob["revenue"], labels=[1,0])


lda_prob.loc[lda_prob["technology"] >= 0.1271271, "technology"]=1
lda_prob.loc[lda_prob["technology"] < 0.1271271, "technology"]=0
metrics.confusion_matrix(feedback.Technology, lda_prob["technology"], labels=[1,0])


lda_prob.loc[lda_prob["operations"] >= 0.04704705, "operations"]=1
lda_prob.loc[lda_prob["operations"] < 0.04704705, "operations"]=0
metrics.confusion_matrix(feedback.Operations, lda_prob["operations"], labels=[1,0])







feedback_test=pd.read_csv("data/feedBack_codedDataTest.csv")
lda_probTest=pd.read_csv("data/lda_thresoldDataTest.csv")

lda_probTest.loc[lda_probTest["performance"] >= 0.27, "performance"]=1
lda_probTest.loc[lda_probTest["performance"] < 0.27, "performance"]=0
metrics.confusion_matrix(feedback_test.Performance, lda_probTest.performance, labels=[1,0])

lda_probTest.loc[lda_probTest["price"] >= 0.1, "price"]=1
lda_probTest.loc[lda_probTest["price"] < 0.1, "price"]=0
metrics.confusion_matrix(feedback_test.Price, lda_probTest.price, labels=[1,0])


lda_probTest.loc[lda_probTest["product"] >= 0.041, "product"]=1
lda_probTest.loc[lda_probTest["product"] < 0.041, "product"]=0
metrics.confusion_matrix(feedback_test.Product, lda_probTest["product"], labels=[1,0])


lda_probTest.loc[lda_probTest["distribution"] >= 0.2152, "distribution"]=1
lda_probTest.loc[lda_probTest["distribution"] < 0.2152, "distribution"]=0
metrics.confusion_matrix(feedback_test.Distribution, lda_probTest["distribution"], labels=[1,0])


lda_probTest.loc[lda_probTest["revenue"] >=0.0259, "revenue"]=1
lda_probTest.loc[lda_probTest["revenue"] < 0.0259, "revenue"]=0
metrics.confusion_matrix(feedback_test.New_Revenue, lda_probTest["revenue"], labels=[1,0])


lda_probTest.loc[lda_probTest["technology"] >= 0.1271271, "technology"]=1
lda_probTest.loc[lda_probTest["technology"] < 0.1271271, "technology"]=0
metrics.confusion_matrix(feedback_test.Technology, lda_probTest["technology"], labels=[1,0])


lda_probTest.loc[lda_probTest["operations"] >= 0.04704705, "operations"]=1
lda_probTest.loc[lda_probTest["operations"] < 0.04704705, "operations"]=0
metrics.confusion_matrix(feedback_test.Operations, lda_probTest["operations"], labels=[1,0])




################# Oversampled Data Analysis ########################
train_smData=pd.read_csv("data/validData/train_smData.csv")
train_smPred=pd.read_csv("data/validData/train_smPricePrediction1.csv")

train_smPred.loc[train_smPred["Price"] >= 0.41, "Price"]=1
train_smPred.loc[train_smPred["Price"] < 0.41, "Price"]=0
metrics.confusion_matrix(train_smData.Price, train_smPred["Price"], labels=[1,0])





