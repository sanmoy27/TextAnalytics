# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 17:18:39 2019
@author: Sanmoy Paul
This module builds a deep learning model on Product
variable to classify into threat and no threat category.
"""

import os
path="/home/maguser/Sanmoy/RNN_CNN"
os.chdir(path)

import numpy as np
import pandas as pd
import regex as re
from sklearn.neural_network import MLPRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from gensim.models.doc2vec import TaggedDocument
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from string import punctuation
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr=WordNetLemmatizer()
stop=set(stopwords.words('english')+['“','’', '”', '’', '``', '‘', '=', ':',  '``', "the", "as", "crore", 'rs', 'j', 'copywright', 'click', 'here', "escorts", "escort", "john deere", "kubota", "new holland", "sonalika", "tafe", "vst tillers", "tractor", "said", "cm", "chowdari", "year", "holland", "compani", "farm", "farmer", "month", "decemb", "market", "chairman", "countri", "tiller", "r", "india", "john", "deer", "k", "delhi", "pakistan", "algeria"])
exclude=set(punctuation)
from sklearn import utils
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.utils.class_weight import compute_class_weight
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, BatchNormalization, Input, Embedding, Flatten, GlobalMaxPool1D, GlobalAveragePooling1D, Dropout, Conv1D, MaxPooling1D
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.losses import binary_crossentropy
from keras.optimizers import Adam, RMSprop, SGD
from keras import regularizers
import keras
from keras.models import Model
from keras.models import load_model
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
import logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] - %(message)s',
    filename='logs/product.log',
    filemode='a')

"""
    function_name: pre_process
    description: removes stop words and punctuation marks, word tokenization
    return_value: returns cleaned text
"""
def pre_process(text):
    filtered_tokens=[]
    for sent in sent_tokenize(text):
        s=" ".join([lmtzr.lemmatize(re.sub(r"([a-z])([A-Z])","\g<1> \g<2>", word).lower()) for word in word_tokenize(sent) if word.isalpha()])
        s=" ".join([word for word in word_tokenize(s) if not word in stop])
        filtered_tokens.append(s)
    return filtered_tokens


"""
    function_name: processed_text
    description: The prec-processed text is subjected to further processing 
    in order to replace the blank Story column with Head attribute value
"""
def processed_text(df):
    for idx, row in enumerate(df['Story']):
        if pd.isnull(row):
            df.loc[idx, 'Story']=". ".join(pre_process(df.loc[idx, 'Head']))+"."
        else:
            df.loc[idx, 'Story']=". ".join(pre_process(df.loc[idx, 'Story']))+"."
            
            
df_train=pd.read_csv("data/train_clust.csv")
df_test=pd.read_csv("data/test_clust.csv")

processed_text(df_train)
processed_text(df_test)

"""
    function_name: replace_values
    description: Replace the values in bucket attributes with 0 and 1
    zero values are left as it is but greater than zero are replaced
    with 1 indicating some amount of threat is present.
"""
def replace_values(df):
    buckets=["Price_y", "Product_y", "Distribution_y", "New_Revenue_y", "Technology_y", "Performance_y", "Operations_y"]
    for bucket in buckets:
        df.loc[df[bucket] > 0, bucket]=1 
    
replace_values(df_train)
replace_values(df_test)




#from sklearn.preprocessing import MultiLabelBinarizer
#multilabel_binarizer = MultiLabelBinarizer()
#multilabel_binarizer.fit(df_questions.Tags)
#labels = multilabel_binarizer.classes_
"""
    description: Tokenize the text of having dim 1000x150
"""
maxlen = 150
max_words = 1000
tokenizer = Tokenizer(num_words=max_words, lower=True)
tokenizer.fit_on_texts(df_train.Story)
tokenizer.fit_on_texts(df_test.Story)

"""
    function_name: replace_values
    description: transforms text data to feature_vectors that can be used in the ml model.
    tokenizer must be available.
"""
def get_features(text_series):
    sequences = tokenizer.texts_to_sequences(text_series)
    return pad_sequences(sequences, maxlen=maxlen)




#from sklearn.preprocessing import normalize
x_train = get_features(df_train.Story)
#x_train=normalize(x_train)
y_prodTrain=df_train.iloc[:,5]
y_prodTrain=y_prodTrain.values

x_test = get_features(df_test.Story)
#x_test=normalize(x_test)
y_prodTest=df_test.iloc[:,5]
y_prodTest=y_prodTest.values

logging.info("Before OverSampling, counts of label '1': {}".format(sum(y_prodTrain==1)))
logging.info("Before OverSampling, counts of label '0': {} \n".format(sum(y_prodTrain==0)))

"""
    description: Applied SMOTE to deal with the issue of class
    imbalance problem, oversampled the minority class with
    more than 15% event rate.
"""
sm=SMOTE(random_state=2,k_neighbors=2, ratio=0.18)
X_smProd, Y_smProd=sm.fit_sample(x_train, y_prodTrain)

logging.info("After OverSampling, counts of label '1': {}".format(sum(Y_smProd==1)))
logging.info("After OverSampling, counts of label '0': {}".format(sum(Y_smProd==0)))

x_smTrain, x_smTest, y_smTrain, y_smTest = train_test_split(X_smProd, Y_smProd, test_size=0.23, random_state=9000)
logging.info("Oversampled x_smTrain shape: %s", x_smTrain.shape)
logging.info("Oversampled x_smTest shape: %s", x_smTest.shape)
logging.info("Oversampled y_smTrain shape: %s", y_smTrain.shape)
logging.info("Oversampled y_smTest shape: %s", y_smTest.shape)

"""
    function_name: CNN_RNN
    description: This function describes a RCNN architecture with
    a LSTM layer after the convulated layes.
    Early stopping is used to have an l2 regularization effect
    The minority class is given more weightage during model building.
    return_value: model object
"""
#def CNN_RNN():
#    inputs = Input(name='inputs',shape=[maxlen])
#    layer = Embedding(max_words,32,input_length=maxlen)(inputs)
#    layer = Dropout(0.35)(layer)
#    layer = Conv1D(64, 5, padding='same', activation='relu', strides=1, activity_regularizer=regularizers.l2())(layer)
#    layer = MaxPooling1D(pool_size=4)(layer)
#    layer = Dropout(0.2)(layer)
#    layer = LSTM(100)(layer)
#    layer = Dropout(0.15)(layer)
#    layer = Dense(2,name='out_layer')(layer)
#    layer = Activation('softmax')(layer)
#    model = Model(inputs=inputs,outputs=layer)
#    return model
#    
#
#model = CNN_RNN()
#optimizer = Adam(lr=0.01, clipvalue=0.5)
#model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
#model.summary()
#
#callbacks = [
#    ReduceLROnPlateau(), 
#    EarlyStopping(monitor='val_loss',min_delta=0.01),
#    ModelCheckpoint(filepath='model/prod_model-conv1d1.h5', save_best_only=True)
#]
#
#class_weight=compute_class_weight("balanced", np.unique(y_smTrain), y_smTrain)
#history=model.fit(x_smTrain, y_smTrain,
#                    epochs=15,
#                    batch_size=5,
#                    validation_split=0.4,
#                    callbacks=callbacks, class_weight=class_weight)


"""
    description: load the model from the saved directory
"""
model = load_model('model/prod_model-conv1d.h5')
logging.info("Product model loaded successfully!!")

"""
    description: Model evaluation
"""
logging.info("================= Model Evaluation ===================")
metrics_test = model.evaluate(x_smTest, y_smTest)
logging.info("Test {}: {}".format(model.metrics_names[0], metrics_test[0]))
logging.info("Test {}: {}".format(model.metrics_names[1], metrics_test[1]))

metrics_train = model.evaluate(x_smTrain, y_smTrain)
logging.info("Train {}: {}".format(model.metrics_names[0], metrics_train[0]))
logging.info("Train {}: {}".format(model.metrics_names[1], metrics_train[1]))

"""
    description: predict the oversampled train data
    and build the confusion matrix.
"""
train_labels=model.predict(x_smTrain)
y_smTrainPred_df=pd.DataFrame(train_labels)
y_smTrainPred_df.columns=["Product"]
#print("SM Prediction shape: ", y_smTrainPred_df.shape)
#y_smTrainPred_df.to_csv("outputs/train_smProdPrediction.csv", index=False)

y_smTrain_df=pd.DataFrame(y_smTrain)
y_smTrain_df.columns=["Product"]
#print("SM_train shape: ", y_smTrain_df.shape)
#y_smTrain_df.to_csv("outputs/train_smProdData.csv", index=False)

y_smTrainPred_df.loc[y_smTrainPred_df["Product"] >= 0.26, "Product"]=1
y_smTrainPred_df.loc[y_smTrainPred_df["Product"] < 0.26, "Product"]=0
logging.info("Oversampled Train Matrix")
logging.info(confusion_matrix(y_smTrain_df.Product, y_smTrainPred_df["Product"], labels=[1,0]))
logging.info("\n")


"""
    description: predict the oversampled test data
    and build the confusion matrix.
"""
test_labels=model.predict(x_smTest)
y_smTestPred_df=pd.DataFrame(test_labels)
y_smTestPred_df.columns=["Product"]
#y_smTestPred_df.head(3)
#y_smTestPred_df.to_csv("outputs/test_smProdPrediction.csv", index=False)

y_smTest_df=pd.DataFrame(y_smTest)
y_smTest_df.columns=["Product"]
#y_smTest_df.to_csv("outputs/test_smProdData.csv", index=False)

y_smTestPred_df.loc[y_smTestPred_df["Product"] >= 0.26, "Product"]=1
y_smTestPred_df.loc[y_smTestPred_df["Product"] < 0.26, "Product"]=0
logging.info("Oversampled Test Matrix")
logging.info(confusion_matrix(y_smTest_df.Product, y_smTestPred_df["Product"], labels=[1,0]))
logging.info("\n")

"""
    description: predict the original train data
    and build the confusion matrix.
"""
beforeSampleTrain_labels=model.predict(x_train)
y_trainPred_df=pd.DataFrame(beforeSampleTrain_labels)
y_trainPred_df.columns=["Product"]
#y_trainPred_df.head(3)
#y_trainPred_df.to_csv("outputs/train_prodPrediction.csv", index=False)

y_train_df=pd.DataFrame(y_prodTrain)
y_train_df.columns=["Product"]
#y_train_df.head(3)
#y_train_df.to_csv("outputs/train_prodData.csv", index=False)

y_trainPred_df.loc[y_trainPred_df["Product"] >= 0.26, "Product"]=1
y_trainPred_df.loc[y_trainPred_df["Product"] < 0.26, "Product"]=0
logging.info("\Original Train Matrix")
logging.info(confusion_matrix(y_train_df.Product, y_trainPred_df["Product"], labels=[1,0]))
logging.info("\n")

"""
    description: predict unknown test data
    and build the confusion matrix.
"""
beforeSampleTest_labels=model.predict(x_test)
y_testPred_df=pd.DataFrame(beforeSampleTest_labels)
y_testPred_df.columns=["Product"]
#y_testPred_df.head(3)
#y_testPred_df.to_csv("outputs/test_prodPrediction.csv", index=False)

y_test_df=pd.DataFrame(y_prodTest)
y_test_df.columns=["Product"]
#y_test_df.head(3)
#y_test_df.to_csv("outputs/test_prodData.csv", index=False)

y_testPred_df.loc[y_testPred_df["Product"] >= 0.26, "Product"]=1
y_testPred_df.loc[y_testPred_df["Product"] < 0.26, "Product"]=0
logging.info("\Original Test Matrix")
logging.info(confusion_matrix(y_test_df.Product, y_testPred_df["Product"], labels=[1,0]))
logging.info("\n")
