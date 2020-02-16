# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 17:18:39 2019
@author: Sanmoy Paul
This module builds a deep learning model on Performance
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
import gensim
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
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, BatchNormalization, Input, Embedding, Flatten, GlobalMaxPool1D, GlobalAveragePooling1D, Dropout, Conv1D, MaxPooling1D, AveragePooling1D
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.losses import binary_crossentropy
from keras.optimizers import Adam, RMSprop, SGD
from keras import regularizers
import keras
from keras.models import Model
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] - %(message)s',
    filename='logs/performance.log',
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
        df[bucket] = (df[bucket] > 0).astype(int)
    
replace_values(df_train)
replace_values(df_test)


"""
    description: Tokenize the text of having dim 2000x200
"""
maxlen = 200
max_words = 2000
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


#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import normalize
x_train = get_features(df_train.Story)
#x_train=normalize(x_train)
y_perfTrain=df_train.iloc[:,9]
y_perfTrain=y_perfTrain.values

x_test = get_features(df_test.Story)
#x_test=normalize(x_test)
y_perfTest=df_test.iloc[:,9]
y_perfTest=y_perfTest.values

logging.info('Shape of x_train: %s', x_train.shape)
logging.info('Shape of x_test: %s', x_test.shape)
logging.info('Shape of y_train: %s', y_perfTrain.shape)
logging.info('Shape of y_test: %s', y_perfTest.shape)

logging.info("Counts of label '1': {}".format(sum(y_perfTrain==1)))
logging.info("Counts of label '0': {} \n".format(sum(y_perfTrain==0)))

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
#    layer = Dropout(0.3)(layer)
#    layer = Conv1D(64, 5, padding='same', activation='relu', strides=1)(layer)
#    #layer = Conv1D(64, 5, padding='same', activation='relu', strides=1)(layer)
#    layer = MaxPooling1D(pool_size=8)(layer)
#    layer = Dropout(0.2)(layer)
#    layer = LSTM(100, activation='relu')(layer)
#    layer = Dropout(0.1)(layer)
#    layer = Dense(1,name='out_layer')(layer)
#    layer = Activation('sigmoid')(layer)
#    model = Model(inputs=inputs,outputs=layer)
#    return model
#
#model = CNN_RNN()
#optimizer = Adam(lr=0.01, clipvalue=0.5)
#model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
#model.summary()
#
#callbacks = [
#    ReduceLROnPlateau(), 
#    EarlyStopping(monitor='val_loss', min_delta=0.01),
#    ModelCheckpoint(filepath='model/perf_model-conv1d.h5', save_best_only=True)
#]
#from sklearn.utils.class_weight import compute_class_weight
#class_weight=compute_class_weight("balanced", np.unique(y_perfTrain), y_perfTrain)
##class_weight={0:0.5, 1:0.5}
#history=model.fit(x_train, y_perfTrain,
#                    epochs=50,
#                    batch_size=32,
#                    validation_split=0.25,
#                    shuffle=True,
#                    callbacks=callbacks, class_weight=class_weight)

"""
    description: load the model from the saved directory
"""
model = load_model('model/perf_model-conv1d.h5')
logging.info("Performance model loaded successfully!!")

"""
    description: Model evaluation
"""
logging.info("================= Model Evaluation ===================")
metrics = model.evaluate(x_test, y_perfTest)
logging.info("Test {}: {}".format(model.metrics_names[0], metrics[0]))
logging.info("Test {}: {}".format(model.metrics_names[1], metrics[1]))

metrics_train = model.evaluate(x_train, y_perfTrain)
logging.info("Train {}: {}".format(model.metrics_names[0], metrics_train[0]))
logging.info("Train {}: {}".format(model.metrics_names[1], metrics_train[1]))


"""
    description: predict the original train data
    and build the confusion matrix.
"""
beforeSampleTrain_labels=model.predict(x_train)
y_trainPred_df=pd.DataFrame(beforeSampleTrain_labels)
y_trainPred_df.columns=["Performance"]
#y_trainPred_df.head(3)
#y_trainPred_df.to_csv("outputs/train_perfPrediction.csv", index=False)

y_train_df=pd.DataFrame(y_perfTrain)
y_train_df.columns=["Performance"]
#y_train_df.head(3)
#y_train_df.to_csv("outputs/train_perfData.csv", index=False)


y_trainPred_df.loc[y_trainPred_df["Performance"] >= 0.55, "Performance"]=1
y_trainPred_df.loc[y_trainPred_df["Performance"] < 0.55, "Performance"]=0
logging.info("Original Train Matrix")
logging.info(confusion_matrix(y_train_df.Performance, y_trainPred_df["Performance"], labels=[1,0]))
logging.info("\n")


"""
    description: predict unknown test data
    and build the confusion matrix.
"""
beforeSampleTest_labels=model.predict(x_test)
y_testPred_df=pd.DataFrame(beforeSampleTest_labels)
y_testPred_df.columns=["Performance"]
#y_testPred_df.to_csv("outputs/test_perfPrediction.csv", index=False)

y_test_df=pd.DataFrame(y_perfTest)
y_test_df.columns=["Performance"]
#y_test_df.to_csv("outputs/test_perfData.csv", index=False)

y_testPred_df.loc[y_testPred_df["Performance"] >= 0.55, "Performance"]=1
y_testPred_df.loc[y_testPred_df["Performance"] < 0.55, "Performance"]=0
logging.info("Original Test Matrix")
logging.info(confusion_matrix(y_test_df.Performance, y_testPred_df["Performance"], labels=[1,0]))
logging.info("\n")

