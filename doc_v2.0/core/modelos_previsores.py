# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 16:19:03 2020


Modelos previsores - doc ciclo 2
    Classificadores [BUY, HOLD, SELL]

@author: auW10    
"""

import os
import numpy as np
import csv
import time
import lightgbm as lgbm
import seaborn as sns

from numpy import newaxis
from core.utils import Timer
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from keras.layers import Dense, Dropout, LSTM, GRU
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, Normalizer

class ModelRandomForest():
    """ Classe para criar um modelo Random Forest """
    
    def __int__(self):
        self.model = RandomForestClassifier(n_jobs=-1, random_state=0, n_estimators=5000)

class ModelLGBM():
    """ Classe para criar um modelo LGBM """
    
    def __int__(self):
        self.model = lgbm.LGBMClassifier( n_jobs=-1, random_state=0, n_estimators=5000, learning_rate=0.001, num_leaves=2**6,
                                          subsample=0.9, subsample_freq=1, colsample_bytree=1. )

class ModelLSTM():
    """ Classe para criar um modelo LSTM """
    
    def __int__(self):
        self.model = Sequential()
    
    def build_model(self, configs, timesteps, dim):
        timer = Timer()
        timer.start()
       
class ModelDNN():
    """ Classe para criar um modelo DNN """
    
    def __int__(self):
        self.model = Sequential()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
