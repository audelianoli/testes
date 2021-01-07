# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 12:20:32 2020

@author: auW10
"""

import numpy as np
import pandas as pd
import category_encoders as ce

from core.talib_dim_red import TaLib
from sklearn.preprocessing import LabelEncoder

class CreateDataFrame():

    def __init__(self, dataframe, configs):
        self.dataframe = dataframe
        self.configs = configs


    def create_df(self):
        # Selecionar apenas as colunas com DateTime e OHLCV
        data_ohlcv = pd.DataFrame(self.dataframe, columns=['DATE','TIME','OPEN','HIGH','LOW','CLOSE','VOL'])
        
        # Carregando cada coluna do dataframe em vetores 
        input_candlesticks = {
                                'date': np.array(data_ohlcv['DATE'].astype(str)),
                                'time': np.array(data_ohlcv['TIME'].astype(str)),
                                'open': np.array(data_ohlcv['OPEN'].astype(float)),
                                'high': np.array(data_ohlcv['HIGH'].astype(float)),
                                'low': np.array(data_ohlcv['LOW'].astype(float)),
                                'close': np.array(data_ohlcv['CLOSE'].astype(float)),
                                'volume': np.array(data_ohlcv['VOL'].astype(float))
                            }
         # Gerar os TIs através do TaLib
        indicadores = TaLib(input_candlesticks)
        indicadores_DB = indicadores.lista_de_indicadores()
        
        # Apagando linhas com dados NaN
        indicadores_DB.dropna(axis=0, inplace=True)
        
        # Concatenar colunas 'DATE' e 'TIME' e converter para DateTime
        indicadores_DB['DATETIME'] = indicadores_DB[['DATE', 'TIME']].agg(' '.join, axis=1)
        indicadores_DB.drop(['DATE', 'TIME'], axis=1,inplace=True)
        indicadores_DB['DATETIME'] = pd.to_datetime(indicadores_DB['DATETIME'])
        
        col = indicadores_DB.pop("DATETIME")
        indicadores_DB.insert(0, col.name, col)
        
        # Ajustar o index do DataFrame
        indicadores_DB = indicadores_DB.reset_index()
        indicadores_DB.drop('index', axis=1, inplace=True)
        
        '''
        Criação de Novas Features
        '''
        
        indicadores_DB['MES'] = indicadores_DB['DATETIME'].dt.month
        indicadores_DB['DIA_SEMANA'] = indicadores_DB['DATETIME'].dt.weekday
        
        def define_period(num):
            if num <= 11:
                return '1'
            else:
                return '2'
        
        indicadores_DB['MANHA_TARDE'] = indicadores_DB['DATETIME'].dt.hour.map(define_period)
            
        # Criando a coluna Target com diff(horizonte de previsão)
        horizon = self.configs['dataset']['horizon']
        indicadores_DB['TARGET_WIN'] = indicadores_DB['CLOSE'].diff(periods=horizon).shift(-horizon)    
            
        # Criando coluna Target Categórica

        indicadores_DB['TARGET_WIN_CAT3'] = indicadores_DB['TARGET_WIN'].map(self.define_multi_class)
        indicadores_DB['TARGET_WIN_CAT3_LABEL'] = indicadores_DB['TARGET_WIN_CAT3'].map(self.define_multi_class_label)
        
        ce_one_hot = ce.OneHotEncoder(cols = ['TARGET_WIN_CAT3'])

        X = indicadores_DB['TARGET_WIN_CAT3']
        y = indicadores_DB['TARGET_WIN_CAT3_LABEL']
        
        df2 = ce_one_hot.fit_transform(X, y)
        df2.columns = ['HOLD', 'SELL', 'BUY']
        df2.drop(['HOLD'], axis=1, inplace=True)
        
        indicadores_DB2 = pd.concat([indicadores_DB, df2], axis=1)
        indicadores_DB2.dropna(axis=0, inplace=True)
        
        filepath = 'C:/Users/auW10/Documents/CodigosDoAu/000_Doutorado/dataset/'
        indicadores_DB2.to_pickle(filepath+'dataset_with_TI_ciclo2.pkl')
        
        print(indicadores_DB2)
        
        return indicadores_DB2
    
    def define_multi_class(self, num):
            if num >= self.configs['dataset']['buy']:
                return 'BUY'
            elif num <= self.configs['dataset']['sell']:
                return 'SELL'
            else:
                return 'HOLD'
            
    def define_multi_class_label(self, num):
            if num == 'BUY':
                return 0
            elif num == 'SELL':
                return 1
            else:
                return 2
    
    def define_binary_class(self, num):
            if num >= 0:
                return 'BUY'
            else:
                return 'SELL'   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    