__author__ = "Audeliano Li"
__date__ = "2021/03/13"
__version__ = "2.0.0"

import numpy as np
import time

from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

class DataLoader():
    """A class for loading and transforming data for the lstm model"""

    def __init__(self, configs, dataframe):
        
        df2 = dataframe.drop([ 'OPEN',                  
                               'HIGH',                   
                               'LOW',
                               'CLOSE',                   
                               'VOL',
                               'MES',
                               'TARGET_WIN',
                               'TARGET_WIN_CAT3', 
                               'TARGET_WIN_CAT3_LABEL',      
                               'PRICE_PERCENTAGE'], 
                            axis=1)
        
        train_data_inicio = configs['dataset']['train_data_inicio']+'-01-01'
        valid_data_inicio = configs['dataset']['valid_data_inicio']+'-01-01'
        valid_data_fim = configs['dataset']['valid_data_fim']+'-01-01'
        test_data_fim = configs['dataset']['test_data_fim']+'-01-01'
  
        '''
        # Descomentar apenas para testes rápidos
        train_data_inicio = '2017-01-01'
        valid_data_inicio = '2018-01-01'
        valid_data_fim = '2018-07-01'
        test_data_fim = '2019-01-01'
        '''
        
        df_train = df2[(df2['DATETIME'] >= train_data_inicio) & 
                       (df2['DATETIME'] < valid_data_inicio)]        
        df_valid = df2[(df2['DATETIME'] >= valid_data_inicio) & 
                       (df2['DATETIME'] < valid_data_fim)]     
        df_test  = df2[(df2['DATETIME'] >= valid_data_fim) & 
                       (df2['DATETIME'] < test_data_fim)]
        
        print('\n\nProporção Train: {:.1%} | Validation: {:.1%} | Test: {:.1%}'.format(
                                        df_train['DATETIME'].count()/df2['DATETIME'].count(),
                                        df_valid['DATETIME'].count()/df2['DATETIME'].count(),
                                        df_test['DATETIME'].count()/df2['DATETIME'].count()))
        
        
        df_train.drop(['DATETIME', 'DATE', 'TIME'], axis=1, inplace=True)
        df_valid.drop(['DATETIME', 'DATE', 'TIME'], axis=1, inplace=True)
        df_test.drop(['DATETIME', 'DATE', 'TIME'], axis=1, inplace=True)
        
        self.df_train = df_train
        self.df_valid = df_valid
        self.df_test = df_test
        self.seq_len = configs['dataset']['sequence_length']
        
        print(df_train.shape, df_valid.shape, df_test.shape)
        print('\n\n')
        
    def get_train(self):
        return self.get_train_valid_test_data(1, self.df_train)
        
    def get_valid(self):
        return self.get_train_valid_test_data(1, self.df_valid)

    def get_test(self):
        return self.get_train_valid_test_data(1, self.df_test)
    
    def get_train_sw(self):
        return self.get_train_valid_test_data_Slide_Window(self.seq_len, self.df_train)
        
    def get_valid_sw(self):
        return self.get_train_valid_test_data_Slide_Window(self.seq_len, self.df_valid)

    def get_test_sw(self):
        return self.get_train_valid_test_data_Slide_Window(self.seq_len, self.df_test)
    

    def undersampling(self, df):
               
        # Separando apenas os dados que estejam dentro do HORÁRIO NOBRE
        # print("aaaaaaaaaaa: ", df.groupby(['HORA_NOBRE'])['TARGET_PERCENTAGE'].value_counts())
        num_buy = df.groupby(['HORA_NOBRE'])['TARGET_PERCENTAGE'].value_counts()[2]
        num_sell = df.groupby(['HORA_NOBRE'])['TARGET_PERCENTAGE'].value_counts()[1]
        num_hold = df.groupby(['HORA_NOBRE'])['TARGET_PERCENTAGE'].value_counts()[0]
        
        # print('\nDados Categóricos na Hora Nobre -- BUY: %d | SELL: %d | HOLD: %d' % (num_buy, num_sell, num_hold))
        
        np.random.seed(2345)
        porcentagem_selecao = (num_buy / num_hold)
        amostra = np.random.choice( a=[0,1], size=num_hold, replace=True, 
                                    p=[1-porcentagem_selecao, porcentagem_selecao] )
        # print('Nº total da amostra: ', len(amostra))
        # print('Nº de dados selecionados: ', len(amostra[amostra == 1]))
        # print('Nº de dados restantes que não serão considerados: ', len(amostra[amostra == 0]))
        
        return amostra

    def get_train_valid_test_data(self, seq_len, dataframe):  
        df = dataframe[dataframe['HORA_NOBRE'] == '1'].reset_index(drop=True)
        # print('df.shape: ', df.shape)
        amostragem = self.undersampling(df)
        a=0
        # Loop para carregar apenas os dados do 'Undersampling'
        for i in range(len(df) - seq_len):
            if df['TARGET_PERCENTAGE'][i+seq_len-1] == 'HOLD':
                if amostragem[a] == 0:
                    df.drop((i+seq_len-1), axis=0, inplace=True) 
                a+=1
    
        x, y = self.normalize(df)

        return x, y
        
    # Função para carregar os dados de treino, validação e teste usando técnica de Slide Window
    def get_train_valid_test_data_Slide_Window(self, seq_len, dataframe):
        df = dataframe[dataframe['HORA_NOBRE'] == '1'].reset_index(drop=True)
        # print('df.shape: ', df.shape)
        amostragem = self.undersampling(df)
        a=0
        data_x = []
        data_y = []
        for i in range(len(df) - seq_len):
            if df['TARGET_PERCENTAGE'].iloc[i+seq_len-1] == 'HOLD':
                if amostragem[a] == 1:    
                    x, y = self.next_window(i, seq_len, df)
                    data_x.append(x)
                    data_y.append(y)
                a+=1
            else:
                x, y = self.next_window(i, seq_len, df)
                data_x.append(x)
                data_y.append(y)
    
        return np.array(data_x), np.array(data_y)
    
    # Padronização dos dados
    def normalize(self, data):
    
        scaler_x = StandardScaler()
    
        x = data.iloc[:, :-2]
        x = scaler_x.fit_transform(x) 
    
        y = data.iloc[:,-1]
        y_list = np.array(y.to_list())
    
        return x, y_list
    
    # Movimentação da janela: normalizando janela a janela
    def next_window(self, i, seq_len, df):
        df = df[i:i+seq_len]
        scaler_x = StandardScaler()
        
        x = df.iloc[:, :-2]
        # xxx = np.array(x)
        x = scaler_x.fit_transform(x) 
        
        y = df.iloc[-1,-1]
        # y_list = np.array(y.to_list())
        
        return x, y
    
    
    
    
    
    
    
    
    
    
 