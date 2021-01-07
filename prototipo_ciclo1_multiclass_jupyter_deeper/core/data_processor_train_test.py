import numpy as np
import time

from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

class DataLoader():
    """A class for loading and transforming data for the lstm model"""

    def __init__(self, dataframe):

        self.dataframe = dataframe
        
        print('self.dataframe: \n', self.dataframe)
        
        df2 = self.dataframe.drop(['TARGET_WIN', 
                                   'TARGET_WIN_CAT3', 
                                   'TARGET_WIN_CAT3_LABEL'], axis=1)
        
        df_train = df2[(df2['DATETIME'] < '2018-01-01')]        
        df_valid = df2[(df2['DATETIME'] >= '2018-01-01') & (df2['DATETIME'] < '2019-01-01')]     
        df_test  = df2[(df2['DATETIME'] >= '2019-01-01')]
        
        df_train = df_train.iloc[-1000:, :]
        df_valid = df_valid.iloc[:250, :]
        df_test = df_test.iloc[:23, :]
        
        print('\n\nProporção Train: {:.1%} | Validation: {:.1%} | Test: {:.1%}'.format(
                                                    df_train['CLOSE'].count()/df2['CLOSE'].count(),
                                                    df_valid['CLOSE'].count()/df2['CLOSE'].count(),
                                                    df_test['CLOSE'].count()/df2['CLOSE'].count()))
        
        
        df_train.drop(['DATETIME'], axis=1, inplace=True)
        df_valid.drop(['DATETIME'], axis=1, inplace=True)
        df_test.drop(['DATETIME'], axis=1, inplace=True)
        
        self.df_train = df_train
        self.df_valid = df_valid
        self.df_test = df_test
        
        print(df_train.shape, df_valid.shape, df_test.shape)
        print('\n\n')
        
        print('df_train: \n', df_train)
        print('df_test: \n', df_test)
        
        hold_train = int(df_train['BUY'].value_counts()[0]) - int(df_train['SELL'].value_counts()[1])
        hold_valid = int(df_valid['BUY'].value_counts()[0]) - int(df_valid['SELL'].value_counts()[1])
        # hold_test = int(df_test['BUY'].value_counts()[0]) - int(df_test['SELL'].value_counts()[1])
        print('\nDados Categóricos de Treino -- BUY: %d | SELL: %d | HOLD: %d' % (df_train['BUY'].value_counts()[1], df_train['SELL'].value_counts()[1], hold_train))
        print('\nDados Categóricos de Validação -- BUY: %d | SELL: %d | HOLD: %d' % (df_valid['BUY'].value_counts()[1], df_valid['SELL'].value_counts()[1], hold_valid))
        # print('\nDados Categóricos de Teste -- BUY: %d | SELL: %d | HOLD: %d' % (df_test['BUY'].value_counts()[1], df_test['SELL'].value_counts()[1], hold_test))

    def get_test_data(self, seq_len):
        '''
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        '''
        data_x = []
        data_y = []
    
        for i in range(len(self.df_test.iloc[:]) - seq_len):
            x, y = self._next_window(i, seq_len, self.df_test.iloc[:])
            data_x.append(x)
            data_y.append(y)
        
        return np.array(data_x), np.array(data_y)
    
    def get_valid_data(self, seq_len):
        '''
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        '''
        data_x = []
        data_y = []
    
        for i in range(len(self.df_valid.iloc[:]) - seq_len):
            x, y = self._next_window(i, seq_len, self.df_valid.iloc[:])
            data_x.append(x)
            data_y.append(y)
        
        return np.array(data_x), np.array(data_y)

    def get_train_data(self, seq_len):
        '''
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        '''
        data_x = []
        data_y = []
        for i in range(len(self.df_train.iloc[:]) - seq_len):
            x, y = self._next_window(i, seq_len, self.df_train.iloc[:])
            data_x.append(x)
            data_y.append(y)

        
        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, batch_size):
        '''Yield a generator of training data from filename on given list of cols split for train/test'''
        i = 0
        aux = len(self.df_train.iloc[:]) - seq_len
        while i < aux:
            x_batch = []
            y_batch = []
            
            for b in range(batch_size):
                if i >= aux:
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len, self.df_train.iloc[:])
                
                x_batch.append(x)
                y_batch.append(y)
               
                i += 1

            yield np.array(x_batch), np.array(y_batch)
            
    def generate_valid_batch(self, seq_len, batch_size):
        '''Yield a generator of training data from filename on given list of cols split for train/test'''
        i = 0
        aux = len(self.df_valid.iloc[:]) - seq_len
        while i < aux:
            x_batch = []
            y_batch = []
            
            for b in range(batch_size):
                if i >= aux:
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len, self.df_valid.iloc[:])
                
                x_batch.append(x)
                y_batch.append(y)
               
                i += 1

            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len, window_data):
        '''Generates the next data window from the given index location i'''
        # window = window_data[i:i+seq_len+1]
        # Não preciso mais que seja i+seq_len+1 porque foi criado uma coluna target
        window = window_data[i:i+seq_len]
        
        scaler_x = StandardScaler()
        
        x = window.iloc[:, :-2]
        x = scaler_x.fit_transform(x) 
        
        # y = window.iloc[-1,-1]
        y = window.iloc[-1,-2:]
        y_list = np.array(y.to_list())
        
        # print('\nx: ', x)
        
        # print('\ny_list: ', y_list)
        
        # time.sleep(5)
    
        return x, y_list

    def get_simple_train_data(self):
        
        # x_train = self.df_train.drop(['TARGET_WIN'])       
        x_train = self.df_train.iloc[:,:-2] 
        x_train = self._normalize_simple_data(self, x_train)
        
        # y_train = self.df_train['TARGET_WIN']
        y_train = self.df_train.iloc[:, -2:] 
        
        return np.array(x_train), np.array(y_train)
    
    def get_simple_valid_data(self):
        
        # x_valid = self.df_valid.drop(['TARGET_WIN'])
        x_valid = self.df_valid.iloc[:,:-2]
        x_valid = self._normalize_simple_data(self, x_valid)
        
        # y_valid = self.df_valid['TARGET_WIN']
        y_valid = self.df_valid.iloc[:, -2:] 
        
        return np.array(x_valid), np.array(y_valid)
    
    def get_simple_test_data(self):
        
        # x_test = self.df_test.drop(['TARGET_WIN'])
        x_test = self.df_test.iloc[:,:-2]
        x_test = self._normalize_simple_data(self, x_test)
        
        # y_test = self.df_test['TARGET_WIN']
        y_test = self.df_test.iloc[:, -2:] 
        
        return np.array(x_test), np.array(y_test)
    
    def _normalize_simple_data(self, x_data):
        
        scaler = StandardScaler()
        n = scaler.fit_transform(x_data)
        
        return n

''' 
    def normalise_windows(self, window_data, single_window=False):
        #Normalise window with a base value of zero
        eps = 0.00001
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                
                normalised_col = [((float(p) / (float(window[0, col_i]) + eps) ) - 1) for p in window[:, col_i]]
                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T # reshape and transpose array back into original multidimensional format
            normalised_data.append(normalised_window)
        return np.array(normalised_data)
'''      
