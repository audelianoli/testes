__author__ = "Audeliano Li"
__date__ = "2021/03/13"
__version__ = "2.0.0"

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import time

from numpy import newaxis
from core.utils import Timer
from datetime import datetime
from keras import Input
from keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

class Model():
    """A class for an building and inferencing an lstm model"""

    def __init__(self):
        self.model = Sequential()

    def evaluate_model(self, x, y):
        loss, acc = self.model = Sequential.evaluate(x,y)
        print("Restored model, accuracy: {:5.2f}%".format(100*acc))
        
    def load_model(self, filepath):
        print('[Model] Loading model from file %s' % filepath)
        self.model = load_model(filepath)

    def build_model(self, configs, timesteps, dim):
        
        timer = Timer()
        timer.start()

        for layer in configs['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else timesteps
            # input_dim = layer['input_dim'] if 'input_dim' in layer else dim
            input_dim = dim

            if layer['type'] == 'input_dim':
                self.model.add(Input(shape=(input_dim,)))
            if layer['type'] == 'dense':
                self.model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'simplernn':
                self.model.add(SimpleRNN(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'lstm':
                self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'gru':
                self.model.add(GRU(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))

        self.model.compile(  loss=configs['model']['loss'], 
                             optimizer=configs['model']['optimizer'], 
                             metrics=['accuracy'])

        print('[Model] Model Compiled')
        print(self.model.summary())
        timer.stop()

    def encode_categ_data(self, data):
        labelencoder = LabelEncoder()
        x = labelencoder.fit_transform(data)
        data_encoded = to_categorical(x)
        
        return data_encoded

    def train(self, configs, x_train, y_train, x_valid, y_valid, batch_size, save_dir, model_name, loop):
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%Y%m%d_%H-%M-%S")
        
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        
        # save_fname = os.path.join(save_dir, '%s.h5' % model_name)
        save_fname = os.path.join('{}/{}_{}.h5'.format(save_dir, model_name, dt_string))
        callbacks = [ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)]
        
        y_train_enc = self.encode_categ_data(y_train)
        y_valid_enc = self.encode_categ_data(y_valid)
        H = self.model.fit(    x_train, y_train_enc, 
                               validation_data=(x_valid, y_valid_enc), 
                               epochs=configs['training']['epochs'],
                               callbacks=callbacks, workers=1 )
            
        self.model.save(save_fname)
        print('[Model] Training Completed.')
        print('Model saved as %s' % save_fname)
        
        self.train_monitor(model_name, H)

        timer.stop()
        
        return self.model
    
    def train_monitor(self, my_model, H):       
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%Y%m%d_%H-%M-%S")

        # list all data in history       
        print(H.history.keys())
        # summarize history for accuracy
        plt.figure()
        plt.plot(H.history['accuracy'])
        plt.plot(H.history['val_accuracy'])
        plt.title('model accuracy {}'.format(my_model))
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['accuracy', 'val_accuracy'], loc='upper left')
        fig_path = 'figures/%s_train_accuracy_history_%s.png' % (dt_string, my_model)
        plt.savefig(fig_path, dpi=600)     
        # plt.show()
        
        # summarize history for loss
        plt.figure()
        plt.plot(H.history['loss'])
        plt.plot(H.history['val_loss'])
        plt.title('model loss {}'.format(my_model))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['loss', 'val_loss'], loc='upper left')
        fig_path = 'figures/%s_train_loss_history_%s.png' % (dt_string, my_model)
        plt.savefig(fig_path, dpi=600) 
        # plt.show()

        pd.DataFrame.from_dict(H.history).to_csv('history/%s_%s_%s.csv' % (dt_string, 'history_pd', my_model) ,index=False)

        csv_name = 'history/%s_%s_%s.csv' % (dt_string, 'history_csv', my_model)
                
        with open(csv_name, 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=';')
            for c in range(len(H.history['accuracy'])):
                spamwriter.writerow([H.history['accuracy'][c]] + [H.history['val_accuracy'][c]] + 
                                    [H.history['loss'][c]] + [H.history['val_loss'][c]])
 
        
