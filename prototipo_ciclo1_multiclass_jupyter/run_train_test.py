__author__ = "Audeliano Li"
__version__ = "1.0.0"

import os
import json
import math
import pygame
import numpy as np
import pandas as pd
import csv

from core.create_dataframe import CreateDataFrame
from core.data_processor_train_test import DataLoader
from core.model_train_test import Model
from datetime import datetime

    
def main():
    
    configs = json.load(open('config_modelo.json', 'r'))

    if not os.path.exists(configs['model']['save_dir']):
        os.makedirs(configs['model']['save_dir'])
    
    
    if(configs['dataset']['create_new_dataframe']):
        print('Criando um novo DataFrame')
        datapath = os.path.join('dataset', configs['dataset']['filename'])
        dataframe = pd.read_csv(datapath)   
        
        data = CreateDataFrame(dataframe, configs)
        
        df = data.create_df()
    
    else:
        print('Carregando um DataFrame existente via pickle')
        filepath = 'C:/Users/auW10/Documents/CodigosDoAu/000_Doutorado/dataset/'
        df = pd.read_pickle(filepath+'dataset_with_TI_ciclo2.pkl')

    data = DataLoader(df)  
    
    model = Model()
    # len(data.df_train)-1 -> Porque para o treinamento não é usado a coluna TARGET_WIN
    model.build_model(configs, 
                      configs['dataset']['sequence_length'],
                      len(data.df_train.columns)-2)

    
    if(not (configs['training']['train']) and configs['training']['test']):
        loops = 1
    else:
        loops = configs['training']['loops']

    for loop in range(loops):
        
        print("Loop ", loop+1,"/", loops)
        
        save_dir=configs['model']['save_dir']
        my_model=configs['model']['model_name']
        if not os.path.exists(save_dir+'/%s.h5' % my_model):
            print("Não existe modelo salvo")
            
        else:    
            model.load_model(save_dir+'/%s.h5' % my_model)

        if(configs['training']['train']):
            # Utilizo o tamanho do df_train subtraio pelo tamanho da 
            # seq_length ~ lag e divido tudo pelo tamanho do batch do treinamento
            steps_per_epoch = math.ceil((len(data.df_train) - configs['dataset']['sequence_length']) / configs['training']['batch_size'])  
            steps_per_epoch_valid = math.ceil((len(data.df_valid) - configs['dataset']['sequence_length']) / configs['training']['batch_size']) 
            
            print('Criando data_gen')
            data_gen = data.generate_train_batch(seq_len=configs['dataset']['sequence_length'],
                                                 batch_size=configs['training']['batch_size'])
            
            print('Criando data_valid_gen')
            data_valid_gen = data.generate_valid_batch(seq_len=configs['dataset']['sequence_length'],
                                                 batch_size=configs['training']['batch_size'])
            
            # print(list(data_gen))
            
            model.train_generator(  configs,
                                    data_gen, 
                                    data_valid_gen,
                                    epochs=configs['training']['epochs'],
                                    batch_size=configs['training']['batch_size'],
                                    steps_per_epoch=steps_per_epoch,
                                    steps_per_epoch_valid=steps_per_epoch_valid,
                                    save_dir=save_dir,
                                    model_name=my_model, 
                                    loop=loop)
            
            pygame.mixer.init()
            pygame.mixer.music.load("RICOCHET.wav")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy() == True:
                continue
    
   
    if(configs['training']['test']):
        x_test, y_test = data.get_test_data(seq_len=configs['dataset']['sequence_length'])
        
        predictions = model.predict_sequences_multiple( x_test,
                                                        configs['dataset']['sequence_length'],
                                                        1)
        
        print(predictions)
        
        y_test = np.ravel(y_test)
        predictions = np.ravel(predictions)
        
        print('y_test: \n', y_test)
        print('predictions: \n', predictions)
        
        print('y_test.size: ', y_test.size)
        print('predictions.size: ', predictions.size)
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        mae = mean_absolute_error(y_test, predictions)
        print('MAE: ', mae)
        
        mse = mean_squared_error(y_test, predictions, squared=True)
        print('MSE: ', mse)
        
        rmse = mean_squared_error(y_test, predictions, squared=False)
        print('RMSE: ', rmse)
        
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%Y%m%d_%H-%M-%S")
        
        csv_name = ('previsoes/%s_%s_%s.csv' % ('previsoes_reg', dt_string, my_model))
                
        with open(csv_name, 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=';')
            for c in range(len(predictions)):
                spamwriter.writerow([y_test[c]] + [predictions[c]])
            spamwriter.writerow(['MAE: '] + [mae])
            spamwriter.writerow(['MSE: '] + [mse])
            spamwriter.writerow(['RMSE: '] + [rmse])
        
        predictions_cat = []
        for p in predictions:
            if p < 0.5:
                # BUY
                predictions_cat.append(0)
            else:
                # SELL
                predictions_cat.append(1)
        
        from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, confusion_matrix
        recall_macro = recall_score(y_test, predictions_cat, average='macro')
        print('recall_macro: ', recall_macro)
        
        recall_micro = recall_score(y_test, predictions_cat, average='micro')
        print('recall_micro: ', recall_micro)
        
        recall_weighted = recall_score(y_test, predictions_cat, average='weighted')
        print('recall_weighted: ', recall_weighted)
        
        precision = precision_score(y_test, predictions_cat)
        print('precision: ', precision)
        
        accuracy = accuracy_score(y_test, predictions_cat)
        print('accuracy: ', accuracy)
        
        f1_s = f1_score(y_test, predictions_cat)
        print('f1_score: ', f1_s)
        
        confusion_matrix = confusion_matrix(y_test, predictions_cat)
        print('confusion_matrix: \n', confusion_matrix)

        csv_name = 'previsoes/%s_%s_%s.csv' % ('previsoes_cat', dt_string, my_model)
                
        with open(csv_name, 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=';')
            for c in range(len(predictions_cat)):
                spamwriter.writerow([y_test[c]] + [predictions_cat[c]])
            spamwriter.writerow(['recall_macro: '] + [recall_macro])
            spamwriter.writerow(['recall_micro: '] + [recall_micro])
            spamwriter.writerow(['recall_weighted: '] + [recall_weighted])   
            spamwriter.writerow(['precision: '] + [precision])              
            spamwriter.writerow(['accuracy: '] + [accuracy]) 
            spamwriter.writerow(['f1_score: '] + [f1_s]) 
                
    pygame.mixer.init()
    pygame.mixer.music.load("SkiiiiinTheGame.wav")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy() == True:
        continue

if __name__ == '__main__':
    main()
