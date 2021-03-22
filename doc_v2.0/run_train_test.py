__author__ = "Audeliano Li"
__date__ = "2021/03/13"
__version__ = "2.0.0"

import os
import json
import math
import pygame
import numpy as np
import pandas as pd
import csv
import seaborn as sns
#from pylab import savefig
import matplotlib.pyplot as plt
import sqlite3

import lightgbm as lgbm
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score as accs
from sklearn.metrics import confusion_matrix as cm

from core.create_dataframe import CreateDataFrame
from core.data_processor_train_test import DataLoader
from core.model_train_test import Model
from datetime import datetime

def model_ml(model_type, x_train, y_train, n_estimators):

    if(model_type == 'lgbm'): 
        print('\n\n--*--*--*--Iniciando Treinamento do Modelo LightGBM')
        model_trained = lgbm.LGBMClassifier( n_jobs=-1, 
                                             random_state=0, 
                                             n_estimators=n_estimators, 
                                             learning_rate=0.001, 
                                             num_leaves=2**6,
                                             subsample=0.9, 
                                             subsample_freq=1, 
                                             colsample_bytree=1.)
        model_trained.fit(x_train, y_train)
        
    if(model_type == 'random_forest'):
        print('\n\n--*--*--*--Iniciando Treinamento do Modelo Random Forest')
        model_trained = RandomForestClassifier(n_jobs=-1, 
                                               random_state=0, 
                                               n_estimators=n_estimators)
        model_trained.fit(x_train, y_train)

    return model_trained, model_type

def metricas_ml(model_trained, model_type, x_train, y_train, x_valid, y_valid, x_test, y_test):
    predict_model = model_trained.predict(x_test)
    
    accuracy_score = accs(y_test, predict_model)
    print('Accuracy Score {0}: {1:.4f}'.format(model_type, accuracy_score))
    
    cm_data = cm(y_test, predict_model, labels=["BUY", "HOLD", "SELL"]) 
    # visualize confusion matrix with seaborn heatmap
    cm_matrix = pd.DataFrame(   data=cm_data,
                                columns=['BUY_P', 'HOLD_P', 'SELL_P'], 
                                index=['BUY_R', 'HOLD_R', 'SELL_R'])
    
    # heat = sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
    
    weightedFScore = weightedFScoreFn(cm_data)[0] 
    print('Weighted FScore {0}: {1:.4f}'.format(model_type, weightedFScore))
    
    y_pred_train = model_trained.predict(x_train)
    print('Train Versus Predict Accuracy Score {0}: {1:.4f}'.format(model_type, accs(y_train, y_pred_train)))
    
    print('\nPrint the scores on training, valid and test set')

    print('Training set score {0}: {1:.4f}'.format(model_type, model_trained.score(x_train, y_train)))
    print('Validation set score {0}: {1:.4f}'.format(model_type, model_trained.score(x_valid, y_valid)))
    print('Test set score {0}: {1:.4f}'.format(model_type, model_trained.score(x_test, y_test)))

    return model_type, accuracy_score, weightedFScore, cm_matrix   

def metricas_deepl(model_trained, model, my_model, x_test, y_test):
    # new instances where we do not know the answer
    y_pred = model_trained.predict(x_test) 
    
    pred = []
    for i in y_pred:
        pred.append(np.argmax(i))
    pred_array = np.array(pred)       
    
    real = []
    y_test_enc = model.encode_categ_data(y_test)
    for i in y_test_enc:
        real.append(np.argmax(i))
    real_array = np.array(real)    
            
    accuracy_score_deep = accs(real_array, pred_array)
    print('Accuracy Score {0}: {1:.4f}'.format(my_model, accuracy_score_deep))
    
    cm_data = cm(real_array, pred_array)  
    # visualize confusion matrix with seaborn heatmap
    cm_matrix = pd.DataFrame(   data=cm_data,
                                columns=['BUY_P', 'HOLD_P', 'SELL_P'], 
                                index=['BUY_R', 'HOLD_R', 'SELL_R'])
    
    # heat = sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
    
    weightedFScore = weightedFScoreFn(cm_data)[0] 
    print('Weighted FScore {0}: {1:.4f}'.format(my_model, weightedFScore))

    return my_model, accuracy_score_deep, weightedFScore, cm_matrix   

def weightedFScoreFn(confusion_matrix):
    beta_1 = math.pow(0.5, 2)
    beta_2 = math.pow(0.25, 2)
    beta_3 = math.pow(0.125, 2)
    
    num_tu = confusion_matrix[[0],[0]]
    num_tf = confusion_matrix[[1],[1]]
    num_td = confusion_matrix[[2],[2]]
    
    wrongF_trueU = confusion_matrix[[0],[1]]
    wrongD_trueU = confusion_matrix[[0],[2]]
    
    wrongU_trueF = confusion_matrix[[1],[0]]
    wrongD_trueF = confusion_matrix[[1],[2]]
  
    wrongU_trueD = confusion_matrix[[2],[0]]
    wrongF_trueD = confusion_matrix[[2],[1]]
    
    num_tp = num_tu + num_td + (beta_3 * num_tf)
    erro_tipo_1 = wrongU_trueD + wrongD_trueU
    erro_tipo_2 = wrongU_trueF + wrongD_trueF
    erro_tipo_3 = wrongF_trueU + wrongF_trueD
    
    num = (1 + beta_1 + beta_2) * num_tp
    den = num + erro_tipo_1 + (beta_1 * erro_tipo_2) + (beta_2 * erro_tipo_3)
    
    wFScore = num / den
    
    return wFScore

def dbSQLite(dt_string, my_model, dataframe_pkl_name, accuracy, weightedFScore, n_estimators, janela_lag, epochs, loops):
    # Criar conexão com o Banco de Dados
    # Caso não exista o arquivo, este é criado
    con = sqlite3.connect('models_performance.db')
    
    # Cursos para percorrer todos os registros em um conjunto de dados
    cur = con.cursor()
    
    # Instrução SQL
    sql_create = 'create table if not exists models '+\
                '(Datetime TEXT PRIMARY KEY, '+\
                'Nome_Do_Modelo TEXT, '+\
                'dataframe_pkl_name TEXT, '+\
                'Accuracy_Score FLOAT, '+\
                'Weighted_FScore FLOAT, '+\
                'n_estimators INTEGER, '+\
                'sequence_length INTEGER, '+\
                'epochs INTEGER, '+\
                'loops INTEGER)'
    
    # Exutar a instrução SQL no cursor
    cur.execute(sql_create)
    
    # Criando uma sentença SQL para inserir registros
    sql_insert = 'insert into models values (?,?,?,?,?,?,?,?,?)'
    
    # Dados a serem inseridos
    recset = (dt_string, my_model, dataframe_pkl_name, accuracy, weightedFScore, n_estimators, janela_lag, epochs, loops)
    
    # Inserir registros no banco de dados
    cur.execute(sql_insert, recset)
    
    #Gravar a Transção
    con.commit()
    
    return 0

def main():
    
    configs = json.load(open('configs/config.json', 'r'))

    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")
    if not os.path.exists("figures"):
        os.makedirs("figures")
    if not os.path.exists("history"):
        os.makedirs("history")
    
    if(configs['dataset']['create_new_dataframe']):
        print('Criando um novo DataFrame')
        datapath = os.path.join('dataset', configs['dataset']['filename'])
        dataframe = pd.read_csv(datapath, sep=configs['dataset']['separador'])   
        
        data = CreateDataFrame(dataframe, configs)
        
        df = data.create_df()
    
    else:
        print('Carregando um DataFrame existente via pickle')
        filepath = configs['dataset']['filepath']
        df = pd.read_pickle(filepath+configs['dataset']['dataframe_pkl_name'])

    if(not configs['training']['train'] and not configs['training']['test']):
        quit()

    data = DataLoader(configs, df)  
    
    x_train, y_train = data.get_train()
    print('x_train: ', x_train.shape,' | y_train: ', y_train.shape)
    x_valid, y_valid = data.get_valid()
    print('x_valid: ', x_valid.shape,' | y_valid: ', y_valid.shape)
    x_test, y_test = data.get_test()
    print('x_test: ', x_test.shape,' | y_test: ', y_test.shape)
    
    x_train_sw, y_train_sw = data.get_train_sw()
    print('x_train_sw: ', x_train_sw.shape,' | y_train_sw: ', y_train_sw.shape)
    x_valid_sw, y_valid_sw = data.get_valid_sw()
    print('x_valid_sw: ', x_valid_sw.shape,' | y_valid_sw: ', y_valid_sw.shape)
    x_test_sw, y_test_sw = data.get_test_sw()
    print('x_test_sw: ', x_test_sw.shape,' | y_test_sw: ', y_test_sw.shape)
    
    
    
    # MODELO LightGBM
    model_lgbm, model_type_lgbm = model_ml('lgbm', x_train, y_train, configs['training']['n_estimators'])
    a, b, c, d = metricas_ml(   model_lgbm, 
                                model_type_lgbm,
                                x_train, y_train, 
                                x_valid, y_valid, 
                                x_test, y_test)

    model_type_lgbm     = a 
    accuracy_score_lgbm = b 
    weightedFScore_lgbm = c 
    cm_matrix_lgbm      = d
    
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%Y%m%d_%H-%M-%S")
    
    if not os.path.exists("metricas"):
        os.makedirs("metricas")
    csv_name = ('metricas/%s_%s_%s.csv' % (dt_string, 'metricas', model_type_lgbm))
            
    with open(csv_name, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';')

        spamwriter.writerow(['model_type_lgbm: '] + [model_type_lgbm])
        spamwriter.writerow(['accuracy_score_lgbm: '] + [accuracy_score_lgbm])
        spamwriter.writerow(['weightedFScore_lgbm: '] + [weightedFScore_lgbm])
    
    plt.figure()
    plt.title('Heatmap {0} | acc: {1:.2f} | wfs: {2:.2f}'.format(model_type_lgbm, accuracy_score_lgbm, weightedFScore_lgbm))
    heat = sns.heatmap(cm_matrix_lgbm, annot=True, fmt='d', cmap='YlGnBu')
    plt.savefig('figures/{} Heatmap {}'.format(dt_string, model_type_lgbm), dpi=600)
    
    dbSQLite(dt_string, model_type_lgbm, 
             configs['dataset']['dataframe_pkl_name'], 
             accuracy_score_lgbm, weightedFScore_lgbm, 
             configs['training']['n_estimators'], 
             'NULL', 'NULL', 'NULL')
    
    # MODELO Random Forest
    model_randomF, model_type_randomF = model_ml('random_forest', x_train, y_train, configs['training']['n_estimators'])
    a, b, c, d  = metricas_ml(  model_randomF, 
                                model_type_randomF,
                                x_train, y_train, 
                                x_valid, y_valid, 
                                x_test, y_test)
                            
    model_type_randomF     = a
    accuracy_score_randomF = b
    weightedFScore_randomF = c
    cm_matrix_randomF      = d                  

    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%Y%m%d_%H-%M-%S")
    
    if not os.path.exists("metricas"):
        os.makedirs("metricas")
    csv_name = ('metricas/%s_%s_%s.csv' % (dt_string, 'metricas', model_type_randomF))
            
    with open(csv_name, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';')

        spamwriter.writerow(['model_type_randomF: '] + [model_type_randomF])
        spamwriter.writerow(['accuracy_score_randomF: '] + [accuracy_score_randomF])
        spamwriter.writerow(['weightedFScore_randomF: '] + [weightedFScore_randomF])    
    
    plt.figure()
    plt.title('Heatmap {0} | acc: {1:.2f} | wfs: {2:.2f}'.format(model_type_randomF, accuracy_score_randomF, weightedFScore_randomF))
    heat = sns.heatmap(cm_matrix_randomF, annot=True, fmt='d', cmap='YlGnBu')
    plt.savefig('figures/{} Heatmap {}'.format(dt_string, model_type_randomF), dpi=600)
    
    dbSQLite(dt_string, model_type_randomF, 
             configs['dataset']['dataframe_pkl_name'], 
             accuracy_score_randomF, weightedFScore_randomF, 
             configs['training']['n_estimators'], 
             'NULL', 'NULL', 'NULL')
    
    models_list = []
    model_01 = json.load(open('configs/modelo_01.json', 'r'))
    models_list.append(model_01)
    model_02 = json.load(open('configs/modelo_02.json', 'r'))
    models_list.append(model_02)
    model_03 = json.load(open('configs/modelo_03_qiu2020novel.json', 'r'))
    models_list.append(model_03)
    model_04 = json.load(open('configs/modelo_04_ijcnn.json', 'r'))
    models_list.append(model_04)
    model_05 = json.load(open('configs/modelo_05.json', 'r'))
    models_list.append(model_05)
    model_06 = json.load(open('configs/modelo_06.json', 'r'))
    models_list.append(model_06)
    
    for num_deep_models in range(len(models_list)):
        
        for janela_lag in range(configs['dataset']['sequence_length'], configs['dataset']['max_sequence_length']+1 , 5):
        
            model = Model()
            dim = x_train.shape[1]
            batch_size=configs['training']['batch_size']
            # timesteps = configs['dataset']['sequence_length']
            timesteps = janela_lag
            model.build_model(models_list[num_deep_models], 
                              timesteps, dim)
            
            if(not (configs['training']['train']) and configs['training']['test']):
                loops = 1
            else:
                loops = configs['training']['loops']
        
            for loop in range(loops):
                
                print("Loop ", loop+1,"/", loops)
                
                save_dir=models_list[num_deep_models]['model']['save_dir']
                my_model=models_list[num_deep_models]['model']['model_name']
                if not os.path.exists(save_dir+'/%s.h5' % my_model):
                    print("Nao existe modelo salvo")
                    
                else:    
                    model.load_model(save_dir+'/%s.h5' % my_model)
        
                if(configs['training']['train']):
                    
                    if(not models_list[num_deep_models]['model']['use_lstm']):
                        model_trained = model.train(    configs,
                                                        x_train, y_train, 
                                                        x_valid, y_valid,
                                                        batch_size=batch_size,
                                                        save_dir=save_dir,
                                                        model_name=my_model, 
                                                        loop=loop)
                    else:
                       model_trained = model.train(     configs,
                                                        x_train_sw, y_train_sw, 
                                                        x_valid_sw, y_valid_sw,
                                                        batch_size=batch_size,
                                                        save_dir=save_dir,
                                                        model_name=my_model, 
                                                        loop=loop)
                    '''
                    pygame.mixer.init()
                    pygame.mixer.music.load("RICOCHET.wav")
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy() == True:
                        continue
                    '''
            if(configs['training']['test']):
                if(not models_list[num_deep_models]['model']['use_lstm']):
                    a, b, c, d = metricas_deepl(model_trained, model, my_model, x_test, y_test)
                else:
                    a, b, c, d = metricas_deepl(model_trained, model, my_model, x_test_sw, y_test_sw)
                
                my_model            = a
                accuracy_score_deep = b
                weightedFScore_deep = c
                cm_matrix_deep      = d      
                
                now = datetime.now()
                # dd/mm/YY H:M:S
                dt_string = now.strftime("%Y%m%d_%H-%M-%S")
                
                if not os.path.exists("metricas"):
                    os.makedirs("metricas")
                csv_name = ('metricas/%s_%s_%s.csv' % (dt_string, 'metricas', my_model))
                        
                with open(csv_name, 'w', newline='') as csvfile:
                    spamwriter = csv.writer(csvfile, delimiter=';')
        
                    spamwriter.writerow(['my_model: '] + [my_model])
                    spamwriter.writerow(['accuracy_score_deep: '] + [accuracy_score_deep])
                    spamwriter.writerow(['weightedFScore_deep: '] + [weightedFScore_deep])
                
                plt.figure()
                plt.title('Heatmap {0} | acc: {1:.2f} | wfs: {2:.2f} | seq_len: {3}'.format(my_model, accuracy_score_deep, weightedFScore_deep, janela_lag))
                heat = sns.heatmap(cm_matrix_deep, annot=True, fmt='d', cmap='YlGnBu')
                plt.savefig('figures/{} Heatmap {}'.format(dt_string, my_model), dpi=600)
                
                dbSQLite(dt_string, my_model, 
                         configs['dataset']['dataframe_pkl_name'], 
                         accuracy_score_randomF, weightedFScore_deep, 
                         'NULL', janela_lag, 
                         configs['training']['epochs'], 
                         loop+1)
                
            del model
            
    pygame.mixer.init()
    pygame.mixer.music.load("SkiiiiinTheGame.wav")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy() == True:
        continue
        
if __name__ == '__main__':
    main()
