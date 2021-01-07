import os
import numpy as np
import matplotlib.pyplot as plt
import csv
import time

from numpy import newaxis
from core.utils import Timer
from datetime import datetime
from keras.layers import Dense, Dropout, LSTM, GRU
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

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
			input_dim = layer['input_dim'] if 'input_dim' in layer else dim

			if layer['type'] == 'dense':
				self.model.add(Dense(neurons, activation=activation))
			if layer['type'] == 'lstm':
				self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
			if layer['type'] == 'gru':
				self.model.add(GRU(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
			if layer['type'] == 'dropout':
				self.model.add(Dropout(dropout_rate))

		self.model.compile(loss=configs['model']['loss'], 
                             optimizer=configs['model']['optimizer'], 
                             metrics=['accuracy'])

		print('[Model] Model Compiled')
		print(self.model.summary())
		timer.stop()



	def train_generator(self, configs, data_gen, data_valid_gen, epochs, batch_size, steps_per_epoch, steps_per_epoch_valid, save_dir, model_name, loop):
		timer = Timer()
		timer.start()
		print('[Model] Training Started')
		print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))

		save_fname = os.path.join(save_dir, '%s.h5' % model_name)
		callbacks = [ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)]
		H = self.model.fit_generator(data_gen, validation_data=data_valid_gen, 
                                			validation_steps=steps_per_epoch_valid,
                                			steps_per_epoch=steps_per_epoch,
                                			epochs=epochs,
                                			callbacks=callbacks,
                                			workers=1)

		my_model=configs['model']['model_name']
		now = datetime.now()
		# dd/mm/YY H:M:S
		dt_string = now.strftime("%Y%m%d_%H-%M-%S")

		# list all data in history       
		print(H.history.keys())
        # summarize history for accuracy
		plt.plot(H.history['accuracy'])
		plt.plot(H.history['loss'])
		plt.plot(H.history['val_accuracy'])
		plt.plot(H.history['val_loss'])
        
		plt.title('model accuracy and loss')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['accuracy', 'loss', 'val_accuracy', 'val_loss'], loc='upper left')

		fig_path = 'figures/train_history_%s_%s.png' % (dt_string, my_model)
		plt.savefig(fig_path) 
        
		plt.show()
        

        
		import pandas as pd

		pd.DataFrame.from_dict(H.history).to_csv('history/%s_%s_%s.csv' % ('history_pd', dt_string, my_model) ,index=False)

		csv_name = 'history/%s_%s_%s.csv' % ('history_csv', dt_string, my_model)
                
		with open(csv_name, 'w', newline='') as csvfile:
			spamwriter = csv.writer(csvfile, delimiter=';')
			for c in range(len(H.history['accuracy'])):
				spamwriter.writerow([H.history['accuracy'][c]] + [H.history['val_accuracy'][c]] + 
                                    [H.history['loss'][c]] + [H.history['val_loss'][c]])
 

		# summarize history for loss
# 		plt.plot(H.history['loss'])
# 		plt.plot(H.history['val_loss'])
# 		plt.title('model loss')
# 		plt.ylabel('loss')
# 		plt.xlabel('epoch')
# 		plt.legend(['train', 'test'], loc='upper left')
# 		plt.show()

		print('[Model] Training Completed.')
		print('Model saved as %s' % save_fname)
		timer.stop()

	def train(self, x, y, epochs, batch_size, save_dir, model_name):
		timer = Timer()
		timer.start()
		print('[Model] Training Started')
		print('[Model] %s epochs, %s batch size' % (epochs, batch_size))

		save_fname = os.path.join(save_dir, '%s.h5' % model_name)
		callbacks = [
			EarlyStopping(monitor='val_loss', patience=2),
			ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
		]
		self.model.fit(
			x,
			y,
			epochs=epochs,
			batch_size=batch_size,
			callbacks=callbacks
		)
		self.model.save(save_fname)

		print('[Model] Training Completed. Model saved as %s' % save_fname)
		timer.stop()

   	# def predict_point_by_point(self, data):
   	#  	#Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
   	#  	print('[Model] Predicting Point-by-Point...')
   	#  	predicted = self.model.predict(data)
   	#  	predicted = np.reshape(predicted, (predicted.size,))
   	#  	return predicted

	def predict_classes_au(self, data, sequence_length, horizonte):
		#Predict sequence of 50 steps before shifting prediction run forward by 50 steps
		print('[Model] Predicting Sequences Multiple...')
		prediction_seqs = []
		obo = len(data)/horizonte

		for i in range(int(obo)):
			curr_frame = data[i*horizonte]         
# 			print('curr_frame: ', curr_frame)
			predicted = []
			for j in range(horizonte):
				
				previsao = (self.model.predict(curr_frame[newaxis,:,:])[0,0])
# 				print('curr_frame[newaxis,:,:]: ', curr_frame[newaxis,:,:])
				predicted.append(previsao)
# 				print('predicted: ', predicted)
				
				curr_frame = curr_frame[1:]
# 				print('curr_frame = curr_frame[1:]: ', curr_frame[1:])
				curr_frame = np.insert(curr_frame, [sequence_length-1], predicted[-1], axis=0)
# 				print('[sequence_length-1]: ', [sequence_length-1])
# 				print('predicted[-1]: ', predicted[-1])                
# 				print('predicted: ', predicted)                
# 				print('curr_frame: ', curr_frame)     
                
			prediction_seqs.append(predicted)
# 			print('prediction_seqs: ', prediction_seqs)   
# 			time.sleep(120)
        
		return prediction_seqs 

        
	def predict_sequences_multiple(self, data, sequence_length, horizonte):
		#Predict sequence of 50 steps before shifting prediction run forward by 50 steps
		print('[Model] Predicting Sequences Multiple...')
		prediction_seqs = []
		obo = len(data)/horizonte

		for i in range(int(obo)):
			curr_frame = data[i*horizonte]         
# 			print('curr_frame: ', curr_frame)
			predicted = []
			for j in range(horizonte):
				
				previsao = (self.model.predict(curr_frame[newaxis,:,:])[0,0])
# 				print('curr_frame[newaxis,:,:]: ', curr_frame[newaxis,:,:])
				predicted.append(previsao)
# 				print('predicted: ', predicted)
				
				curr_frame = curr_frame[1:]
# 				print('curr_frame = curr_frame[1:]: ', curr_frame[1:])
				curr_frame = np.insert(curr_frame, [sequence_length-1], predicted[-1], axis=0)
# 				print('[sequence_length-1]: ', [sequence_length-1])
# 				print('predicted[-1]: ', predicted[-1])                
# 				print('predicted: ', predicted)                
# 				print('curr_frame: ', curr_frame)     
                
			prediction_seqs.append(predicted)
# 			print('prediction_seqs: ', prediction_seqs)   
# 			time.sleep(120)
        
		return prediction_seqs   


#  	def predict_sequence_full(self, data, window_size):
# 		#Shift the window by 1 new prediction each time, re-run predictions on new window
# 		print('[Model] Predicting Sequences Full...')
# 		curr_frame = data[0]
# 		predicted = []
# 		for i in range(len(data)):
#  			predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
#  			curr_frame = curr_frame[1:]
#  			curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
# 		return predicted
