import numpy as np
from keras import backend as K
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten,Reshape,Lambda
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution2D,MaxPooling2D,Convolution1D,MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adadelta,SGD
from keras.constraints import UnitNorm
from keras.regularizers import l2
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.metrics import f1_score
from vec_util import build_embedding_matrix

W = build_embedding_matrix('../data/glove_vec')

class model_1d(object):

	def __init__(self,
					n_voc=4281, max_len=75, n_fm=100,
					window_size=5, batch_size=32, emb_dim=100, 
					drop_rate=0.5, lr=0.1, l2_strength=0.0001,
					n_epoch=50,model_file='./model.h5',W=W):
					#W=np.zeros((4282,100))):
		self.n_voc=n_voc
		self.max_len = max_len
		self.emb_dim = emb_dim
		self.hidden_dims = n_fm

		self.n_fm = n_fm
		self.window_size = window_size
		self.batch_size = batch_size
		self.drop_rate = drop_rate
		self.lr = lr
		self.l2_strength = l2_strength
		self.n_epoch = n_epoch
		self.model_file = model_file
		self.W = W

		self.model = Sequential()
		self.model.add(Embedding(input_dim=self.n_voc+1,
							output_dim=self.emb_dim,
							input_length=self.max_len,
							weights=[self.W]))

		self.model.add(Convolution1D(nb_filter=self.n_fm,
								filter_length=self.window_size,
								border_mode='valid',
								activation='relu',
								W_regularizer=l2(self.l2_strength),
								subsample_length=1))
		self.model.add(MaxPooling1D(pool_length=self.model.output_shape[1]))
		self.model.add(Flatten())		
		self.model.add(Dropout(self.drop_rate))
		self.model.add(Dense(self.hidden_dims,activation='relu'))
		self.model.add(Dense(3,activation='softmax'))
		opt = Adadelta(lr=self.lr, rho=0.95, epsilon=1e-6)
		self.model.compile(loss='categorical_crossentropy', 
		              optimizer=opt)
		# self.model.summary()

	def get_params(self, deep = True):
		return {
			'n_voc' : self.n_voc,
			'max_len': self.max_len,
			'n_fm': self.n_fm,
			'window_size': self.window_size,
			'batch_size':self.batch_size,
			'emb_dim':self.emb_dim,
			'drop_rate':self.drop_rate,
			'lr': self.lr,
			'l2_strength':self.l2_strength,
			'n_epoch': self.n_epoch,
			'model_file': self.model_file,
			'W':self.W
		}

	def set_params(self, **parameters):
		for parameter, value in parameters.items():
			self.setattr(parameter, value)
		return self

	def setattr(self, parameter, value):
		name = parameter
		super(model_1d, self).__setattr__(name, value)

	def fit(self,X, y):
		#early stopping based on validation accuracy
		earlyStopping = EarlyStopping(monitor='loss',patience=5,verbose=1,mode='auto')
		callbacks_list = [earlyStopping]#,checkpoint]
		self.model.fit(X, y,batch_size=self.batch_size,
					nb_epoch=self.n_epoch,verbose=1,
					callbacks=callbacks_list,
					shuffle=True)
		return self
		

	def predict(self,X):
		y_pred = self.model.predict_classes(X,batch_size=self.batch_size,verbose=1)
		self.prediction = np.zeros((len(y_pred),3))
		i = 0 
		for i in range(len(y_pred)):
			self.prediction[i,y_pred[i]] = 1
		# print('predictions:')
		# print(self.prediction[0:50])
		return self.prediction


class model_1d_train(model_1d):
	def __init__(self,para_dict, model_file):		
		super(model_1d_train, self).__init__(n_voc=para_dict['n_voc'],
										max_len = para_dict['max_len'],
										n_fm = para_dict['n_fm'],
										window_size = para_dict['window_size'],
										batch_size = para_dict['batch_size'],
										emb_dim = para_dict['emb_dim'],
										drop_rate = para_dict['drop_rate'],
										lr = para_dict['lr'],
										l2_strength = para_dict['l2_strength'],
										n_epoch = para_dict['n_epoch'],
										model_file = model_file,
										W = para_dict['W'])

	def fit(self,train_X,train_Y,val_X,val_Y):	
		val_f1s = []
		f1_best = 0 
		best_epoch = 0
		#early stopping according to F1 score on validation set
		for i in range(self.n_epoch):
			print('training model...')
			self.model.fit(train_X, train_Y,batch_size=self.batch_size,nb_epoch=1,shuffle=True,verbose=0)
			print('validating...')
			val_classes = np.where(val_Y == 1)[1]
			prediction = self.model.predict_classes(val_X,batch_size=self.batch_size,verbose=0)
			val_f1 = f1_score(val_classes, prediction,average='weighted')
			val_f1_tmp = f1_score(val_classes, prediction, average=None)
			print('tmp validation F1')
			print(val_f1_tmp)
			if val_f1 > f1_best:
				f1_best = val_f1
				best_epoch = i
				self.model.save(self.model_file)
			print('\nEpoch {}: validation f1 = {:.3%}'.format(i, val_f1))
			val_f1s.append(val_f1)
			if i - best_epoch > 10:
				print('No more improving on validation F1. Stopping...')
				break
			i += 1
		return f1_best
