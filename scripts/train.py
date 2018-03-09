import sys
import numpy as np
import subprocess
from keras.preprocessing import sequence
from keras.utils import np_utils
import sklearn
from sklearn.metrics import f1_score
from preprocessing import tokenizer
import itertools
from vec_util import dataset2array,build_embedding_matrix,build_voc_map
from model_1d import *
#nohup python train.py ../data/original/train_full.tsv ../data/original/test_13.tsv google >./plotting/google-ada1-50epo.out 2>&1&
#python train.py ../data/original/train_full.tsv ../data/original/test_13.tsv train
if(len(sys.argv) < 4):
	message = "Usage: " + sys.argv[0] + " <raw_train_file> <raw_val_file> <vec-to-use>\n" + \
			"train/val file format:\n" + \
			"id1<TAB>id2<TAB>pred_class<TAB>tweet_text\n" +\
			"vec-to-use:\n" +\
			"train|google|glove"
	print(message)
	sys.exit(0)
elif sys.argv[3] not in ['train','random','glove']:
	print("<vec-to-use> has to be defined in one of train|random|glove!")
	sys.exit(0)

voc_file = '../data/volcabulary'
train_tokenized_file='../data/train_tokenized.tsv'
val_tokenized_file='../data/val_tokenized.tsv'
train_input_file='../data/train_input.tsv'
val_input_file='../data/val_input.tsv'

#preprocessing
tokenizer(sys.argv[1], train_tokenized_file, train_input_file)
tokenizer(sys.argv[2], val_tokenized_file, val_input_file)
if sys.argv[3] == 'glove':	
	W = build_embedding_matrix('../data/glove_vec')
	model_file = '../model/model-glove.h5'
# elif sys.argv[3] == 'google':
# 	W = build_embedding_matrix('../data/google_vec')
# 	model_file = './model-google.h5'
elif sys.argv[3] == 'train':
	subprocess.call('bash vec_gen.sh',shell=True)
	W = build_embedding_matrix('../data/embeddings_sorted')
	model_file = '../model/model-train.h5'
else:
	W = np.random.uniform(-0.25,0.25,(4282,100))
	model_file = '../model/model-random.h5'

#get train and val set
voc_size = build_voc_map(voc_file)
# print('voc size/n_volc: '+str(voc_size)) #4281
train_set = dataset2array(train_input_file)
val_set =dataset2array(val_input_file)

N = train_set.shape[0] #number of training examples
conv_input_width = W.shape[1] #vector dimension i.e. 100
conv_input_height = train_set.shape[1] - 1 #the length of sentence,removing y label i.e. 75

train_X = np.array(train_set[:,0:-1], dtype=np.int)
train_y = train_set[:,-1]
#0:neg 1:pos 2:neu
train_Y = np.array(np_utils.to_categorical(train_y,nb_classes=3),dtype=np.int)
train_Y_classes = np.where(train_Y == 1)[1]

val_X = np.array(val_set[:,0:-1],dtype=np.int)
val_y = val_set[:,-1]
val_Y = np.array(np_utils.to_categorical(val_y,nb_classes=3),dtype=np.int)

para_range = {
	'n_fm' : range(50,201,50),
	
	'window_size' : [3,4,5],
	'batch_size' : [32,64,128],

	'drop_rate' : [0.2,0.3,0.4,0.5],
	'lr' : [0.01,0.1,1.0],
	'l2_strength': [0.0001,0.001,0.01],	
	'n_epoch' : [10]
}

para_set = {'n_voc':voc_size,
			'max_len':75,
			'n_fm':100,
			'window_size':3,
			'batch_size':32,
			'emb_dim':W.shape[1],
			'drop_rate':0.3,
			'lr':1.0,
			'l2_strength':0.001,
			'n_epoch':50,
			'W':W
			}

def random_search():
	"""random search for parameter optimization"""
	para_set['n_epoch'] = 10
	para_product = list(itertools.product(para_range['n_fm'],
									para_range['window_size'],
									para_range['batch_size'],
									para_range['drop_rate'],
									para_range['lr'],
									para_range['l2_strength']
									))
	rand_index = np.random.randint(len(para_product),size=10)
	print(rand_index)
	rand_set = []
	for i in list(rand_index):
		rand_set.append(para_product[i])
	print('random parameter set to try:')
	print('n_fm | window_size | batch_size | drop_rate | lr | l2_strength')
	for i in range(len(rand_set)):
		print(rand_set[i])
	f1s = []
	for i in range(len(rand_set)): #20
		print('\nstarting '+str(i+1)+'th set of parameters...')
		print(rand_set[i])
		(para_set['n_fm'], para_set['window_size'], para_set['batch_size'], \
			para_set['drop_rate'], para_set['lr'], para_set['l2_strength']) = rand_set[i]
		keras_model = model_1d_train(para_set, model_file)
		best_f1 = keras_model.fit(train_X, train_Y, val_X, val_Y)
		print('the best F1 score obtained from this model is '+ str(best_f1))
		f1s.append(best_f1)
	best_param = rand_set[f1s.index(np.max(f1s))]
	# print(best_param)
	para_best = {'n_voc':voc_size,
				'max_len':75,
				'n_fm':best_param[0],
				'window_size':best_param[1],
				'batch_size':best_param[2],
				'emb_dim':W.shape[1],
				'drop_rate':best_param[3],
				'lr':best_param[4],
				'l2_strength':best_param[5],
				'n_epoch':50,
				'W':W
				}
	keras_model = model_1d_train(para_best, model_file)
	keras_model.fit(train_X, train_Y, val_X, val_Y)
	
def train_keras():
	keras_model = model_1d_train(para_set, model_file)
	keras_model.fit(train_X, train_Y, val_X, val_Y)

def  main():
	# random_search()
	train_keras()

if __name__ == "__main__":
	main()