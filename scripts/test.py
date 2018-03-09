import subprocess
import sys
import numpy as np 
from keras.models import load_model
from keras.utils import np_utils
from sklearn.metrics import f1_score
from vec_util import dataset2array,output_test, output_test_15
from preprocessing import tokenizer_test

#python test.py ../data/test_merged_15.tsv ./model-glove.h5 ./score-semeval2015-task10-subtaskB.pl
#python test.py ../data/test_merged.tsv ../model/model-glove.h5 ./score-semeval2014-task9-subtaskB.pl
if(len(sys.argv) < 3):
	message = "Usage: " + sys.argv[0] + " <raw_test_file> <model_file> <scoring script>\n" + \
			"test file format:\n" + \
			"id1<TAB>id2<TAB>unknwn<TAB>tweet_text\n"
	print(message)
	sys.exit(0)

tokenized_file = '../data/test_tokenized.tsv'
input_file = '../data/test_input.tsv'
output_file = '../data/test_output.tsv'
tokenizer_test(sys.argv[1], tokenized_file, input_file)

test_set = dataset2array(input_file)
# print(test_set.shape)
model_file = sys.argv[2]
model = load_model(model_file)

test_X = np.array(test_set[:,0:-1],dtype=np.int)
test_y = test_set[:,-1]
test_Y = np.array(np_utils.to_categorical(test_y,nb_classes=3), dtype=np.int)
Y_classes = np.where(test_Y == 1)[1]

prediction = model.predict_classes(test_X,batch_size=8,verbose=1)
# output_test(tokenized_file, output_file, prediction)

output_test(tokenized_file, output_file, prediction)

# subprocess.call("perl ./score-semeval2014-task9-subtaskB.pl "+output_file,shell=True)
subprocess.call("perl "+ sys.argv[3] + " " + output_file,shell=True)
test_f1 = f1_score(Y_classes, prediction,average='weighted')
print('overall test f1: '+str(test_f1))