import numpy as np
from collections import defaultdict
import pickle
import array
import binascii

float_formatter = lambda x: "%.8f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

word2idx_map_file = '../data/word2idx_map.pickle'
idx2word_map_file = '../data/idx2word_map.pickle'

def build_voc_map(volc_file):
	"""
	given a sorted vocabulary file,
	build word2idx and idx2word map from volcabulary file
	index starts from 1
	"""
	word2idx_map = defaultdict(int)
	idx2word_map = defaultdict(str)
	with open(volc_file,'r+') as f:
		for line in f:
			data = line.split('\t')
			word2idx_map[data[1].strip()] = int(data[0])
			idx2word_map[int(data[0])] = data[1].strip()

	f.close()
	with open(word2idx_map_file,'wb') as handle:
		pickle.dump(word2idx_map, handle,protocol=2)
		# pickle.dump(word2idx_map, handle)
	handle.close()
	with open(idx2word_map_file,'wb') as handle:
		pickle.dump(idx2word_map, handle,protocol=2)
		# pickle.dump(idx2word_map, handle)
	handle.close()
	voc_size = len(word2idx_map.items())
	return voc_size

def build_embedding_matrix(em_file):
	"""
	given embedding file,
	return the matrix of word embeddings
	"""
	emb = np.genfromtxt(em_file,
					dtype=float,
					delimiter=' ',
					skip_header=0,
					autostrip=True)
	#for the unk word seen in val/test set
	z = np.random.uniform(-0.25,0.25,emb.shape[1])
	# z = np.zeros(emb.shape[1])
	W = np.row_stack((z,emb))
	# print(W.shape) #(n_volc+1,em_dim) 4281 + 1 i.e. (4282,100)
	return W

def get_vec_from_google(vec_bin_file,volc_file,dist_vec_file,em_dim=300):
	"""
	return: an embedding matrix, same word order as volc_file
	"""

	volcabulary = list(line.split('\t')[1].strip() for line in open(volc_file,'r+'))
	with open(vec_bin_file,'rb') as in_f:#, open('../data/volc_google_tmp','w+') as out_f:
		header = in_f.readline()
		vocab_size, layer1_size = map(int, header.split())
		binary_len = np.dtype('float32').itemsize * layer1_size #4*300  ##
		vecs = np.zeros((len(volcabulary),layer1_size))
		# print('binary len: '+str(binary_len))
		i=0
		# m=0
		for line in range(vocab_size):
			word = []
			while True:
				byte = in_f.read(1) #read one byte
				if byte == ' ':
					word = ''.join(word)
					break
				if byte != '\n':
					word.append(byte)			

			if word in volcabulary:
				# out_f.write(word+'\n')
				data = np.fromstring(in_f.read(binary_len),dtype='float32')
				vecs[volcabulary.index(word)][:] = data
				i+=1
			else:
 				in_f.read(binary_len)
 		print("number of words in the google vec file: "+str(i-1)) #3779
		# print(vecs.shape)

		#adding unknown words
		idxs = np.unique(np.where(vecs==0.0)[0])
		# print('missing unk words: '+str(len(idxs)))
		for i in idxs:
			vecs[i] = np.random.uniform(-0.25,0.25,em_dim)
		###### need to add a last one for unk word
		np.savetxt(dist_vec_file, vecs,fmt='%.9f',delimiter=' ')
		# google_vec = np.fromfile(out_f,dtype=float)
		# print(google_vec.shape)

def get_vec_from_glove(vec_file,volc_file,dist_vec_file,em_dim=100):
	volcabulary = list(line.split('\t')[1].strip() for line in open(volc_file,'r+'))
	vecs = np.zeros((len(volcabulary)+1,em_dim))
	i = 0
	with open(vec_file,'r+') as in_f, open(dist_vec_file,'w+') as out_f:# open('../data/tmp','w+') as tmp_f:
		next(in_f)
		for line in in_f:
			word_end = line.find(' ')
			if line[:word_end] in volcabulary:
				i += 1
				# print(line[word_end:])
				# tmp_f.write(line)
				data = np.fromstring(line[word_end+1:],dtype='float',count=em_dim,sep=' ')
				# data = np.fromiter(line[word_end+1:],dtype='float32',count=em_dim)
				vecs[volcabulary.index(line[:word_end])][:] = data
		print("number of words in the glove vec file: "+str(i-1)) #4003

		#adding unknown words
		idxs = np.unique(np.where(vecs==0.0)[0])
		# print('missing unk words: '+str(len(idxs)))
		for i in idxs:
			vecs[i] = np.random.uniform(-0.25,0.25,em_dim)
		###### need to add a last one for unk word		
		np.savetxt(dist_vec_file, vecs,fmt='%.9f',delimiter=' ')

def get_idx_from_sentence(sentence,word2idx_map_file,max_len=67,kernel_size=5):
	"""
	given a sentence, translate it into list of indexes with padding of 0s
	padded length of sequence: 67+2*(5-1) = 75
	"""
	with open(word2idx_map_file,'rb') as handle:
		volc_map = pickle.load(handle)
	handle.close()

	idxs = []
	pad = kernel_size - 1
    #no matter how long is the sentence,
    #always pad with at least begining padding and ending padding
    #hence the every input is of length (max_len+2*pad) =75
	for i in range(pad):
		#padding in the beginning of a sentence
		idxs.append(0)
	# words = sentence.split()

	for word in sentence:
		if word in volc_map:
			idxs.append(volc_map[word])
	while len(idxs) < max_len+2*pad:
		#padding at the end 
		idxs.append(0)

	return idxs

def encode_label(label):
	labels = ['negative','positive','neutral']
	if label == 'unknown':
		return -1
	else:
		return labels.index(label)

def decode_label(index):
	labels= ['negative','positive','neutral']
	return labels[index]

def dataset2array(input_flie):
	"""
	given datafile (train/val) in such format:

	and map each word to index given in word2idx_map_file,
	return array of (N,len+1) where
	N: the number of training examples
	len+1: the length of each sentence + encoded sentiment label
	"""
	data = []
	with open(input_flie,'r+') as f:
		for line in f:
			sentence = line.split()
			sentence[0:-1] = get_idx_from_sentence(sentence[0:-1], word2idx_map_file)
			sentence[-1] = encode_label(sentence[-1])
			data.append(sentence)
	idx_array = np.array(data, dtype=np.int)
	return idx_array

def output_test(tokenized_file, output_file,prediction):
	"""
	create test file in the correct format for scoring script
	"""
	i=0
	with open(tokenized_file,'r+') as f, open(output_file,'w+') as out_f:
		for line in f:
			id1,id2,sentiment,sentence = line.split('\t')
			sentiment = decode_label(prediction[i])
			newline = 'NA\t'+str(i+1)+'\t'+sentiment+'\n'
			out_f.writelines(newline)
			i+=1

def output_test_15(tokenized_file, output_file,prediction):
	"""
	create test file in the correct format for scoring script
	in accordance with score-semeval2015-task10-subtaskB.pl
	"""
	i=0
	with open(tokenized_file,'r+') as f, open(output_file,'w+') as out_f:
		for line in f:
			id1,id2,sentiment,sentence = line.split('\t')
			sentiment = decode_label(prediction[i])
			newline = 'NA\t'+id2+'\t'+sentiment+'\n'
			out_f.writelines(newline)
			i+=1

def merge_test_set(index_file,complete_file,test_file):
	"""
	Routine for merging the downloaded incomplete test file
	with given complete test file in different format.
	final test file format:
	idx1<TAB>idx2<TAB>unknown<TAB>sentence
	"""
	with open(index_file,'r+') as index_f, open(complete_file,'r+') as comp_f, open(test_file,'w+') as f:
		for line in zip(index_f,comp_f):
			# print(line[0])
			info = line[0].split('\t')
			id1,id2,sentiment = info[0],info[1],info[2].strip()
			content = line[1].split('\t')[2]
			newline = id1.strip()+'\t'+id2.strip()+'\t'+sentiment+'\t'+content.strip()+'\n'
			f.write(newline)

def main():
	merge_test_set('../data/original/SemEval2015-task10-test-B-gold.txt', '../data/original/test-15.csv', '../data/test_merged_15.tsv')


if __name__ == "__main__":
	main()