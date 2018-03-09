# CNN for Sentiment Analysis
Convolutional Neural Network for Sentiment Analysis on Twitter data.
The model is implemented with Keras in Python2. 

## Files
data/ contains original data set, as well as some file generated during data pre-processing.
scripts/ contains scripts for preprocessing data, training CNN and testing CNN.
model/ contains best CNN model that is trained so far.

## Usage

For training the model:
```
	python scripts/train.py <raw_train_file> <raw_val_file> <vec-to-use>
```

e.g.:
```
	python scripts/train.py ../data/original/train_full.tsv ../data/original/test_13.tsv glove
```

For testing the model:
```

	python scripts/test.py <raw_test_file> <model_file> <scoring script>
```

e.g.:
```
	python scripts/test.py ../data/test_merged.tsv ../model/model-glove.h5 ./score-semeval2014-task9-subtaskB.pl
```
