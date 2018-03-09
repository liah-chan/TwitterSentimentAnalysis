#! /bin/bash

#all word embeddings

# time ../word2vec/trunk/word2vec -train ./train_tokenized.txt \
# -output ./word_embeddings -cbow 0 \
# -size 100 -window 10  \
# -negative 1 -hs 0 -sample 1e-4 -threads 10 \
# -min-count 1 -binary 0 -iter 15;

time ../word2vec/trunk/word2vec -train ../data/train_tokenized.tsv \
-output ../data/word_embeddings_unsort -cbow 0 \
-size 100 -window 5  \
-negative 1 -hs 0 -sample 1e-4 -threads 10 \
-min-count 5 -binary 0 -iter 15;

#sort vectors
cat ../data/word_embeddings_unsort |sed '1d' | sed '1d' |sort > ../data/word_embeddings_sorted
cat ../data/word_embeddings_sorted | \
cut -d ' ' -f 2-101 >../data/embeddings_sorted

#create volcabulary with index
# cat train_tokenized.txt | tr ' ' '\n' | sort | uniq -i > volcabulary.txt
cat ../data/word_embeddings_sorted | \
cut -d ' ' -f 1 > ../data/vol.txt
awk '{ printf("%d\t%s\n", NR, $0) }' ../data/vol.txt > ../data/volcabulary
rm ../data/vol.txt
# awk 'BEGIN {idx=0} {printf("%d\t%s\n",$idx,$1); $idx++;} END{}' vol.txt > volcabulary