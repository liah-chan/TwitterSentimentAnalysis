import re
import sys

"""
remove unavailable tweets and tokenize tweet message
"""

# if(len(sys.argv) < 2):
#     print("Usage: " + sys.argv[0] + " <raw_file> <tokenized_file>")
#     sys.exit(0)

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""
 
regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
 
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r'(?:\d*\.\d+)', #any real numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    
    # r"(?:[?@#$%^&*!~`|\\+\-.,'\"]){2,}", #repeated special charactors
    # r'(?:&amp;)(?:&lt;)(?:&gt;)\1*'
    # r'(?:\w+&amp;\w+)',
    r'(?:&amp;|&gt;|&lt;+)',
    r'(?:\S)' # anything else
]
    
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)
 
def tokenize(s):
    return tokens_re.findall(s)
 
def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens
 
# tweet = "RT @marcobonzanini: just an example! :D http://example.com #NLP"
# print(preprocess(tweet))
# # ['RT', '@marcobonzanini', ':', 'just', 'an', 'example', '!', ':D', 'http://example.com', '#NLP']
def tokenizer_test(raw_file,tokenized_file,input_file):

    neu_sen = ['objective-OR-neutral','objective','neutral']
    with open(raw_file, 'r') as f, open(tokenized_file,'w+') as dest_f, open(input_file,'w+') as input_f:
        for line in f:
            id1,id2,sentiment,sentence = line.split('\t')
            if sentence == 'Not Available\n':
                continue
            if sentiment in neu_sen:
                sentiment = 'neutral'
            tokens = preprocess(sentence,lowercase=True)
            # print(tokens)
            newline = ' '.join(tokens)
            # newline += '\n'
            dest_f.writelines(id1+'\t'+id2+'\t'+sentiment+'\t'+newline+'\n')
            input_f.writelines(newline+' '+sentiment+'\n')

def tokenizer(raw_file,tokenized_file,input_file):

    # neu_sen = ['objective-OR-neutral','objective','neutral']
    with open(raw_file, 'r') as f, open(tokenized_file,'w+') as dest_f, open(input_file,'w+') as input_f:
        for line in f:
            content = line.split('\t')
            sentiment,idx,sentence = content[0],content[1],content[2]
            # if sentence == 'Not Available\n':
            #     continue
            # if sentiment in neu_sen:
            #     sentiment = 'neutral'
            tokens = preprocess(sentence,lowercase=True)
            # print(tokens)
            newline = ' '.join(tokens)
            # newline += '\n'
            # dest_f.writelines(idx+'\t'+sentiment+'\t'+newline+'\n')
            dest_f.writelines(newline+'\n')
            input_f.writelines(newline+' '+sentiment+'\n')

# def tokenizer_for_test(raw_test_file,tokenized_file,input_file):
#     with open(raw_test_file,'r') as f, open(tokenized_file,'w+') as dest_f, open(input_file,'w+') as input_f:
#         for line in f:
            

def main():
    tokenizer('../data/original/train_13.tsv', '../data/train_tokenized.tsv', '../data/train_input.tsv')

if __name__ == "__main__":
    main()