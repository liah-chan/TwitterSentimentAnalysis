import re
import sys

if(len(sys.argv) < 2):
    print "Usage: " + sys.argv[0] + "<reduced_train_file> <tokenized_train_file>"
    sys.exit(0)

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
file_for_tokenize = sys.argv[1] #'./train_messages.txt'
tokenized_file = sys.argv[2] #'train_tokenized.txt'
with open(file_for_tokenize, 'r') as f, open(tokenized_file,'w+') as dest_f:
    for line in f:
        tokens = preprocess(line,lowercase=True)
        # print(tokens)
        newline = ' '.join(tokens)
        newline += '\n'
        dest_f.writelines(newline)
        # break
dest_f.close()
f.close()