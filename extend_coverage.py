import pandas as pd 
import numpy as np
import operator
from gensim.models import KeyedVectors
import re

def build_vocab(sentences, verbose =  True):
    """
    :param sentences: list of list of words
    :return: dictionary of words and their count
    """
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab

def check_coverage(vocab,embeddings_index):

    a = {}
    oov = {}
    k = 0
    i = 0
    for word in vocab:
        try:
            a[word] = embeddings_index[word]
            k += vocab[word]
        except:

            oov[word] = vocab[word]
            i += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x

def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re


mispell_dict = {'quora':'social media',
                'instagram':'social media',
                'btech':'professional degree',
                'upsc':'entrance exam',
                'bitcoin':'online currency',
                'quorans':'forum members',
                'whatsapp':'social media',
                'mbbs':'professional degree',
                'isnt':'is not',
                'indias':'countrymen',
                'narendra':'politician',
                'iit':'education institute',
                'iim':'education institute',
                'doesnt':'does not'}
mispellings, mispellings_re = _get_mispell(mispell_dict)

def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)   

if __name__ == '__main__':
    
    train=pd.read_csv('Data/cleaned_text.csv')
    train=train.dropna()
    news_path = 'GoogleNews-vectors-negative300.bin'
    embeddings_index = KeyedVectors.load_word2vec_format(news_path, binary=True)
    train["question_text"] = train["question_text"].apply(lambda x: replace_typical_misspell(x))
    to_remove = ['a','and','to','it','of','and','is']
    train['question_text']=train['question_text'].apply(lambda x:' '.join([t for t in x.split() if t not in to_remove]))
    sentences = train["question_text"].apply(lambda x: x.split())
    vocab = build_vocab(sentences)
    oov = check_coverage(vocab,embeddings_index)
    print(oov[:20])
    train.to_csv('Data/extended_text.csv',index=False)