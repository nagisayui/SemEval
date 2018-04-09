#!/usr/bin/env python
#-*- coding:utf-8 -*-
import os
from gensim.models import Word2Vec
RootPath=os.path.join(os.path.dirname(os.path.abspath(__file__)))

def extract_model_data(model_path):

    word2vec_model=Word2Vec.load_word2vec_format(model_path,binary=True)
    # keyedvectors = KeyedVectors.load_word2vec_format(model_path, binary=True)
    keyedvectors=word2vec_model.wv

    vec_dim=word2vec_model.vector_size
    voc_size=len(keyedvectors.vocab)
    print(keyedvectors.vector_size)
    print('index2word len is %s ,example %s' % (len(keyedvectors.index2word), keyedvectors.index2word[:10]))
    print('vocab len is %s ,example %s ' % (len(keyedvectors.vocab), keyedvectors.vocab['good']))
    print('127 is %s , vocab is %s , the item is %s' % (
    word2vec_model.index2word[127], type(keyedvectors.vocab), type(keyedvectors.vocab['good'])))
    print('index %s , count %s' % (keyedvectors.vocab['good'].index, keyedvectors.vocab['good'].count))
    print('word2vec.wv is %s , the len si %s' % (keyedvectors, len(keyedvectors)))

    write2path=os.path.join(os.path.dirname(os.path.abspath(model_path)),os.path.basename(model_path))
    fw=open(write2path,'w')
    # fw.write()


if __name__=='__main__':
    word2vec_model_path = os.path.join(RootPath, 'GoogleNews-vectors-negative300.bin')
    # extract_model_data(word2vec_model_path)
    word2vec_model = Word2Vec.load_word2vec_format(word2vec_model_path, binary=True)

    write2path = os.path.join(os.path.dirname(os.path.abspath(word2vec_model_path)), os.path.basename(word2vec_model_path))
    word2vec_model.save_word2vec_format()
