#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os

from gensim.models import word2vec
RootPath=os.path.join(os.path.dirname(os.path.abspath(__file__)))

class Word2Vec():
    def __init__(self,model_file_path):
        word2vec_model=word2vec.Word2Vec.load(model_file_path)


if __name__=='__main__':
    pass