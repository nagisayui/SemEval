#!/usr/bin/env python
#-*- coding:utf-8 -*-
import os
from gensim.models import KeyedVectors
RootPath=os.path.join(os.path.dirname(os.path.abspath(__file__)))

#　转换模型信息的格式,二进制转为txt
def bin2txt(model_path):
    word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    txt_path = os.path.basename(model_path) + '.txt'
    fvocab = 'vocab.txt'
    word2vec_model.wv.save_word2vec_format(fname=txt_path, fvocab=fvocab, binary=False, total_vec=None)

#　转换模型信息的格式,txt转为二进制
def txt2bin(model_path):
    word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=False)
    bin_path = os.path.basename(model_path) + '.bin'
    fvocab = 'vocab.txt'
    word2vec_model.wv.save_word2vec_format(fname=bin_path, fvocab=fvocab, binary=True, total_vec=None)


#计算行数，就是单词数
def getFileLineNums(filename):
    f = open(filename, 'r',encoding='utf-8')
    count = 0
    for line in f:
        count += 1
    return count

# 在glove模型文件前增加一行
def glove2wordvec(file_path,dim):
    voc_size=getFileLineNums(file_path)
    line='%d %d' % (voc_size,dim)
    with open(file_path, 'r',encoding='utf-8') as fin:
        with open(file_path+'.txt', 'w',encoding='utf-8') as fout:
            fout.write(line + "\n")
            for line in fin:
                fout.write(line)


if __name__=='__main__':
    # word2vec_model_path = os.path.join(RootPath, 'GoogleNews-vectors-negative300.bin')
    # # 从binary信息转为txt信息
    # bin2txt(word2vec_model_path)

    # 从txt信息转为bin信息
    # txt2bin(otherpath) #自定义路径

    glove_path=os.path.join(RootPath,'glove.6B.300d.txt')
    glove2wordvec(glove_path,300)

