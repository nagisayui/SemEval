#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import six.moves.cPickle as pickle

RootPath=os.path.join(os.path.dirname(os.path.abspath(__file__)))

if __name__=='__main__':
    train_data_path=os.path.join(RootPath,'data','STS.input.track5.en-en.txt')
    train_goldlabel_data_path=os.path.join(RootPath,'data','STS.gs.track5.en-en.txt')

    sents1=[]
    sents2=[]
    labels=[]

    with open(train_data_path,'r') as fr:
        for line in fr:
            sents = line.split('\t')
            if(len(sents)!=2):
                print('Error: %s' %(line))
                exit(-1)
            sents1.append(sents[0].strip())
            sents2.append(sents[1].strip())
    with open(train_goldlabel_data_path,'r') as fr:
        for line in fr:
            labels.append(line.strip())

    if(len(sents1)!=len(sents2) or len(sents1)!=len(labels)):
        print('Error:The number is not same!')
        exit(-1)

    datadict={}
    datadict['sents1']=sents1
    datadict['sents2']=sents2
    datadict['labels']=labels

    write2pickle = open(os.path.join(RootPath,'train.pickle'),'wb')
    pickle.dump(datadict,write2pickle,protocol=2)
    write2pickle.close()

    # b = pickle.load(open('train.pickle', 'rb'))
    # print(b)