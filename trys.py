#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os
import re
from six.moves import cPickle as pickle
from string import punctuation

RootPath=os.path.join(os.path.dirname(os.path.abspath(__file__)))

def getbatch(batch_size,sents1,sents2,labels):
    sents1_ids = []
    sents1_lens = []
    sents2_ids = []
    sents2_lens = []
    batch_labels = []

    for i in range(len(sents1)):
        t1=re.sub(pattern="([%s])" % (punctuation), repl=r" \1 ",string=sents1[i])
        t1=t1.split()
        sents1_ids.append(t1)
        sents1_lens.append([len(w) for w in t1])

        t2 = re.sub(pattern="([%s])" % (punctuation), repl=r" \1 ", string=sents2[i])
        t2 = t2.split()
        sents2_ids.append(t2)
        sents2_lens.append([len(w) for w in t2])

        batch_labels.append(labels[i])
        if (len(batch_labels) == batch_size):
            yield sents1_ids, sents1_lens, sents2_ids, sents2_lens, batch_labels
            sents1_ids = []
            sents1_lens = []
            sents2_ids = []
            sents2_lens = []
            batch_labels = []
    else:
        if (len(batch_labels)>0):
            yield sents1_ids, sents1_lens, sents2_ids, sents2_lens, batch_labels



if __name__=='__main__':
    test='asdd.sf?sdfew+&%$$@!$%^&*rtwead'
    print(test)
    # pattern中，用括号分组，repl中用\id 如\1替换组号
    t1 = re.sub(pattern="([%s])" % (punctuation), repl=r" \1 ", string=test)
    print(t1)

    train_data_path = os.path.join(RootPath, 'SemanticTextualSimilarity' ,'eval.pickle')

    data_dict = pickle.load(open(train_data_path, 'rb'))
    sents1 = data_dict['sents1']
    sents2 = data_dict['sents2']
    labels = data_dict['labels']

    batchs=getbatch(5, sents1, sents2, labels)
    for sents1_ids, sents1_lens, sents2_ids, sents2_lens, batch_labels in batchs:
        print('%s %s' % (len(sents1_ids),sents1_ids))
        print('%s %s' % (len(sents2_ids),sents2_ids))
