#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import csv
import six.moves.cPickle as pickle

RootPath=os.path.join(os.path.dirname(os.path.abspath(__file__)))

def txt2pickle():
    train_data_path = os.path.join(RootPath, 'data', 'STS.input.track5.en-en.txt')
    train_goldlabel_data_path = os.path.join(RootPath, 'data', 'STS.gs.track5.en-en.txt')

    sents1 = []
    sents2 = []
    labels = []

    with open(train_data_path, 'r') as fr:
        for line in fr:
            sents = line.split('\t')
            if (len(sents) != 2):
                print('Error: %s' % (line))
                exit(-1)
            sents1.append(sents[0].strip())
            sents2.append(sents[1].strip())
    with open(train_goldlabel_data_path, 'r') as fr:
        for line in fr:
            labels.append(line.strip())

    if (len(sents1) != len(sents2) or len(sents1) != len(labels)):
        print('Error:The number is not same!')
        exit(-1)

    datadict = {}
    datadict['sents1'] = sents1
    datadict['sents2'] = sents2
    datadict['labels'] = labels

    write2pickle = open(os.path.join(RootPath, 'eval.pickle'), 'wb')
    pickle.dump(datadict, write2pickle, protocol=2)
    write2pickle.close()

    # b = pickle.load(open('eval.pickle', 'rb'))
    # print(b)

def csv2pickle():
    with open(os.path.join(RootPath, 'data', 'Stsbenchmark', 'sts-test.csv'), 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = []
        c = 0
        for orow in reader:
            line = ''.join(orow)
            row = line.split('\t')
            try:
                if (float(row[4]) > 5 or float(row[4]) < 0):
                    c += 1
                    print(len(row))
                    print(orow)
                    print(row[-3:])
                else:
                    rows.append(row)
            except:
                print('ERROR')
    print(c)

    need_data = [row[4:7] for row in rows]
    print(need_data)

    sents1 = []
    sents2 = []
    labels = []
    for label, sent1, sent2 in need_data:
        sents1.append(sent1)
        sents2.append(sent2)
        labels.append(label)

    datadict = {}
    datadict['sents1'] = sents1
    datadict['sents2'] = sents2
    datadict['labels'] = labels

    assert (len(sents1) == len(sents2) and len(sents1) == len(labels))

    write2pickle = open(os.path.join(RootPath, 'test.pickle'), 'wb')
    pickle.dump(datadict, write2pickle, protocol=2)
    write2pickle.close()

    b = pickle.load(open('train.pickle', 'rb'))
    print(b)

if __name__=='__main__':
    csv2pickle()

