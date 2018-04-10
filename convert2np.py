#!/usr/bin/env python
# -*- coding:utf8 -*-

import numpy as np
import os, sys

def word2vec2np(path):
    fr = open(path, 'r',encoding='utf-8')
    # Read basic infomation
    vocab_size, dim = map(int, fr.readline().strip().split())
    # Initialize
    data = np.zeros((vocab_size, dim), dtype=np.float)
    vocab = []
    vocab_dict={}
    delete_idxs=[] #记录不合理的元素下标，用以删除
    dul_idx=[]
    # Read data
    for i, line in enumerate(fr):
        # line = line.strip()
        if not line:
            continue
        parts = line.split()
        if(len(parts)!=dim+1):
            delete_idxs.append(i)
            print('Error in index %s ' %(i))
            print('The error line is %s' % (line[:30]))
            continue
        word=parts.pop(0)
        vocab.append(word)
        vocab_dict[word]=i
        if(len(vocab_dict)+len(delete_idxs)-1<i):
            dul_idx.append(len(vocab)-1)
            delete_idxs.append(i)
            print('The same: %s' %(word))

        data[i]=list(map(float,parts))

        if ((i + 1) % 100000 == 0):
            print(i+1)

    # Delete InCorrect data
    delete_idxs.reverse() #从idx大的开始删除，以免删除了前面的idx，出现错位
    for idx in delete_idxs:
        print(idx)
        data=np.delete(data,idx,0)

    dul_idx.reverse()
    for idx in dul_idx:
        vocab.pop(idx)

    # Save data
    dname = os.path.dirname(path)
    fname = os.path.basename(path)
    fname_data = fname.rsplit('.', 1)[0] + ".data"
    fname_vocab = fname.rsplit('.', 1)[0] + '.vocab'

    print(len(vocab))
    print(len(data))
    np.save(fname_data, data)
    with open(fname_vocab, 'w') as fp:
        fp.write('\n'.join(vocab))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: %s input' % sys.argv[0])
        sys.exit(1)
    word2vec2np(sys.argv[1])
    pass
