#!/usr/bin/env python
# coding:utf8

import numpy as np
import os, sys


def word2vec2np(path):
    fr = open(path, 'r')
    # Read basic infomation
    vocab_size, dim = map(int, fr.readline().strip().split())
    # Initialize
    data = np.zeros((vocab_size, dim), dtype=np.float)
    vocab = []
    # Read data
    for i, line in enumerate(fr):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        vocab.append(parts.pop(0))
        data[i] = map(float, parts)
    # Save data
    dname = os.path.dirname(path)
    fname = os.path.basename(path)
    fname_data = fname.rsplit('.', 1)[0] + '.data'
    fname_vocab = fname.rsplit('.', 1)[0] + '.vocab'
    np.save(fname_data, data)
    with open(fname_vocab, 'w') as fp:
        fp.write('\n'.join(vocab))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print >> sys.stderr, 'Usage: %s input' % sys.argv[0]
        sys.exit(1)
    word2vec2np(sys.argv[1])
    pass
