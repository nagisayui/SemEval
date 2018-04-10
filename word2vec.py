#!/usr/bin/env python
#-*- coding:utf-8 -*-


import os, sys, re
import numpy as np

Root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(Root)


class MyWord2Vec(object):
    def __init__(self, word2vec_class):
        np.random.seed(1234)

        word2vec_dir = Root
        # Load vocab
        with open(os.path.join(word2vec_dir, word2vec_class + '.vocab')) as fr:
            # self.vocab = [line.strip() for line in fr if line.strip()]
            self.vocab = [line.strip() for line in fr]

        # Load words vector data
        self.words_vec = np.load(os.path.join(word2vec_dir, word2vec_class + '.data.npy'))
        self.dim = self.words_vec.shape[1]
        # Check data
        assert len(self.vocab) == self.words_vec.shape[0]

        print(len(self.vocab))
        self.vocab_dict = {w: i for i, w in enumerate(self.vocab)}
        print(len(self.vocab_dict))
        # For special symbol
        special_sym_list = ['_NE_', '<S>', '</S>']
        # special_sym_list = []
        self.words_vec = np.vstack((self.words_vec, np.zeros([len(special_sym_list)+1, self.dim])))
        for sym in special_sym_list:
            self.vocab.append(sym)
            self.vocab_dict[sym] = len(self.vocab) - 1

        print(len(self.vocab))
        print(len(self.vocab_dict))
        print(self.words_vec.shape[0]-1)
        assert len(self.vocab) == len(self.vocab_dict)
        assert len(self.vocab) == self.words_vec.shape[0] - 1
        # Add default vector for words not listed in vocab
        # self.words_vec[-1] = np.random.normal(size=[self.dim])
        self.vocab_size = len(self.vocab)
        self.id_for_unlisted_word = self.vocab_size

        # Pattern for number
        self.num_pat = re.compile('^\d+(\.\d+)*$')
        # self.num_sym = '_NUM_'
        self.num_sym = 'NUMBER'
        assert self.num_sym in self.vocab

    def w2id(self, word):
        return self.vocab_dict.get(word, self.id_for_unlisted_word)

    def unlisted(self, word):
        return self.w2id(word) == self.id_for_unlisted_word

    def get(self, word):
        return self.words_vec[self.w2id(word)]

    def sent2vecs(self, sent):
        vectors = np.zeros((len(sent), self.dim))
        for i, word in enumerate(sent):
            vectors[i] = self.get(word)
        return vectors

    # Text is list of words
    def texts2ids(self, text_list, sequence_length, actual_len=False):
        assert isinstance(text_list[0], list)

        fill_sym = '_FIL_'      # Special symbol for filling
        assert fill_sym not in self.vocab

        alen_list = []
        ids = np.zeros([len(text_list), sequence_length], dtype=np.int32)
        for i, text in enumerate(text_list):
            subbed_text = [self.num_sym if re.match(self.num_pat, w) else w
                           for w in text][:sequence_length]
            alen_list.append(len(subbed_text))
            if len(subbed_text) < sequence_length:
                subbed_text.extend([fill_sym for idx in 
                                    range(sequence_length - len(subbed_text))])
                assert len(subbed_text) == sequence_length
            ids[i] = [self.w2id(w) for w in subbed_text]

        if actual_len:
            return ids, np.array(alen_list)
        else:
            return ids
    
if __name__ == '__main__':
    word2vec = MyWord2Vec('wiki300')
    text_list = [
        ['我', '爱', '吃', '苹果'],
        ['2012', '年', '2', '月'],
        ['你', '知道', '庵摩', '罗识', '相关', '资料', '是', '什么', '吗', '?']]
    ids = word2vec.texts2ids(text_list, 10)
    print(ids)
    pass
