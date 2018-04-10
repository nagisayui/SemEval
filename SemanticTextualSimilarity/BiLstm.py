#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import re
import time
import numpy as np
import tensorflow as tf
from string import punctuation
from word2vec import MyWord2Vec
from six.moves import cPickle as pickle

RootPath=os.path.join(os.path.dirname(os.path.abspath(__file__)))

class BiLSTMModel():
    def __init__(self,max_sent_len,word2vec,lstm_num_units,sem_vec_len):
        # 句子1的词向量对应ID
        self.sent1_ids=tf.placeholder(dtype=tf.int32,shape=[None,max_sent_len],name='sent1_ids')
        # 每个句子1对应的实际长度
        self.sent1_lens=tf.placeholder(dtype=tf.int32,shape=[None],name='sent1_lens')
        # 句子2的词向量对应ID
        self.sent2_ids = tf.placeholder(dtype=tf.int32, shape=[None, max_sent_len], name='sent2_ids')
        # 每个句子2对应的实际长度
        self.sent2_lens = tf.placeholder(dtype=tf.int32, shape=[None], name='sent2_lens')
        # 每一对句子的相似度，0-5, 0 几乎不相关,5 几乎一样
        self.score_labels=tf.placeholder(dtype=tf.float32,shape=[None],name='score_labels')

        #Dropout keep prob
        self.input_keep_prob=tf.placeholder(dtype=tf.float32,shape=[None],name='input_keep_prob')
        self.output_keep_prob=tf.placeholder(dtype=tf.float32,shape=[None],name='output_keep_prob')

        #Embedding Layer
        with tf.device('/cpu:0'),tf.name_scope('Embedding'):
            word2vec=tf.Variable(word2vec,dtype=tf.float32,trainable=False,name='word2vec')
            self.sent1_embed=tf.nn.embedding_lookup(self.sent1_ids,word2vec,name='sent1_embed')
            self.sent2_embed=tf.nn.embedding_lookup(self.sent2_ids,word2vec,name='sent2_embed')

        def BiLstmLayer(inputs,seq_len,lstm_num_units,output_size):
            cell_fw=tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_num_units)
            cell_fw=tf.nn.rnn_cell.DropoutWrapper(cell_fw,input_keep_prob=1.0,output_keep_prob=1.0)
            cell_bw=tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_num_units)
            cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, input_keep_prob=1.0, output_keep_prob=1.0)

            outputs,states=tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,inputs,sequence_length=seq_len,dtype=tf.float32)

            concat_output=tf.concat(outputs,axis=2)

            #Get last outputs
            batch_size = tf.shape(outputs)[0]
            max_seq_len = tf.shape(outputs)[1]
            idx_list = tf.range(batch_size) * max_seq_len + (seq_len - 1)
            last_outputs = tf.gather(tf.reshape(concat_output,[-1,2*lstm_num_units]),idx_list)

            W = tf.Variable(tf.truncated_normal((2*lstm_num_units,output_size),stddev=0.1),name='W')
            b = tf.Variable(tf.constant(0.1,shape=[output_size]),name='b')
            sem = tf.sigmoid(tf.nn.xw_plus_b(last_outputs,W,b),name='sem')

            return sem

        #sent1 Sem
        with tf.variable_scope('sent1'):
            self.sent1_sem=BiLstmLayer(self.sent1_embed,self.sent1_lens,lstm_num_units,output_size=sem_vec_len)

        #sent2 Sem
        with tf.variable_scope('sent2'):
            self.sent2_sem=BiLstmLayer(self.sent2_embed,self.sent2_lens,lstm_num_units,output_size=sem_vec_len)

        #Calcuate semantic (cosine) similarity
        sent1_norm=tf.sqrt(tf.reduce_sum(tf.square(self.sent1_sem),axis=1))
        sent2_norm=tf.sqrt(tf.reduce_sum(tf.square(self.sent2_sem),axis=1))
        sent1xsent2=tf.reduce_sum(tf.multiply(self.sent1_sem,self.sent2_sem),axis=1)
        self.sem_sim=tf.divide(tf.multiply(sent1_norm,sent2_norm),sent1xsent2,name='sem_sim')

        #Predict
        self.pred_score=tf.multiply(self.sem_sim,5)

        #Loss
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.score_labels,logits=self.pred_score,name='loss')

        #Accuracy
        self.accuracy = tf.reduce_mean(tf.equal(tf.less(tf.abs(tf.subtract(self.pred_score,self.score_labels)))))

# 将英语句子拆分为一个个单词
def split_sents(sents):
    s_sents=[]
    for sent in sents:
        s = re.sub(pattern="([%s])" % (punctuation), repl=r" \1 ", string=sent)
        s = s.split()
        s_sents.append(s)
    return s_sents


# get batch data
def get_batch(batch_size, sents1,sents1_lens, sents2, sents2_lens,labels):
    batch_sents1 = []
    batch_sents2= []
    batch_labels = []

    for i in range(len(sents1)):
        t1 = re.sub(pattern="([%s])" % (punctuation), repl=r" \1 ", string=sents1[i])
        t1 = t1.split()
        batch_sents1.append(t1)

        t2 = re.sub(pattern="([%s])" % (punctuation), repl=r" \1 ", string=sents2[i])
        t2 = t2.split()
        batch_sents2.append(t2)

        batch_labels.append(labels[i])
        if (len(batch_labels) == batch_size):
            yield batch_sents1, batch_sents2, batch_labels
            batch_sents1 = []
            batch_sents2 = []
            batch_labels = []
    else:
        if (len(batch_labels) > 0):
            yield batch_sents1, batch_sents2, batch_labels

if __name__=='__main__':

    #Model
    max_seq_len=30
    lstm_num_units=300
    sem_vec_len=300
    input_keep_prob=1.0
    output_keep_prob=1.0

    #Train
    batch_size=5
    num_checkpoins=5
    max_num_undesc=5
    evaluate_every=5

    train_data_path=os.path.join(RootPath,'train.pickle')
    dev_data_path=os.path.join(RootPath,'dev.pickle')

    train_data_dict=pickle.load(open(train_data_path,'rb'))
    train_sents1=train_data_dict['sents1']
    train_sents1 = split_sents(train_sents1)
    train_sents2=train_data_dict['sents2']
    train_sents2 = split_sents(train_sents2)
    train_labels=train_data_dict['labels']

    dev_data_dict=pickle.load(open(dev_data_path),'rb')
    dev_sents1 = dev_data_dict['sents1']
    dev_sents1 = split_sents(dev_sents1)
    dev_sents2 = dev_data_dict['sents2']
    dev_sents2 = split_sents(dev_sents2)
    dev_labels = dev_data_dict['labels']

    assert(len(train_sents1)==len(train_sents2) and len(train_sents1)==len(train_labels))

    myWord2Vec = MyWord2Vec('GoogleNews300')
    sents1_id_list, sents1_len_list = myWord2Vec.texts2ids(text_list=train_sents1, sequence_length=max_seq_len, actual_len=True)
    sents2_id_list, sents2_len_list = myWord2Vec.texts2ids(text_list=train_sents2, sequence_length=max_seq_len, actual_len=True)

    dev_sents1_id_list,dev_sents1_len_list = myWord2Vec.texts2ids(dev_sents1,max_seq_len,actual_len=True)
    dev_sents2_id_list,dev_sents2_len_list = myWord2Vec.texts2ids(dev_sents2,max_seq_len,actual_len=True)


    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.allow_soft_placement=True

    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            bilstmmodel = BiLSTMModel(max_sent_len=max_seq_len,word2vec=myWord2Vec.words_vec, lstm_num_units=lstm_num_units,
                                      sem_vec_len=sem_vec_len)
            global_step=tf.Variable(0,trainable=False,name='global_step')
            optimizer=tf.train.AdagradOptimizer(learning_rate=1e-4)
            grads_and_vars=optimizer.compute_gradients(bilstmmodel.loss)
            train_op=optimizer.apply_gradients(grads_and_vars=grads_and_vars,global_step=global_step)

            #output dir : models and summaries
            out_dir=os.path.join(RootPath,'model')

            # summaries for loss and accuracy
            loss_summary=tf.summary.scalar('loss',bilstmmodel.loss)
            accuracy_summary=tf.summary.scalar('accuary',bilstmmodel.accuracy)

            # Train summaries
            train_summary_op=tf.summary.merge([loss_summary,accuracy_summary])
            train_summary_dir=os.path.join(out_dir,'summary','train')
            train_summary_writer=tf.summary.FileWriter(train_summary_dir,sess.graph)

            # dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, accuracy_summary])
            dev_summary_dir = os.path.join(out_dir, 'summary', 'dev')
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # checkpoint directory
            checkpoint_dir=os.path.join(out_dir,'checkpoint')
            checkpoint_prefix=os.path.join(checkpoint_dir,'model')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver=tf.train.Saver(tf.global_variables(),max_to_keep=num_checkpoins)

            #Initilize all variables
            sess.run(tf.global_variables_initializer())

            def one_step(tag,summary_writer,sent1_ids,sent1_lens,sent2_ids,sent2_lens,labels,input_keep_prob,output_keep_prob):
                feeddict={bilstmmodel.sent1_ids:sent1_ids,
                          bilstmmodel.sent1_lens:sent1_lens,
                          bilstmmodel.sent2_ids:sent2_ids,
                          bilstmmodel.sent2_lens:sent2_lens,
                          bilstmmodel.score_labels:labels,
                          bilstmmodel.input_keep_prob:input_keep_prob,
                          bilstmmodel.output_keep_prob:output_keep_prob}

                _train_op,_step,summary,loss,accuracy=sess.run([train_op,global_step,summary_writer,bilstmmodel.loss,bilstmmodel.accuracy],feed_dict=feeddict)
                print('%s :step: %s ,loss: %s ,acc: %s' %(tag,_step,loss,accuracy))
                summary_writer.add_summary(summary=summary,global_step=_step)
                return loss,accuracy

            #Start Training

            batches=get_batch(batch_size=batch_size,sents1=sents1_id_list,sents1_lens=sents1_len_list,sents2=sents2_id_list,sents2_lens=sents2_len_list)

            min_dev_loss = float('inf')
            dev_acc_for_min_loss = 0
            num_undesc = 0

            for batch_sents1_ids,batch_sents1_lens,batch_sents2_ids,batch_sents2_lens,batch_labels in batches:
                if num_undesc > max_num_undesc:
                    break

                one_step('Train',train_summary_writer,batch_sents1_ids,batch_sents1_lens,batch_sents2_ids,batch_sents2_lens,batch_labels)
                current_step=tf.train.global_step(sess=sess,global_step_tensor=global_step)

                if(current_step%evaluate_every==0):
                    print('\nEvaluation:')
                    dev_loss_list = []
                    dev_acc_list = []
                    for dev_batch in get_batch(batch_size=len(dev_labels)/20,sents1=dev_sents1_id_list,sents1_lens=sents1_len_list,sents2=dev_sents2_id_list,sents2_lens=sents2_len_list):
                        dev_batch_sents1_ids, dev_batch_sents1_lens, dev_batch_sents2_ids,dev_batch_sents2_lens, dev_batch_labels=dev_batch
                        dev_batch_loss,dev_batch_acc=one_step('Dev',dev_summary_writer,dev_batch_sents1_ids,dev_batch_sents1_lens,dev_batch_sents2_ids,dev_batch_sents2_lens,dev_batch_labels)

                        dev_loss_list.append(dev_batch_loss)
                        dev_acc_list.append(dev_batch_acc)

                    dev_loss = np.mean(dev_loss_list)
                    dev_acc = np.mean(dev_acc_list)
                    if dev_loss < min_dev_loss:
                        min_dev_loss = dev_loss
                        dev_acc_for_min_loss = dev_acc
                        # Save checkpoints
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print('Saved model checkpoint to %s' % path)
                        num_undesc = 0
                    else:
                        num_undesc += 1
                    print('DEV: loss %s, acc %s\n' % (dev_loss, dev_acc))
                    time.sleep(3)

                print('Minimum dev loss: %g, acc: %g' % (min_dev_loss, dev_acc_for_min_loss))