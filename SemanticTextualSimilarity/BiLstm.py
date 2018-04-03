#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import tensorflow as tf
from gensim.models import word2vec
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




    word2vec_model_file_path=os.path.join(RootPath,'..','GoogleNews-vectors-negative300.bin')
    train_data_path=os.path.join(RootPath,'train.pickle')

    word2vec_model=word2vec.KeyedVectors.load_word2vec_format(word2vec_model_file_path,binary=True)
    print(word2vec_model.vector_size)
    print('index2word len is %s ,example %s' %(len(word2vec_model.index2word),word2vec_model.index2word[:10]))
    print('vocab len is %s ,example %s ' %(len(word2vec_model.vocab),word2vec_model.vocab['good']))
    print('127 is %s , vocab is %s , the item is %s' %(word2vec_model.index2word[127],type(word2vec_model.vocab),type(word2vec_model.vocab['good'])))
    print('index %s , count %s' %(word2vec_model.vocab['good'].index,word2vec_model.vocab['good'].count))
    print('word2vec.wv is %s , the len si %s' %(word2vec_model.wv,len(word2vec_model.wv)))

    data_dict=pickle.load(open(train_data_path,'rb'))
    sents1=data_dict['sents1']
    sents2=data_dict['sents2']
    labels=data_dict['labels']

    print('sents1 is %s , sents2 is %s , labels is %s' % (len(sents1),len(sents2),len(labels)))

    assert(len(sents1)==len(sents2) and len(sents1)==len(labels))

    def get_batch(batch_size,sents1,sents2,labels):
        count=0
        sents1_ids=[]
        sents1_lens=[]
        sents2_ids=[]
        sents2_lens=[]
        batch_labels=[]
        for i in range(len(sents1)):
            t1=sents1[i].split()
            sents1_ids.append([word2vec_model.vocab[w.strip()].index for w in t1])
            sents1_lens.append([len(w) for w in t1])

            t2 = sents2[i].split()
            sents2_ids.append([word2vec_model.vocab[w.strip()].index for w in t2])
            sents2_lens.append([len(w) for w in t2])

            batch_labels.append(labels[i])
            count+=1
            if(count==5):
                yield sents1_ids,sents1_lens,sents2_ids,sents2_lens,batch_labels
        else:
            if(count>0):
                yield sents1_ids, sents1_lens, sents2_ids, sents2_lens, batch_labels

    results=get_batch(batch_size,sents1,sents2,labels)
    print(results)

    exit(-1)


    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.allow_soft_placement=True

    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            bilstmmodel = BiLSTMModel(max_sent_len=max_seq_len,word2vec=word2vec_model.wv, lstm_num_units=lstm_num_units,
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

            def train_step(summary_writer,sents1_ids,sents1_lens,sents2_ids,sents2_lens,labels,input_keep_prob,output_keep_prob):
                feeddict={bilstmmodel.sent1_ids:sents1_ids,
                          bilstmmodel.sent1_lens: sents1_lens,
                          bilstmmodel.sent2_ids:sents2_ids,
                          bilstmmodel.sent2_lens:sents2_lens,
                          bilstmmodel.score_labels:labels,
                          bilstmmodel.input_keep_prob:input_keep_prob,
                          bilstmmodel.output_keep_prob:output_keep_prob}

                _train_op,_step,summary,loss,accuracy=sess.run([train_op,global_step,summary_writer,bilstmmodel.loss,bilstmmodel.accuracy],feed_dict=feeddict)
                print('Train:step: %s ,loss: %s ,acc: %s' %(_step,loss,accuracy))
                summary_writer.add_summary(summary=summary,global_step=_step)



            #Start Training
