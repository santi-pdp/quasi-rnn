from __future__ import print_function
from collections import Counter
import cPickle as pickle
import numpy as np
import json
import timeit
import codecs
import gzip
import os
import re


class ptb_batch_loader(object):

    def __init__(self, data_dir, batch_size, seq_len):
        train_path = os.path.join(data_dir, 'train.txt')
        valid_path = os.path.join(data_dir, 'valid.txt')
        test_path = os.path.join(data_dir, 'test.txt')
        inputs_path = {'train':train_path, 'valid':valid_path, 'test':test_path}
        vocab_path = os.path.join(data_dir, 'vocab.pkl.gz')
        data_path = os.path.join(data_dir, 'data')
        self.batch_size = batch_size
        self.seq_len = seq_len
        if not os.path.exists(vocab_path):
            print('Processing raw PTB to make vocabulary and tensors...')
            beg_preproc = timeit.default_timer()
            self.dataX,self.dataY,self.vocab = self.text_to_tensor(inputs_path,
                                                                   vocab_path,
                                                                   data_path)
            print('dataX: ', self.dataX)
            with gzip.open(vocab_path, 'wb') as gh:
                pickle.dump(self.vocab, gh)
            for split in inputs_path.keys():
                print('dataX {}: {}'.format(split, self.dataX[split].shape))
                print('dataY {}: {}'.format(split, self.dataY[split].shape))
                np.save(data_path + 'X_' + split + '.npy', self.dataX[split])
                np.save(data_path + 'Y_' + split + '.npy', self.dataY[split])
            end_preproc = timeit.default_timer()
            print('Raw PTB processing: {} s'.format(end_preproc - beg_preproc))
        else:
            print('Loading cached data...')
            beg_load = timeit.default_timer()
            # load the vocab and data
            self.dataX = {}
            self.dataY = {}
            for split in inputs_path.keys():
                self.dataX[split] = np.load(data_path + 'X_' + split + '.npy')
                self.dataY[split] = np.load(data_path + 'Y_' + split + '.npy')
            with gzip.open(vocab_path, 'rb') as gh:
                self.vocab = pickle.load(gh)
            end_load = timeit.default_timer()
            print('Loaded vocab and tensors in {} s'.format(end_load - \
                                                            beg_load))
        self.batches_per_epoch = {'train':self.dataX['train'].shape[0] / \
                                          self.batch_size,
                                  'valid':self.dataX['valid'].shape[0] / \
                                          self.batch_size,
                                  'test':self.dataX['test'].shape[0] / \
                                          self.batch_size}

    def next_batch(self, split):
        # generator method
        batch_idx = {'train': 0, 'valid':0, 'test':0}
        dataX = self.dataX[split]
        dataY = self.dataY[split]
        batch_size = self.batch_size
        while True:
            if batch_idx[split] + 1 >= dataX.shape[0]:
                # re-start
                batch_idx[split] = 0
            batchX = dataX[batch_idx[split]:batch_idx[split] + batch_size]
            batchY = dataY[batch_idx[split]:batch_idx[split] + batch_size]
            yield batchX, batchY
            batch_idx[split] += batch_size


    def text_to_tensor(self, inputs_path, out_vocab_path, out_data_path):
        # first set up the <unk> token
        word2idx = {'<unk>': 0}
        # prepare the spacings splitter
        prog = re.compile('\s+')
        word2counts = {}
        split_counts = {}
        # build the vocab first
        for split in inputs_path.keys():
            # train, valid and test
            with codecs.open(inputs_path[split], 'r', encoding='utf8') as inf:
                #txt = inf.read()
                txt_count = None
                for txt in inf:
                    txt_words = prog.split(txt) + ['+']
                    if txt_count is None:
                        txt_count = Counter(txt_words)
                    else:
                        txt_count.update(Counter(txt_words))
                word2counts.update(txt_count)
                split_counts[split] = np.sum(txt_count.values())
        print('word counts per split: ', json.dumps(split_counts, indent=2))
        # construct the vocab with freq ordered strategy
        freq_words = [pair[0] for pair in sorted(word2counts.items(),
                                                 reverse=True,
                                                 key=lambda item: item[1])
                                                 if pair[0] != '<unk>']
        word2idx.update(dict((k, i) for i, k in enumerate(freq_words)))
        idx2word = dict((v, k) for k, v in word2idx.iteritems())
        print('Total vocab size: ', len(word2idx))
        dataX = {}
        dataY = {}
        # now build the 3 tensors
        for split in inputs_path.keys():
            with codecs.open(inputs_path[split], 'r', encoding='utf8') as inf:
                # initialize the whole tensor
                dataset = np.zeros(split_counts[split], dtype=np.int32)
                widx = 0
                for line in inf:
                    # appending EOS (+)
                    words = prog.split(line) + ['+']
                    for cword in filter(None, words):
                        try:
                            # set the index of the word into the tensor
                            dataset[widx] = word2idx[cword]
                        except:
                            dataset[widx] = word2idx['<unk>']
                        widx += 1
                L = (self.batch_size * self.seq_len)
                # chop last examples to make it stateful, if required
                print('{} split: samples before '
                      'trimming:{}'.format(split, dataset.shape[0]))
                dataset = dataset[:(dataset.shape[0] // L) * L + 1]
                print('{} split: samples after '
                      'trimming:{}'.format(split, dataset.shape[0]))
                #construct the X and Y data.
                # X never sees last sample, Y is the 1-shifted version of X
                y_data = dataset[1:]
                x_data = dataset[:-1]
                def interleave_tensor(x):
                    # break data to make chunks of (batch_size, seq_len)
                    aug_T = x.reshape((self.batch_size, -1,
                                             self.seq_len))
                    # make interleaving
                    tr_T = aug_T.transpose((1, 0, 2))
                    return tr_T.reshape((-1, self.seq_len))
                dataX[split] = interleave_tensor(x_data)
                dataY[split] = interleave_tensor(y_data)
        return dataX, dataY, {'word2idx':word2idx, 'idx2word':idx2word}
