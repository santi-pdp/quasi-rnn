import tensorflow as tf
from qrnn import QRNN_layer
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.layers import flatten, fully_connected
import os

def scalar_summary(name, x):
    try:
        summ = tf.summary.scalar(name, x)
    except AttributeError:
        summ = tf.scalar_summary(name, x)
    return summ

def histogram_summary(name, x):
    try:
        summ = tf.summary.histogram(name, x)
    except AttributeError:
        summ = tf.histogram_summary(name, x)
    return summ

class QRNN_lm(object):
    """ Implement the Language Model from https://arxiv.org/abs/1611.01576 """
    def __init__(self, args, infer=False):
        self.batch_size = args.batch_size
        self.infer = infer
        self.seq_len = args.seq_len
        self.vocab_size = args.vocab_size
        self.emb_dim = args.emb_dim
        self.zoneout = args.zoneout
        self.dropout = args.dropout
        self.qrnn_size = args.qrnn_size
        self.qrnn_layers = args.qrnn_layers
        self.words_in = tf.placeholder(tf.int32, [self.batch_size,
                                                  self.seq_len])
        self.words_gtruth = tf.placeholder(tf.int32, [self.batch_size,
                                                      self.seq_len])

        self.logits, self.output = self.inference()
        self.loss = self.lm_loss(self.logits, self.words_gtruth)
        self.loss_summary = scalar_summary('loss', self.loss)
        self.perp_summary = scalar_summary('perplexity', tf.exp(self.loss))
        # set up optimizer
        self.lr = tf.Variable(args.learning_rate, trainable=False)
        self.lr_summary = scalar_summary('lr', self.lr)
        tvars = tf.trainable_variables()
        grads = []
        for grad in tf.gradients(self.loss, tvars):
            if grad is not None:
                grads.append(tf.clip_by_norm(grad, args.grad_clip))
            else:
                grads.append(grad)
        #grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
        #                                  args.grad_clip)
        self.opt = tf.train.GradientDescentOptimizer(self.lr)
        self.train_op = self.opt.apply_gradients(zip(grads, tvars))

    def inference(self):
        words_in = self.words_in
        embeddings = None
        # keep track of Recurrent states to re-initialize them when needed
        self.initial_states = []
        self.last_states = []
        self.qrnns = []
        with tf.variable_scope('QRNN_LM'):
            word_W = tf.get_variable("word_W",
                                     [self.vocab_size,
                                      self.emb_dim])
            words = tf.split(1, self.seq_len, tf.expand_dims(words_in, -1))
            # print('len of words: ', len(words))
            for word_idx in words:
                word_embed = tf.nn.embedding_lookup(word_W, word_idx)
                if not self.infer and self.dropout > 0:
                    word_embed = tf.nn.dropout(word_embed, (1. - self.dropout),
                                               name='dout_word_emb')
                # print('word embed shape: ', word_embed.get_shape().as_list())
                if embeddings is None:
                    embeddings = tf.squeeze(word_embed, [1])
                else:
                    embeddings = tf.concat(1, [embeddings,
                                           tf.squeeze(word_embed, [1])])
            qrnn_h = embeddings
            for qrnn_l in range(self.qrnn_layers):
                qrnn_ = QRNN_layer(qrnn_h, self.qrnn_size, pool_type='fo',
                                   zoneout=self.zoneout,
                                   name='QRNN_layer{}'.format(qrnn_l))
                qrnn_h = qrnn_.h
                # apply dropout if required
                if not self.infer and self.dropout > 0:
                    qrnn_h_f = tf.reshape(qrnn_h, [-1, self.qrnn_size])
                    qrnn_h_dout = tf.nn.dropout(qrnn_h_f, (1. - self.dropout),
                                                name='dout_qrnn{}'.format(qrnn_l))
                    qrnn_h = tf.reshape(qrnn_h_dout, [self.batch_size, -1, self.qrnn_size])
                self.last_states.append(qrnn_.last_state)
                histogram_summary('qrnn_state_{}'.format(qrnn_l),
                                  qrnn_.last_state)
                scalar_summary('qrnn_avg_state_{}'.format(qrnn_l),
                               tf.reduce_mean(qrnn_.last_state))
                self.initial_states.append(qrnn_.initial_state)
                self.qrnns.append(qrnn_)
            qrnn_h_f = tf.reshape(qrnn_h, [-1, self.qrnn_size])
            logits = fully_connected(qrnn_h_f,
                                     self.vocab_size,
                                     weights_initializer=xavier_initializer(),
                                     scope='output_softmax')
            output = tf.nn.softmax(logits)
            return logits, output

    def lm_loss(self, logits, words_gtruth):
        f_words_gtruth = tf.reshape(words_gtruth,
                                    [self.batch_size * self.seq_len])
        loss =  tf.nn.sparse_softmax_cross_entropy_with_logits(logits,
                                                               f_words_gtruth)
        return tf.reduce_mean(loss)

    def save(self, sess, save_filename, global_step):
        if not hasattr(self, 'saver'):
            self.saver = tf.train.Saver()
        print('Saving checkpoint to {}...'.format(save_filename))
        self.saver.save(sess, save_filename, global_step)

    def load(self, sess, save_path):
        if not hasattr(self, 'saver'):
            self.saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(save_path)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            print('Loading checkpoint {}...'.format(ckpt_name))
            self.saver.restore(sess, os.path.join(save_path, ckpt_name))
            return True
        else:
            return False
