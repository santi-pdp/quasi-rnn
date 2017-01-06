import tensorflow as tf
import cPickle as pickle
from model import QRNN_lm
import numpy as np
import codecs
import json
import gzip
import os
import re


flags = tf.app.flags
flags.DEFINE_integer("num_words", 100, "Num words to generate (Def: 100).")
flags.DEFINE_string("load_path", "lm-qrnn_model", "Model path "
                                                  "(Def: lm-qrnn_model).")
flags.DEFINE_string("first_word", "hello", "First word to begin sampling "
                                                  "(Def: hello).")
flags.DEFINE_string("save_path", "lm-samples", "Out samples path "
                                               "(Def: lm-samples).")
flags.DEFINE_string("out_filename", "sample.txt", "Output filename for"
                                                  " dumping txt (Def: sample"
                                                  ".txt)")
flags.DEFINE_string("vocab_path", None, "Vocab pickle file path "
                                        "(Def: None).")
flags.DEFINE_boolean("plot_Z", False, "Flag to plot Z (Def: True).")

FLAGS = flags.FLAGS

class Dict2Flags(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

def main(_):
    args = FLAGS
    print('Parsed options: ')
    print(json.dumps(args.__flags, indent=2))
    if args.vocab_path is None:
        raise ValueError('Vocabulary path must be specified!')

    with gzip.open(os.path.join(args.load_path, 'config.pkl.gz'), 'rb') as f:
        saved_args = pickle.load(f)
        saved_args = Dict2Flags(saved_args)
    print('loaded saved args')
    # build the model
    qrnn_lm = QRNN_lm(saved_args, infer=True)
    # set up an interactive session
    sess = tf.InteractiveSession()
    # load the model graph
    qrnn_lm.load(sess, args.load_path)
    with gzip.open(args.vocab_path) as gh:
        vocab = pickle.load(gh)
    out_text, Z = qrnn_lm.sample(sess, args.num_words, vocab,
                              first_word=args.first_word)
    Z1 = np.array(Z[0])
    Z2 = np.array(Z[1])
    if args.plot_Z:
        import matplotlib.pyplot as plt
        Z1 = (Z1 + 1.)/2.
        Z2 = (Z2 + 1.)/2.
        print('Z1 max: {} min: {}'.format(np.max(Z1), np.min(Z1)))
        print('Z2 max: {} min: {}'.format(np.max(Z2), np.min(Z2)))
        ax = plt.subplot(2,1,1)
        plt.imshow(Z1[:, :args.num_words], cmap=plt.cm.Blues)
        plt.ylabel('Cell index')
        plt.xlabel('Input word index')
        plt.subplot(2,1,2)
        plt.ylabel('Cell index')
        plt.xlabel('Input word index')
        plt.imshow(Z2[:, :args.num_words], cmap=plt.cm.Blues)
        plt.show()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    with codecs.open(os.path.join(args.save_path, args.out_filename), 'w',
                     encoding='utf8') as outf:
        outf.write(out_text)

if __name__ == '__main__':
    tf.app.run()
