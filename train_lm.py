from __future__ import print_function
import tensorflow as tf
from qrnn import QRNN_layer
import numpy as np
from model import QRNN_lm
from data_loader import ptb_batch_loader
import timeit
import json
import os


flags = tf.app.flags
flags.DEFINE_integer("epoch", 72, "Epochs to train (Def: 72).")
flags.DEFINE_integer("batch_size", 20, "Batch size (Def: 20).")
flags.DEFINE_integer("seq_len", 105, "Max sequences length. "
                                       " Specified at bucketizing (Def: 105).")
flags.DEFINE_integer("save_every", 100, "Batch frequency to save model and "
                                        "summary (Def: 100).")
flags.DEFINE_integer("qrnn_size", 640, "Number of qrnn units per layer "
                                       "(Def: 640).")
flags.DEFINE_integer("qrnn_layers", 2, "Number of qrnn layers (Def: 2). ")
flags.DEFINE_integer("qrnn_k", 2, "Width of QRNN filter (Def: 2). ")
flags.DEFINE_integer("emb_dim", 640, "Embedding dimension (Def: 650). ")
flags.DEFINE_integer("vocab_size", 10001, "Num words in vocab (Def: 10001). ")
flags.DEFINE_float("zoneout", 0.1, "Apply zoneout (dropout) to F gate (Def: 0.1)")
flags.DEFINE_float("dropout", 0.5, "Apply dropout in hidden layers (Def: 0.5)")
flags.DEFINE_float("learning_rate", 1., "Beginning learning rate (Def: 1).")
flags.DEFINE_float("learning_rate_decay", 0.95, "After 6th epoch this "
                                                "factor is applied (Def: 0.95)")
flags.DEFINE_float("grad_clip", 10., "Clip norm value (Def: 10).")
flags.DEFINE_string("save_path", "lm-qrnn_model", "Save path "
                                                  "(Def: lm-qrnn_model).")
flags.DEFINE_string("data_dir", "data/ptb", "Data dir containing train/valid"
                                            "/test.txt files (Def: lm-qrnn_"
                                            "model).")
flags.DEFINE_boolean("train", True, "Flag for training (Def: True).")
flags.DEFINE_boolean("test", True, "Flag for testing (Def: True).")


FLAGS = flags.FLAGS

def main(_):
    args = FLAGS
    print('Parsed options: ')
    print(json.dumps(args.__flags, indent=2))

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    bloader = ptb_batch_loader(args.data_dir, args.batch_size, args.seq_len)
    qrnn_lm = QRNN_lm(args)
    if args.train:
        train(qrnn_lm, bloader, args)
    if args.test:
        test(qrnn_lm, bloader, args)

def evaluate(sess, lm_model, loader, args, split='valid'):
    """ Evaluate an epoch over valid or test splits """
    val_loss = []
    batches_per_epoch = loader.batches_per_epoch[split]
    batch_i = 0
    # init states to zero
    states = [qrnn_.initial_state.eval() for qrnn_ in lm_model.qrnns]
    for batchX, batchY in loader.next_batch(split):
        fdict = {lm_model.words_in: batchX, lm_model.words_gtruth: batchY}
        # feed last states, this way it's stateful between batches
        for state, init_state in zip(states, lm_model.initial_states):
            fdict.update({init_state: state})
        loss = sess.run(lm_model.loss, feed_dict=fdict)
        val_loss.append(loss)
        batch_i += 1
        if (batch_i + 1) >= batches_per_epoch:
            break

    m_val_loss = np.mean(val_loss)
    print("{} split mean loss: {}, perplexity: {}".format(split, m_val_loss,
                                                          np.exp(m_val_loss)))
    return m_val_loss

def test(lm_model, loader, args):
    # load the model from checkpoint
    with tf.Session() as sess:
        if not lm_model.load(sess, args.save_path):
            raise ValueError('Could not load the saved model!')
        pass


def train(lm_model, loader, args):
    def train_epoch(sess, epoch_idx, writer, merger, saver, save_path):
        """ Train a single epoch """
        tr_loss = []
        b_timings = []
        batches_per_epoch = loader.batches_per_epoch['train']
        batch_i = 0
        # init states to zero
        states = [qrnn_.initial_state.eval() for qrnn_ in lm_model.qrnns]
        for batchX, batchY in loader.next_batch('train'):
            beg_t = timeit.default_timer()
            fdict = {lm_model.words_in: batchX, lm_model.words_gtruth:batchY}
            # feed last states, this way it's stateful between batches
            for state, init_state in zip(states, lm_model.initial_states):
                fdict.update({init_state: state})
            loss, states, _, summary = sess.run([lm_model.loss,
                                                lm_model.last_states,
                                                lm_model.train_op,
                                                merger],
                                                feed_dict=fdict)
            tr_loss.append(loss)
            b_timings.append(timeit.default_timer() - beg_t)
            if batch_i % args.save_every == 0:
                writer.add_summary(summary, epoch_idx * batches_per_epoch + batch_i)
                checkpoint_file = os.path.join(save_path, 'model.ckpt')
                global_step = epoch_idx * batches_per_epoch + batch_i
                lm_model.save(sess, checkpoint_file, global_step)
                #saver.save(sess, checkpoint_file,
                #           global_step=epoch_idx * batches_per_epoch + batch_i)
                print("%4d/%4d (epoch %2d) tr_loss: %2.6f "
                      "mtime/batch: %2.6fs" % (batch_i, batches_per_epoch,
                                               epoch_idx, loss,
                                               np.mean(b_timings)))
            batch_i += 1
            if (batch_i + 1) >= batches_per_epoch:
                break
        return np.mean(tr_loss)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.allow_soft_placement=True
    with tf.Session(config=config) as sess:
        try:
            tf.global_variables_initializer().run()
            merged = tf.summary.merge_all()
        except AttributeError:
            # Backward compatibility
            tf.initialize_all_variables().run()
            merged = tf.merge_all_summaries()
        curr_lr = args.learning_rate
        saver = tf.train.Saver()
        train_writer = tf.train.SummaryWriter(os.path.join(args.save_path,
                                                           'train'),
                                              sess.graph)
        for epoch_idx in range(args.epoch):
            epoch_loss = train_epoch(sess, epoch_idx, train_writer,
                                     merged, saver, args.save_path)
            print('End of epoch {} with avg loss {} and '
                  'perplexity {}'.format(epoch_idx,
                                         epoch_loss,
                                         np.exp(epoch_loss)))
            if epoch_idx >= 5:
                curr_lr = curr_lr * args.learning_rate_decay
                decay_op = lm_model.lr.assign(curr_lr)
                sess.run(decay_op)
            val_loss = evaluate(sess, lm_model, loader, args)



if __name__ == '__main__':
    tf.app.run()
