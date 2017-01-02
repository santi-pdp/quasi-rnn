import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer



class QRNN_pooling(tf.nn.rnn_cell.RNNCell):

    def __init__(self, out_fmaps, pool_type):
        self.__pool_type = pool_type
        self.__out_fmaps = out_fmaps

    @property
    def state_size(self):
        return self.__out_fmaps

    @property
    def output_size(self):
        return self.__out_fmaps

    def __call__(self, inputs, state, scope=None):
        """
        inputs: 2-D tensor of shape [batch_size, Zfeats + [gates]]
        """
        pool_type = self.__pool_type
        # print('QRNN pooling inputs shape: ', inputs.get_shape())
        # print('QRNN pooling state shape: ', state.get_shape())
        with tf.variable_scope(scope or "QRNN-{}-pooling".format(pool_type)):
            if pool_type == 'f':
                # extract Z activations and F gate activations
                Z, F = tf.split(1, 2, inputs)
                # return the dynamic average pooling
                output = tf.mul(F, state) + tf.mul(tf.sub(1., F), Z)
                return output, output
            elif pool_type == 'fo':
                # extract Z, F gate and O gate
                Z, F, O = tf.split(1, 3, inputs)
                new_state = tf.mul(F, state) + tf.mul(tf.sub(1., F), Z)
                output = tf.mul(O, new_state)
                return output, new_state
            elif pool_type == 'ifo':
                # extract Z, I gate, F gate, and O gate
                Z, I, F, O = tf.split(1, 4, inputs)
                new_state = tf.mul(F, state) + tf.mul(I, Z)
                output = tf.mul(O, new_state)
                return output, new_state
            else:
                raise ValueError('Pool type must be either f, fo or ifo')



class QRNN_layer(object):
    """ Quasi-Recurrent Neural Network Layer
        (cf. https://arxiv.org/abs/1611.01576)
    """
    def __init__(self, input_, out_fmaps, fwidth=2,
                 activation=tf.tanh, pool_type='fo', zoneout=0.1, infer=False,
                 name='QRNN'):
        """
        pool_type: can be f, fo, or ifo
        zoneout: > 0 means apply zoneout with p = 1 - zoneout
        """
        self.out_fmaps = out_fmaps
        self.activation = activation
        self.name = name
        self.infer = infer

        batch_size = input_.get_shape().as_list()[0]
        with tf.variable_scope(name):
            # gates: list containing gate activations (num of gates depending
            # on pool_type)
            Z, gates = self.convolution(input_, fwidth, out_fmaps, pool_type,
                                        zoneout)
            # join all features (Z and gates) into Tensor at dim 2 merged
            T = tf.concat(2, [Z] + gates)
            # create the pooling layer
            pooling = QRNN_pooling(out_fmaps, pool_type)
            self.initial_state = pooling.zero_state(batch_size=batch_size,
                                                    dtype=tf.float32)
            # encapsulate the pooling in the iterative dynamic_rnn
            H, last_C = tf.nn.dynamic_rnn(pooling, T,
                                          initial_state=self.initial_state)
            self.h = H
            self.last_state = last_C

    def reset_states(self, sess):
        sess.run(self.initial_state)

    def convolution(self, input_, filter_width, out_fmaps, pool_type, zoneout):
        """ Applies 1D convolution along time-dimension (T) assuming input
            tensor of dim (batch_size, T, n) and returns
            (batch_size, T, out_fmaps)
            zoneout: regularization (dropout) of F gate
        """
        in_shape = input_.get_shape()
        in_fmaps = in_shape[-1]
        num_gates = len(pool_type)
        gates = []
        # pad on the left to mask the convolution (make it causal)
        pinput = tf.pad(input_, [[0, 0], [filter_width - 1, 0], [0, 0]])
        with tf.variable_scope('convolutions'):
            Wz = tf.get_variable('Wz', [filter_width, in_fmaps, out_fmaps],
                                 initializer=xavier_initializer())
            z_a = tf.nn.conv1d(pinput, Wz, stride=1, padding='VALID')
            z = self.activation(z_a)
            # compute gates convolutions
            for gate_name in pool_type:
                Wg = tf.get_variable('W{}'.format(gate_name),
                                     [filter_width, in_fmaps, out_fmaps],
                                     initializer=xavier_initializer())
                g_a = tf.nn.conv1d(pinput, Wg, stride=1, padding='VALID')
                g = tf.sigmoid(g_a)
                if not self.infer and zoneout > 0 and gate_name == 'f':
                    print('Applying zoneout if {} to gate F'.format(zoneout))
                    # appy dropout to F
                    g = 1. - tf.nn.dropout((1. - g), 1. - zoneout)
                gates.append(g)
        return z, gates
