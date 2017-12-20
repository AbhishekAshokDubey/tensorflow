import tensorflow as tf
import capslayers as cl

import flags
import importlib
importlib.reload(flags)

class CapsNet():
    def __init__(self, is_training=True, X_shape = (flags.batch_size, 28, 28, 1), Y_shape= (flags.batch_size, 10)):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.X = tf.placeholder(tf.float32, shape = X_shape, name="images")
            self.Y = tf.placeholder(tf.float32, shape = Y_shape, name="labels")
            self._build()
            if is_training:
                self._loss()
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer()
                self.train_op = self.optimizer.minimize(self.total_loss, global_step=self.global_step)

    def _build(self):
        with tf.variable_scope('Conv1_layer'):
            conv1 = tf.contrib.layers.conv2d(self.X, num_outputs=256,kernel_size=9, stride=1,
                                             padding='VALID', activation_fn=tf.nn.relu) # Conv1: [batch_size, 20, 20, 256]

        # Primary Capsules layer
        with tf.variable_scope('PrimaryCaps_layer'):
            primaryCaps = cl.PrimaryCapsLayer(num_outputs=32, vec_len=8, kernel_size=9, stride=2)
            caps1 = primaryCaps(conv1)             # caps1: [batch_size, 1152, 8, 1]

        # DigitCaps layer
        with tf.variable_scope('DigitCaps_layer'):
            digitCaps = cl.DigiCapsLayer(num_outputs=10, vec_len=16)
            self.caps2 = digitCaps(caps1)            # caps2: [batch_size, 10, 16, 1]


        ### Decoder structure in Fig. 2
        # masking for only the true label for decoder
        with tf.variable_scope('Masking'):
            self.masked_v = tf.matmul(tf.squeeze(self.caps2), tf.reshape(self.Y, (-1, 10, 1)), transpose_a=True)
            # self.masked_v: [batch_size, 16, 10] * [batch_size, 10, 1] = [batch_size, 16, 1]
            # self.masked_v is the vector output of the true label only for each instance in the batch
            self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2), axis=2, keep_dims=True) + flags.epsilon)
            # self.v_length: [batch_size, 10, 1, 1]

        # For Reconstructing the data with 3 FC layers and reconstruction loss
        # [batch_size, 16, 1]  [batch_size, 16]
        # [batch_size, 16] => [batch_size, 512] => [batch_size, 1024] => [batch_size, 784]
        with tf.variable_scope('Decoder'):
            vector_j = tf.reshape(self.masked_v, shape=(flags.batch_size, -1))
            fc1 = tf.contrib.layers.fully_connected(vector_j, num_outputs=512)
            fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
            self.decoded = tf.contrib.layers.fully_connected(fc2, num_outputs=784, activation_fn=tf.sigmoid)
            # self.decoded: [batch_size, 784]

    def _loss(self):
        ### The margin loss
        # self.v_length: [batch_size, 10, 1, 1]
        # max_l = max(0, m_plus-||v_c||)^2
        max_l = tf.squeeze(tf.square(tf.maximum(0., flags.m_plus - self.v_length)))
        # max_r = max(0, ||v_c||-m_minus)^2
        max_r = tf.squeeze(tf.square(tf.maximum(0., self.v_length - flags.m_minus)))
        # max_l, max_r : tf.squeeze([batch_size, 10, 1, 1]) = [batch_size, 10]

        T_c = self.Y         # Y: [batch_size, 10]
        # [batch_size, 10], element-wise multiply
        L_c = T_c * max_l + flags.lambda_val * (1 - T_c) * max_r
        self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

        ### The reconstruction loss
        orgin = tf.reshape(self.X, shape=(flags.batch_size, -1))
        squared = tf.square(self.decoded - orgin)
        # squared [batch_size, 784]
        self.reconstruction_err = tf.reduce_sum(tf.reduce_mean(squared, axis=0))

        # 3. Total loss = margin_loss + regularization_scale * reconstruction_err
        # regularization_scale as in paper = 0.0005
        self.total_loss = self.margin_loss + flags.regularization_scale * self.reconstruction_err