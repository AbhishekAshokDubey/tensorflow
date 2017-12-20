import tensorflow as tf
import numpy as np

import flags
import importlib
importlib.reload(flags)

def squash(caps_vector):
    # caps_vector: [batch_size, num_capsules, caps_vec_len, 1] for primaryCapsLayer
    # caps_vector: [batch_size, 1, num_capsules, caps_vec_len, 1] for DigiCapsLayer
    vec_squared_norm = tf.reduce_sum(tf.square(caps_vector), -2, keep_dims=True)
    # -2 for squashing along 'caps_vec_len' dimesion
    squashed_vector = ((vec_squared_norm / (1 + vec_squared_norm)) / tf.sqrt(vec_squared_norm + flags.epsilon)) * caps_vector
    return squashed_vector

def routing(layer_input, b_IJ, output_vec_len, num_outputs):
    # output_vec_len: 16
    # layer_input: [batch_size, 1152, 8, 1]
    # stack input 10 (num_outputs) times to [batch_size, 1152, 10, 8, 1]
    layer_input = tf.reshape(layer_input, shape=(flags.batch_size, -1, 1, layer_input.shape[-2].value, 1))
    layer_input = tf.tile(layer_input, [1, 1, num_outputs, 1, 1])
    # W: [num_caps_i, num_caps_j, len_u_i, len_v_j]
    # [1, 1152, 10, 8, 16]
    W = tf.get_variable('Weight', shape=(1, layer_input.shape[1].value, layer_input.shape[2].value,
                        layer_input.shape[3].value, output_vec_len), dtype=tf.float32,
                        initializer=tf.random_normal_initializer(stddev=flags.stddev))
    # stack W: for each input in batch to [batch_size, 1152, 10, 8, 16]
    W = tf.tile(W, [flags.batch_size, 1, 1, 1, 1])

    # Note: 'u' = layer_input according to the paper and so 'u_hat' will be
    u_hat = tf.matmul(W, layer_input, transpose_a=True)
    # u_hat:  tf.matmul(W, layer_input, transpose_a=True)
    # u_hat: [batch_size, 1152, 10, 8, 16].T * [batch_size, 1152, 10, 8, 1] = [16,8] * [8, 1] = [16, 1]
    # u_hat: [batch_size, 1152, 10, 16, 1]

    # W (u_hat) will be updated only once in last iteration with back-prop
    # c_IJ, routing weights will be updated in all but last iterations, but we wun update W (hence u_hat_stopped) here
    u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')
    
    # Starts the routing algoritm from the paper, from line 3
    for r_iter in range(flags.iter_routing):
        with tf.variable_scope('iter_' + str(r_iter)):
            
            # Note b_IJ: all zero of dim [batch_size, 1152, 10, 1, 1]
            c_IJ = tf.nn.softmax(b_IJ, dim=2)

            # At last iteration, use `u_hat` in order to receive gradients from the following graph
            if r_iter == flags.iter_routing - 1:
                s_J = tf.multiply(c_IJ, u_hat)
                # s_J: [batch_size, 1152, 10, 16, 1]
                # c_IJ * u_hat: [batch_size, 1152, 10, 1, 1] * [batch_size, 1152, 10, 16, 1]
                # c_IJ will be broadcasted to [batch_size, 1152, 10, 16, 1] for elementwise mul.
                
                # then sum along the input caps dim, to get the linear combination of all vectors
                s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
                v_J = squash(s_J)
                # s_J, v_J: [batch_size, 1, 10, 16, 1]

            elif r_iter < flags.iter_routing - 1:  # Inner iterations, do not apply backpropagation
                s_J = tf.multiply(c_IJ, u_hat_stopped)
                s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
                v_J = squash(s_J)

                # line 7 of algo: Dot product of v_j and u_j|i
                # as we have one v_j from linear combination of 1152 vectors u_hat
                # reshape & tile v_j from [batch_size ,1, 10, 16, 1] to [batch_size, 1152, 10, 16, 1] for dot product
                v_J_tiled = tf.tile(v_J, [1, layer_input.shape[1].value, 1, 1, 1])

                u_produce_v = tf.matmul(u_hat_stopped, v_J_tiled, transpose_a=True)
                # [batch_size, 1152, 10, 16, 1].T *  [batch_size, 1152, 10, 16, 1]
                # [batch_size, 1152, 10, 1, 16].T *  [batch_size, 1152, 10, 16, 1]
                # u_produce_v: [batch_size, 1152, 10, 1, 1]

                b_IJ += u_produce_v
    return(v_J)

class PrimaryCapsLayer():
    def __init__(self, num_outputs = 32, vec_len = 8, kernel_size = 9, stride = 2):
        self.num_outputs = num_outputs
        self.vec_len = vec_len
        self.kernel_size = kernel_size
        self.stride = stride

    def __call__(self, layer_input):
        # input shape = [batch_size, 20, 20, 256]
        capsules = tf.contrib.layers.conv2d(layer_input, self.vec_len * self.num_outputs,
                                    self.kernel_size, self.stride, padding="VALID",
                                    activation_fn=tf.nn.relu)
        # [batch_size, 20, 20, 256] -> conv2d(8 * 32, 9, 2) -> [batch_size, 6, 6, 8 * 32]
        # [batch_size, 20, 20, 256] -> 8 of conv2d(32, 9, 2) -> 32 of [batch_size, 6, 6, 8]
        capsules = tf.reshape(capsules, (flags.batch_size, -1, self.vec_len, 1))
        # reshape: [batch_size, 6, 6, 8 * 32] => [batch_size, 1152, 8, 1]
        capsules = squash(capsules)
        # Normalise (kind of) along vector dimesion        
        return capsules

class DigiCapsLayer():
    def __init__(self, num_outputs, vec_len):
        self.num_outputs = num_outputs #10
        self.vec_len = vec_len #16

    def __call__(self, layer_input):
        # layer_input: [batch_size, 1152, 8, 1]
        b_IJ = tf.constant(np.zeros([flags.batch_size, layer_input.shape[1].value, self.num_outputs, 1, 1], dtype=np.float32))
        # b_IJ: [batch_size, 1152, 10, 1, 1] and not [1, 1152, 10, 1, 1]
        # it should be different for each instant in an sample
        # check: https://github.com/naturomics/CapsNet-Tensorflow/issues/21
        capsules = routing(layer_input, b_IJ, self.vec_len, self.num_outputs)
        capsules = tf.squeeze(capsules, axis=1)
        return capsules