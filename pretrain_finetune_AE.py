import tensorflow as tf
import numpy as np
import math
#import pandas as pd
#import sys

datafile_path = r"/home/abhishek/Desktop/sensor_time_data.csv";
time_col_number = 0;
output_col_number = 2; # ignoring the date col number
train_fraction = 0.8;


#input = pd.read_csv(datafile_path, skipinitialspace=True, parse_dates=[0], infer_datetime_format=True);
#input = np.genfromtxt(datafile_path,dtype=float,delimiter = ',',names = True)
input = np.genfromtxt(datafile_path,dtype=float,delimiter = ',', skip_header=1)

time_col = input[:,time_col_number] # store date column separately
input = np.delete(input,np.s_[time_col_number],1) # remove date column for analysis

output = input[:,output_col_number] # output_col_number after removing date as output
input = np.delete(input,np.s_[output_col_number],1)

train_index = int(input.shape[0] * train_fraction)

train_x = input[:train_index,]
train_y = output[:train_index,]

test_x = input[train_index:,]
test_y = output[train_index:,]

train_x_mean = train_x.mean(0)
train_x_std = train_x.std(0)
train_y_mean = train_y.mean(0)
train_y_std = train_y.std(0)


input_data = np.divide((train_x - train_x_mean),train_x_std)
output_data = np.divide((train_y - train_y_mean),train_y_std)

input_data = np.nan_to_num(input_data)
output_data_ae = np.copy(input_data) # for autoencoder, will be used when we corrupt it a bit
output_data_nn = np.nan_to_num(output_data)

###################################################################
# Pretrain the Autoencoder with 2 hidden layer
n_samp, n_input = input_data.shape
n_hidden1 = n_input/2
n_hidden2 = n_input/4


x = tf.placeholder("float", [None, n_input])
# Weights and biases to hidden layer
ae_Wh1 = tf.Variable(tf.random_uniform((n_input, n_hidden1), -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input)))
ae_bh1 = tf.Variable(tf.zeros([n_hidden1]))
ae_h1 = tf.nn.tanh(tf.matmul(x,ae_Wh1) + ae_bh1)

ae_Wh2 = tf.Variable(tf.random_uniform((n_hidden1, n_hidden2), -1.0 / math.sqrt(n_hidden1), 1.0 / math.sqrt(n_hidden1)))
ae_bh2 = tf.Variable(tf.zeros([n_hidden2]))
ae_h2 = tf.nn.tanh(tf.matmul(ae_h1,ae_Wh2) + ae_bh2)

ae_Wh3 = tf.transpose(Wh2)
ae_bh3 = tf.Variable(tf.zeros([n_hidden1]))
ae_h1_O = tf.nn.tanh(tf.matmul(ae_h2,ae_Wh3) + ae_bh3)

ae_Wh4 = tf.transpose(Wh1)
ae_bh4 = tf.Variable(tf.zeros([n_input]))
ae_y_pred = tf.nn.tanh(tf.matmul(ae_h1_O,ae_Wh4) + ae_bh4)


ae_y_actual = tf.placeholder("float", [None,n_input])
cross_entropy = -tf.reduce_sum(ae_y_actual * tf.log(ae_y_pred))
meansq = tf.reduce_mean(tf.square(ae_y_actual - ae_y_pred))
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(meansq)
############################################################



##############################################################
# fine-tunning neural network 
nn_Wh1 = tf.Variable(tf.random_uniform((n_hidden2, 1), -1.0 / math.sqrt(n_hidden2), 1.0 / math.sqrt(n_hidden2)))
nn_bh1 = tf.Variable(tf.zeros([1]))
nn_y_pred = tf.nn.tanh(tf.matmul(ae_h2,nn_Wh1) + nn_bh1)


nn_y_actual = tf.placeholder("float", [None,1])
nn_cross_entropy = -tf.reduce_sum(nn_y_actual * tf.log(nn_y_pred))
nn_meansq = tf.reduce_mean(tf.square(nn_y_actual - nn_y_pred))
nn_train_step = tf.train.GradientDescentOptimizer(0.05).minimize(nn_meansq)
############################################################


##################################################################
# Run the autoencoder graph
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

n_rounds = 1000
batch_size = min(500, n_samp)

for i in range(n_rounds):
    sample = np.random.randint(n_samp, size=batch_size)
    batch_xs = input_data[sample][:]
    batch_ys = output_data_ae[sample][:]
    sess.run(train_step, feed_dict={x: batch_xs, ae_y_actual:batch_ys})
    if i % 100 == 0:
        print i, sess.run(cross_entropy, feed_dict={x: batch_xs, ae_y_actual:batch_ys}), sess.run(meansq, feed_dict={x: batch_xs, ae_y_actual:batch_ys})

output_data_nn = output_data_nn.reshape((output_data_nn.shape[0],1))

# Run the fine-tunning neural network
for i in range(n_rounds):
    sample = np.random.randint(n_samp, size=batch_size)
    batch_xs = input_data[sample][:]
    batch_ys = output_data_nn[sample][:]
    sess.run(nn_train_step, feed_dict={x: batch_xs, nn_y_actual:batch_ys})
    if i % 100 == 0:
        print i, sess.run(nn_meansq, feed_dict={x: batch_xs, nn_y_actual:batch_ys})

#########################################################################



##########################################################################
# predict the output:

test_input_data = np.divide((test_x - train_x_mean),train_x_std)
test_input_data = np.nan_to_num(test_input_data)

norm_predictions = sess.run(nn_y_pred, feed_dict={x: test_input_data})
predictions = (norm_predictions * train_y_std) + train_y_mean;

output_save = np.concatenate((test_y.reshape(test_y.shape[0],1), predictions), axis=1)
numpy.savetxt("output_sensor_time_date.csv", output_save, delimiter=",")
