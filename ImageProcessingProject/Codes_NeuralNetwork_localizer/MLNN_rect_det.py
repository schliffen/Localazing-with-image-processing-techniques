#
# neural net based localizer 02
# imports -----------------
import tensorflow as tf
import matplotlib.pyplot as plt
import requests
import numpy as np
#

sess = tf.Session()

# preparing the data  .................

path_1 = '/home/'
data_file = path_1
y_vals = np.array()
x_vals = np.array()

# repeatability  -------------------------

seed = 3
tf.set_random_seed(seed)
np.random.seed(seed)
batch_size = 10

# train test split --------------------

train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))

x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]


#  normalization for convergence --------------------------------

def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m-col_min) / (col_max - col_min)

x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

# generating initial weights and biases

def init_weight(shape, st_dev):
    weight = tf.Variable(tf.random_normal(shape, stddev=st_dev))
    return(weight)

def init_bias(shape, st_dev):
    bias = tf.Variable(tf.random_normal(shape, stddev=st_dev))
    return(bias)

# Initialize our placeholders. There will an input Image and two output, diagonal corners:

x_data = tf.placeholder(shape=[None, None], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 2], dtype=tf.float32)

# creating fully connected layers -------------------

def fully_connected(input_layer, weights, biases):
    layer = tf.add(tf.matmul(input_layer, weights), biases)
    return(tf.nn.relu(layer))

# Create second layer (25 hidden nodes)
weight_1 = init_weight(shape=[None, 25], st_dev=10.0)
bias_1   = init_bias(shape=[25], st_dev=10.0)
layer_1  = fully_connected(x_data, weight_1, bias_1)

# Create second layer (10 hidden nodes)
weight_2 = init_weight(shape=[25, 10], st_dev=10.0)
bias_2   = init_bias(shape=[10], st_dev=10.0)
layer_2  = fully_connected(layer_1, weight_2, bias_2)

# Create third layer (3 hidden nodes)
weight_3 = init_weight(shape=[10, 3], st_dev=10.0)
bias_3   = init_bias(shape=[3], st_dev=10.0)
layer_3  = fully_connected(layer_2, weight_3, bias_3)

# Create output layer (1 output value)
weight_4 = init_weight(shape=[3, 2], st_dev=10.0)
bias_4 = init_bias(shape=[2], st_dev=10.0)
final_output = fully_connected(layer_3, weight_4, bias_4)

# creating loss function
loss = tf.reduce_mean(tf.abs(y_target - final_output))
my_opt = tf.train.AdamOptimizer(0.05)
train_step = my_opt.minimize(loss)
init = tf.initialize_all_variables()
sess.run(init)

#
# Initialize the loss vectors, computation and reports
loss_vec = []
test_loss = []
for i in range(200):
    # Choose random indices for batch selection
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    # Get random batch
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])
    # Run the training step
    sess.run(train_step, feed_dict={x_data: rand_x, y_target:rand_y})
    # Get and store the train loss
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)
    # Get and store the test loss
    test_temp_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
    test_loss.append(test_temp_loss)
    if (i+1)%25==0:
        print('Generation: ' + str(i+1) + '. Loss = ' + str(temp_loss))


plt.plot(loss_vec, 'k-', label='Train Loss')
plt.plot(test_loss, 'r--', label='Test Loss')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()
