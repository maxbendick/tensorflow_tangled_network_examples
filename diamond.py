# Goal, binary + digit encoder/decoders, +1, -2 ops

import tensorflow as tf
import numpy as np
import random

num_fuzzy_digit_cells = 12

binary_input = tf.placeholder(
    dtype=tf.float32,
    shape=[None, 4],
    name='binary_input')

digit_input = tf.placeholder(
    dtype=tf.float32,
    shape=[None, 10],
    name='digit_input')

binary_label = tf.placeholder(
    dtype=tf.float32,
    shape=[None, 4],
    name='binary_label')

digit_label = tf.placeholder(
    dtype=tf.float32,
    shape=[None, 10],
    name='digit_label')


# Encoder for the binary input
binary_encoded = tf.layers.dense(
    inputs=binary_input,
    units=64,
    activation=tf.nn.relu)
binary_encoded = tf.layers.dense(
    inputs=binary_encoded,
    units=64,
    activation=tf.nn.relu)
binary_encoded = tf.layers.dense(
    inputs=binary_encoded,
    units=64,
    activation=tf.nn.relu)
binary_encoded = tf.layers.dense(
    inputs=binary_encoded,
    units=num_fuzzy_digit_cells,
    activation=tf.nn.relu)


# Encoder for the digit input
digit_encoded = tf.layers.dense(
    inputs=digit_input,
    units=64,
    activation=tf.nn.relu)
digit_encoded = tf.layers.dense(
    inputs=digit_encoded,
    units=64,
    activation=tf.nn.relu)
digit_encoded = tf.layers.dense(
    inputs=digit_encoded,
    units=64,
    activation=tf.nn.relu)
digit_encoded = tf.layers.dense(
    inputs=digit_encoded,
    units=num_fuzzy_digit_cells,
    activation=tf.nn.relu)


# Math ops, just plus1 for now
plus1_output = (binary_encoded + digit_encoded) / 2
plus1_output = tf.layers.dense(
    inputs=plus1_output,
    units=num_fuzzy_digit_cells,
    activation=tf.nn.relu)
plus1_output = tf.layers.dense(
    inputs=plus1_output,
    units=num_fuzzy_digit_cells,
    activation=tf.nn.relu)
plus1_output = tf.layers.dense(
    inputs=plus1_output,
    units=num_fuzzy_digit_cells,
    activation=tf.nn.relu)
math_output = plus1_output


# Binary decoder
binary_decoded = tf.layers.dense(
    inputs=math_output,
    units=64,
    activation=tf.nn.relu)
binary_decoded = tf.layers.dense(
    inputs=binary_decoded,
    units=64,
    activation=tf.nn.relu)
binary_decoded = tf.layers.dense(
    inputs=binary_decoded,
    units=64,
    activation=tf.nn.relu)
binary_decoded = tf.layers.dense(
    inputs=binary_decoded,
    units=4,
    activation=tf.nn.sigmoid)

binary_prediction = tf.argmax(binary_decoded)

binary_loss = tf.losses.mean_squared_error(
    predictions=binary_decoded,
    labels=binary_label)

# binary_loss = tf.nn.sigmoid_cross_entropy_with_logits(
#     logits=binary_decoded,
#     labels=binary_label)

# binary_loss = tf.nn.softmax_cross_entropy_with_logits(
#     logits=binary_decoded,
#     labels=binary_label)


# Digit decoder
digit_decoded = tf.layers.dense(
    inputs=math_output,
    units=64,
    activation=tf.nn.relu)
digit_decoded = tf.layers.dense(
    inputs=digit_decoded,
    units=64,
    activation=tf.nn.relu)
digit_decoded = tf.layers.dense(
    inputs=digit_decoded,
    units=64,
    activation=tf.nn.relu)
digit_decoded = tf.layers.dense(
    inputs=digit_decoded,
    units=10,
    activation=tf.nn.sigmoid)

digit_prediction = tf.argmax(digit_decoded, 1)
digit_loss = tf.nn.softmax_cross_entropy_with_logits(
    logits=digit_decoded,
    labels=digit_label)



total_loss = (binary_loss + digit_loss) / 2

train = tf.train.AdamOptimizer().minimize(total_loss)


def num_to_bin_arr(a) -> np.ndarray:

    # String 4 characters long of zeros and ones, representing binary
    bin_string = "{0:b}".format(a).rjust(4, '0')

    res = np.zeros((4), dtype=np.float32)

    for i in range(4):
       res[i] = 1. if bin_string[i] == '1' else 0.

    return res



def data_batches(batch_size=64):
    while True:

        binary_input = np.zeros((batch_size, 4), dtype=np.float32)
        binary_label = np.zeros((batch_size, 4), dtype=np.float32)
        digit_input  = np.zeros((batch_size, 10), dtype=np.float32)
        digit_label  = np.zeros((batch_size, 10), dtype=np.float32)


        for i in range(batch_size):

            x = random.randint(0, 9)
            y = (x + 1) % 10

            x_bin = num_to_bin_arr(x)
            y_bin = num_to_bin_arr(y)

            binary_input[i] = x_bin
            binary_label[i] = y_bin
            digit_input[i, y] = 1.
            digit_label[i, y] = 1.

        yield {
            'binary_input': binary_input,
            'digit_input':  digit_input,
            'binary_label': binary_label,
            'digit_label':  digit_label,
        }


with tf.Session() as sess:
    tf.global_variables_initializer().run()

    i = 0
    for batch in data_batches():
        feed_dict = {
            binary_input: batch['binary_input'],
            digit_input:  batch['digit_input'],
            binary_label: batch['binary_label'],
            digit_label:  batch['digit_label'],
        }
        _, loss_value, digit_prediction_val = sess.run([train, total_loss, digit_prediction], feed_dict)

        if i % 300 == 0 or i == 5 or i == 10 or i == 25 or i == 50:
            print('loss @', i, ':', loss_value[64 - 1])

        if i % 1000 == 0:
            answers = np.argmax(batch['digit_input'], 1)
            # print(answers)
            # print(digit_prediction_val)
            print(digit_prediction_val - answers)


        if i > 10000:
            break

        i += 1
