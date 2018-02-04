# Goal, binary + digit encoder/decoders, +1, -2 ops

import tensorflow as tf
import numpy as np
import random

num_fuzzy_digit_cells = 12

def dense_relu_stack(input, units_list, final_activation=tf.nn.relu):
    output = input
    for i, units in enumerate(units_list):

        is_final_layer = i == len(units_list) - 1
        activation = final_activation if is_final_layer else tf.nn.relu

        output = tf.layers.dense(
            inputs=output,
            units=units,
            activation=activation)

    return output


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
binary_encoded = dense_relu_stack(binary_input, [64, 64, num_fuzzy_digit_cells])

# Encoder for the digit input
digit_encoded = dense_relu_stack(digit_input, [64, 64, num_fuzzy_digit_cells])


plus1_output = dense_relu_stack((binary_encoded + digit_encoded) / 2,
                                units_list=[num_fuzzy_digit_cells] * 2)

math_output = plus1_output


# Binary decoder
binary_decoded = dense_relu_stack(math_output, [64, 64, 4], final_activation=tf.nn.sigmoid)

binary_prediction = tf.argmax(binary_decoded)

binary_loss = tf.losses.mean_squared_error(
    predictions=binary_decoded,
    labels=binary_label)


# Digit decoder
digit_decoded = dense_relu_stack(math_output, [64, 64, 10], final_activation=tf.nn.sigmoid)

digit_prediction = tf.argmax(digit_decoded, 1)

digit_loss = tf.losses.mean_squared_error(
    predictions=digit_decoded,
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
        _, loss_value, digit_prediction_val, digit_label_val = \
            sess.run([train, total_loss, digit_prediction, digit_label], feed_dict)

        if i % 300 == 0 or i == 5 or i == 10 or i == 25 or i == 50:
            # print('loss @', i, ':', loss_value[64 - 1])
            print('loss @', i, ':', loss_value)

        if i % 1000 == 0:
            answers = np.argmax(batch['digit_input'], 1)
            print(digit_prediction_val - answers)


        if i > 20000 or loss_value < 1e-5:
            print('\ntraining finished!\n')
            print('loss @', i, ':', loss_value)
            print(digit_prediction_val - np.argmax(digit_label_val, 1))
            break

        i += 1
