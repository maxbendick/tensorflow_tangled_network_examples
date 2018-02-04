import tensorflow as tf
import numpy as np
import random

num_fuzzy_digit_cells = 12

plus1_index = 0
times2_index = 1

math_ops_to_learn = [
    lambda x: (x + 1) % 10,
    lambda x: (x * 2) % 10,
]
num_ops = len(math_ops_to_learn)

binary_encoder_index = 0
digit_encoder_index = 1


def num_to_bin_arr(a):

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
        math_ops     = np.zeros((batch_size, num_ops), dtype=np.bool)
        encoders     = np.zeros((batch_size, 2), dtype=np.bool)

        for i in range(batch_size):
            x = random.randint(0, 9)

            op_index = random.randint(0, num_ops - 1)

            y = math_ops_to_learn[op_index](x)
            math_ops[i, op_index] = True

            encoder_index = random.randint(0, 1)
            encoders[i, encoder_index] = True

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
            'math_op':      math_ops,
            'encoder':      encoders,
        }



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

math_op_selector = tf.placeholder(
    dtype=tf.bool,
    shape=[None, num_ops],
    name='math_op_selector')

encoder_selector = tf.placeholder(
    dtype=tf.bool,
    shape=[None, 2],
    name='encoder_selector')

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

# Select an encoder because we can only train one at a time
math_input = tf.where(encoder_selector[:,binary_encoder_index], binary_encoded, digit_encoded)

plus1_output = dense_relu_stack(math_input,
                                units_list=[num_fuzzy_digit_cells] * 2)

times2_output = dense_relu_stack(math_input,
                                 units_list=[num_fuzzy_digit_cells] * 2)

math_output = tf.where(math_op_selector[:,plus1_index], plus1_output, times2_output)


# Binary decoder
binary_decoded = dense_relu_stack(math_output, [64, 64, 4], final_activation=tf.nn.sigmoid)

# Digit decoder
digit_decoded = dense_relu_stack(math_output, [64, 64, 10], final_activation=tf.nn.sigmoid)

digit_prediction = tf.argmax(digit_decoded, 1)

binary_loss = tf.losses.mean_squared_error(
    predictions=binary_decoded,
    labels=binary_label)

digit_loss = tf.losses.mean_squared_error(
    predictions=digit_decoded,
    labels=digit_label)

# Add the losses. The partial derivative on each will be 1x, so both will be minimized.
total_loss = binary_loss + digit_loss

train = tf.train.AdamOptimizer().minimize(total_loss)


with tf.Session() as sess:
    tf.global_variables_initializer().run()

    i = 0
    for batch in data_batches():
        feed_dict = {
            binary_input: batch['binary_input'],
            digit_input:  batch['digit_input'],
            binary_label: batch['binary_label'],
            digit_label:  batch['digit_label'],
            math_op_selector: batch['math_op'],
            encoder_selector: batch['encoder'],
        }
        _, loss_value, digit_prediction_val = \
            sess.run([train, total_loss, digit_prediction], feed_dict)

        if i % 500 == 0:
            print('loss @ batch', i, ':', loss_value)

        if i >= 5000:
            print('\ntraining finished!')
            answers = np.argmax(batch['digit_input'], 1)
            print('error on digit outputs', digit_prediction_val - answers)
            break

        i += 1
