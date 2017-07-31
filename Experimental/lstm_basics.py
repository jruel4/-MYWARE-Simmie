#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 17:54:17 2017

@author: marzipan
"""

#TODO get it to update!

import tensorflow as tf

# Config
input_size = 3
batch_size = 1
num_units = 2

# Tensors
lstm = tf.contrib.rnn.BasicLSTMCell(num_units)
state = lstm.zero_state(batch_size, tf.float32)
lstm_input = tf.placeholder(tf.float32, shape=(batch_size, input_size))

# Graph
output, state = lstm(lstm_input, state)

# Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Run
lstm_in_data = [[2,3,4]]
feed_dict = {lstm_input:lstm_in_data}
r_output, r_state = sess.run([output,state], feed_dict=feed_dict)

print(r_output, r_state[0])

















