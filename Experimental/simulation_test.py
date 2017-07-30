#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 21:00:00 2017

@author: marzipan
"""

# External 
import tensorflow as tf
import numpy as np

# Internal
from rnn_test_utils import pretty_print_output

# Config
num_cells = 3
input_size = 250
batch_size = 1
num_units = 10
max_time = 1
rnn_input_shape = (batch_size, max_time, input_size)
tf_dtype = tf.float32

# Cells
cells = [tf.contrib.rnn.BasicLSTMCell(num_units, state_is_tuple=True) for i in range(num_cells)]
rnn_inputs = tf.placeholder(tf_dtype, shape=rnn_input_shape)

# Multicell
multicell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

# RNN
outputs, last_states = tf.nn.dynamic_rnn( cell=multicell, inputs=rnn_inputs, dtype=tf_dtype)

# Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Input data
lstm_in_data = np.ones(rnn_input_shape)
feed_dict = {rnn_inputs:lstm_in_data}

# Run
r_outputs, r_states = sess.run([outputs, last_states], feed_dict=feed_dict)
pretty_print_output(r_outputs, r_states)

