# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 17:46:33 2017

@author: marzipan
"""

import tensorflow as tf
import numpy as np

tf.reset_default_graph()

# tf Graph Input
X = tf.placeholder(tf.float32, shape=[None,10])
Y = tf.placeholder(tf.float32, shape=[None,10])

# Set model weights
W = tf.get_variable(name="weight",shape=[1,10])
B = tf.get_variable(name="bias",shape=[1,10])

a = tf.constant(np.arange(4*67).reshape(4,67))
b = tf.constant([4,9,9,11])

p = tf.constant([[0.1,0.4,0.5,0.0]]*1)
dist = tf.contrib.distributions.Multinomial(n=1., p=p)
# Tensorflow Init
sess = tf.Session()
sess.run(tf.global_variables_initializer())