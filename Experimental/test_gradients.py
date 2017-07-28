# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 17:46:33 2017

@author: marzipan
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 13:54:29 2017

@author: marzipan
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 13:53:26 2017

@author: marzipan
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 06:02:07 2017

@author: marzipan
"""

import tensorflow as tf
import numpy as np


# Directories
G_logs = 'C:\\Users\\marzipan\\workspace\\Simmie\\Experimental\\Logs\\MiscTest\\'
G_logdir = G_logs + 'A1\\'

# tf Graph Input
X = tf.placeholder(tf.float32, shape=[None,10])
Y = tf.placeholder(tf.float32, shape=[None,10])

# Set model weights
W = tf.get_variable(name="weight",shape=[1,10])
B = tf.get_variable(name="bias",shape=[1,10])

W2 = W[0,tf.cast(X[0:1],tf.int32)]

a = tf.constant(np.arange(4*67).reshape(4,67))
b = tf.constant([4,9,9,11])

# Construct a linear model
pred = tf.add(X*W, tf.cast(B,tf.float32))

W2 = tf.contrib.layers.fully_connected(inputs=X*W, num_outputs=5, scope='val_dense')

pg = tf.gradients(W2,W)

opt=tf.train.GradientDescentOptimizer(0.1)

opt.compute_gradients()

# Tensorflow Init
sess = tf.Session()
summary_writer = tf.summary.FileWriter(G_logdir, sess.graph)
sess.run(tf.global_variables_initializer())