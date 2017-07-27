# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 17:03:33 2017

@author: marzipan
"""

import tensorflow as tf

G_logs = 'C:\\Users\\marzipan\\workspace\\Simmie\\Simmie\\Logs\\'
G_logdir = G_logs + 'S5\\'

class GBase:
    def __init__(self, _graph):
        self.g = _graph
        with self.g.as_default():
            with tf.name_scope("in"):
                self.i0 = tf.placeholder(tf.float32, shape=[], name="IN_EEG")
                
            with tf.name_scope("nn"):
                self.d0 = tf.contrib.layers.fully_connected(
                        inputs=self.i0,
                        num_outputs=1,
                        activation_fn=tf.sigmoid,
                        scope='d0')
                self.l0 = 1-self.d0
                self.tgt_step = tf.Variable(0, name='TGT_Step', trainable=False)
                self.tgt_optimizer = tf.train.RMSPropOptimizer(learning_rate=0.1, centered=False, decay=0.8)
                self.tgt_train_op = self.tgt_optimizer.minimize(self.l0, global_step=self.tgt_step)

class G0(GBase):
    def __init__(self, _graph):
        super().__init__(_graph)
        with self.g.as_default():
            with tf.name_scope("Base"):
                self.d=self.c + 1
                self.c=self.c + 1

class G1(GBase):
    def __init__(self, _graph):
        super().__init__(_graph)
        with self.g.as_default():
            with tf.name_scope("Base"):
                self.q=self.c


gg= tf.get_default_graph()
q=GBase(gg)
w=G0(gg)
e=G1(gg)

sess = tf.Session()
summary_writer = tf.summary.FileWriter(G_logdir, sess.graph)

print( sess.run(w.c) )
print( sess.run(e.q) )