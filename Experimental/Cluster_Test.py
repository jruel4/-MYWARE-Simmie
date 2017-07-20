# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 04:10:21 2017

@author: marzipan
"""

import tensorflow as tf
from Simmie.TensorFlow.Clusters.Cluster_0 import cluster_0_spec
#from Simmie.TensorFlow.Clusters.Cluster_0 import ip
ip = {
#    "thor"          : "localhost",
    "thor"          : "192.168.2.5",
    "black_velvet"  : "192.168.2.6",
    "tron"          : "192.168.2.8"
    }


server = tf.train.Server(cluster_0_spec, job_name="job_0", task_index=0)

#server = tf.train.Server(cluster_0_spec, job_name="job_0", task_index=1)








#tf.reset_default_graph()

with tf.device("/job:job_0/task:0"):

    
    w0 = tf.get_variable("99", shape=[100], dtype=tf.float32, initializer=tf.random_normal_initializer())
    w1 = tf.get_variable("87", shape=[100], dtype=tf.float32, initializer=tf.random_normal_initializer())

with tf.device("/job:job_0/task:1"):
    with tf.variable_scope("n", reuse=True):
        t0 = tf.get_variable("00", shape=[100], dtype=tf.float32, initializer=tf.random_normal_initializer())
        o0 = w0*w1
        w_upd0 = w0.assign(o0)
    
with tf.device("/job:job_0/task:0"):
    with tf.variable_scope("n", reuse=True):
        t1 = t0*t0
        o1 = t0.assign(t1)

sess_thor = tf.Session("grpc://"+ip['thor']+":2222")
sess_tron = tf.Session("grpc://"+ip['tron']+":2223")

sess_thor.run(tf.global_variables_initializer())
sess_tron.run(tf.global_variables_initializer())

b=sess_thor.run(w0)
a=sess_tron.run(w_upd0.op)
b=sess_thor.run(w0)
b=sess_thor.run(o1)