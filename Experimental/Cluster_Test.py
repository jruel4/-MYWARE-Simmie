# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 04:10:21 2017

@author: marzipan
"""

import tensorflow as tf
from Simmie.TensorFlow.Clusters.Cluster_0 import cluster_0_spec
#from Simmie.TensorFlow.Clusters.Cluster_0 import ip
ip = {
    "thor"          : "localhost",
#    "thor"          : "192.168.2.5",
    "black_velvet"  : "192.168.2.6",
    "tron"          : "192.168.2.8"
    }


server = tf.train.Server(cluster_0_spec, job_name="job_0", task_index=0)

#server = tf.train.Server(cluster_0_spec, job_name="job_0", task_index=1)

with tf.device("/job:job_0/task:0"):
    w0 = tf.get_variable("w0", shape=[100], dtype=tf.float32, initializer=tf.random_normal_initializer())
    w1 = tf.get_variable("w1", shape=[100], dtype=tf.float32, initializer=tf.random_normal_initializer())

with tf.device("/job:job_0/task:0"):
    o0 = w0*w1
    w_upd0 = w0.assign(o0)
    
with tf.Session("grpc://"+ip['thor']+":2222") as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(w_upd0.op)