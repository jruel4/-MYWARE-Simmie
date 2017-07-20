# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 04:10:21 2017

@author: marzipan
"""

import tensorflow as tf

# Create cluster spec
cluster_tst_spec = tf.train.ClusterSpec({
        "job_0": ["localhost:2222", "localhost:2223"]
        })

#