'''
Top and only cluster for Simmie.
'''

import tensorflow as tf

# Create cluster spec
cluster_0_spec = tf.train.ClusterSpec({"job_0": ["localhost:2222", "localhost:2223"]})

#