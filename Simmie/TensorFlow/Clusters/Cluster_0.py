'''
Top and only cluster for Simmie.
'''

import tensorflow as tf

ip = {
#    "thor"          : "localhost",
    "thor"          : "192.168.2.5",
    "black_velvet"  : "192.168.2.6",
    "tron"          : "192.168.2.8"
    }

# Create cluster spec
cluster_0_spec = tf.train.ClusterSpec({
        "job_0": [
                ip['thor']+":2222",
                ip['tron']+":2223"
        ]
        })



#