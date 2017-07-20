
import tensorflow as tf
import numpy as np

# Regressor Model 0 Hyperparameters
G_dense0_size = 10
G_dense1_size = 10


'''
MODEL
'''


# input flags
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":
    
    
with tf.device(tf.train.replica_device_setter(
		worker_device="/job:worker/task:%d" % FLAGS.task_index,
		cluster=cluster)):

		# count the number of updates
		global_step = tf.get_variable(
            'global_step',
            [],
            initializer = tf.constant_initializer(0),
			trainable = False)
    

# Unpack the input features and labels
state = tf.placeholder(tf.float32, shape=[None, 1])

# Dense Layer 1
dense0 = tf.layers.dense(inputs=state, units=G_dense0_size, kernel_initializer=tf.constant_initializer(1), activation=tf.sigmoid)
dense1 = tf.layers.dense(inputs=dense0, units=G_dense1_size, kernel_initializer=tf.constant_initializer(1), activation=tf.sigmoid)
output_layer = tf.layers.dense(inputs=dense1, units=1, activation=None)



# Loss
lbl=tf.placeholder(tf.float32,[None,1])
loss = tf.losses.mean_squared_error(
    labels=lbl,
    predictions=output_layer
    )