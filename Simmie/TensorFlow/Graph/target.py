# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 16:47:27 2017

@author: marzipan
"""

##
# Target parameters
##
tgt_output_units = shape_rpv[1]
tgt_layers = [50,50,tgt_output_units]
tgt_lr = 1e-2
with tf.name_scope("tgt") as tgt_scope:

    tgt_dense0 = tf.contrib.layers.fully_connected(inputs=in_eeg_features, num_outputs=tgt_layers[0], activation_fn=tf.sigmoid,scope='tgt_dense0')
    tgt_dense1 = tf.contrib.layers.fully_connected(inputs=tgt_dense0, num_outputs=tgt_layers[1], activation_fn=tf.sigmoid,scope='tgt_dense1')
    tgt_dense = tf.contrib.layers.fully_connected(inputs=tgt_dense1, num_outputs=tgt_layers[2], activation_fn=None,scope='tgt_dense2')



#==============================================================================
#     tgt_dense = tf.contrib.layers.stack(in_eeg_features, tf.contrib.layers.fully_connected, tgt_layers, weights_initializer=tf.constant_initializer(1), activation_fn=tf.sigmoid,scope='tgt_dense')
#==============================================================================

    with tf.name_scope("predict"):
        tgt_out_softmax = tf.nn.softmax(tgt_dense, name="TGT_Softmax")
        tgt_out_predict = tf.arg_max(tgt_out_softmax, 1, name="TGT_Prediction") # TODO: which dimension is this over?

    with tf.name_scope("loss"):
        tgt_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tgt_dense, labels=in_rpv, name = 'TGT_Loss'))
    tgt_step = tf.Variable(0, name='TGT_Step', trainable=False)
    tgt_optimizer = tf.train.RMSPropOptimizer(learning_rate=tgt_lr, centered=False, decay=0.8)
    tgt_train_op = tgt_optimizer.minimize(tgt_loss, global_step=tgt_step)
    
    with tf.name_scope('summaries'):
        tgt_summaries = tf.summary.merge([
#            tf.summary.scalar("tgt_step", tgt_step),
#           tf.summary.scalar("tgt_predict", tgt_out_predict)
            tf.summary.scalar("tgt_loss", tgt_loss)
            ])
