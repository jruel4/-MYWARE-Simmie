# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 06:02:07 2017

@author: marzipan
"""

import tensorflow as tf

G_logs = 'C:\\Users\\marzipan\\workspace\\Simmie\Experimental\Logs\\'
G_logdir = G_logs + 'A16\\'





'''

All input parameters the next receives

'''

shape_eeg_feat = [None,3,30 * 30]
shape_rpv = [None,2]
shape_act = [None,60]

with tf.name_scope("in"):
    in_eeg_features = tf.placeholder(tf.float32, shape=shape_eeg_feat, name="IN_EEG")
    in_rpv = tf.placeholder(tf.float32, shape=shape_rpv, name="IN_RPV")
    in_action = tf.placeholder(tf.float32, shape=shape_act, name="IN_ACT")

##
# Target parameters
##
tgt_output_units = shape_rpv[1]
tgt_layers = [50,50,tgt_output_units]
tgt_lr = 1e-6
with tf.name_scope("tgt") as tgt_scope:

#    tgt_dense = tf.contrib.layers.fully_connected(in_eeg_features, tgt_layers[0], weights_initializer=tf.constant_initializer(1), activation_fn=tf.sigmoid,scope='layers/l1')
    tgt_dense = tf.contrib.layers.stack(in_eeg_features, tf.contrib.layers.fully_connected, tgt_layers, weights_initializer=tf.constant_initializer(1), activation_fn=tf.sigmoid,scope='tgt_dnn')
    tgt_out_softmax = tf.nn.softmax(tgt_dense, name="TGT_Softmax")
    tgt_out_predict = tf.arg_max(tgt_out_softmax, 1, name="TGT_Prediction") # TODO: which dimension is this over?
    
    tgt_step = tf.Variable(0, name='TGT_Step', trainable=False)
    with tf.name_scope("loss"):
        tgt_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tgt_dense, labels=in_rpv, name = 'TGT_Loss'))
    tgt_optimizer = tf.train.RMSPropOptimizer(learning_rate=tgt_lr, centered=False, decay=0.8)
    tgt_train_op = tgt_optimizer.minimize(tgt_loss, global_step=tgt_step)


##
# Policy / Value parameters
##

# value
val_tgt_weights = tf.constant([-1.,1.],dtype=tf.float32) #weight the output of tgt_out_softmax in calculating value
val_discount_rate = tf.constant(0.9)
val_lr = 1e-6
val_output_units = 1

# policy
pol_output_units = shape_act[1]
pol_imp_lr = 8e-7
pol_unsup_lr = 8e-7

# shared
pv_layers = [100,100, pol_output_units]
pv_unroll_len = 3

with tf.name_scope("pv"):
    LSTMCellOps = [tf.contrib.rnn.BasicLSTMCell(pv_nunits, state_is_tuple=True,forget_bias=2.0) for pv_nunits in pv_layers]
    stackedLSTM = tf.contrib.rnn.MultiRNNCell(LSTMCellOps, state_is_tuple=True)
    unstackedInput = tf.unstack(in_eeg_features, axis=1, num=pv_unroll_len, name="PV_UnrolledInput")

    # cellOutputs corresponds to the output of each multicell in the unrolled LSTM
    # finalState maps to last cell in unrolled LSTM; has one entry for each cell in the multicell ([C0,...,Cn] for an n-celled multicell), each entry is tuple corresponding to (internalHiddenState,outputHiddenState)
    cellOutputs, multicellFinalState = tf.contrib.rnn.static_rnn(stackedLSTM, unstackedInput, dtype=tf.float32, scope='pv_rnn')

    pv_lstm_out = cellOutputs[-1]

with tf.name_scope("val"):
    val_previous_predicted = tf.Variable(0.0, "VAL_PreviousPredicted", dtype=tf.float32)
    val_next_predicted = tf.contrib.layers.fully_connected(inputs=pv_lstm_out, num_outputs=val_output_units, activation_fn=None,scope='pv/val')
    val_actual_reward = tf.tensordot( val_tgt_weights, tgt_out_softmax, axes=1)
    
    val_step = tf.Variable(0, name='VAL_Step', trainable=False)
    val_loss = (val_actual_reward - val_previous_predicted) + (val_discount_rate * val_next_predicted) # need to manage execution order here, this won't work...
    val_optimizer = tf.train.RMSPropOptimizer(learning_rate=val_lr, centered=False, decay=0.8)
    val_train_op = tgt_optimizer.minimize(val_loss, global_step=val_step)

with tf.name_scope("pol"):
    pol_out_softmax = tf.nn.softmax(pv_lstm_out)
    
    with tf.name_scope("pol_imp"):
        # imprinting
        pol_imp_step = tf.Variable(0, name='POLIMP_Step', trainable=False)
        pol_imp_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pol_out_softmax, labels=in_action, name = 'POLIMP_Loss'))
        pol_imp_optimizer = tf.train.RMSPropOptimizer(learning_rate=pol_imp_lr, centered=False, decay=0.8)
        pol_imp_train_op = tgt_optimizer.minimize(pol_imp_loss, global_step=pol_imp_step)

    with tf.name_scope("pol_unsup"):
        pol_unsup_step = tf.Variable(0, name='POLUNSUP_Step', trainable=False)
        pol_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "pv") +\
                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "pol/pol_unsup")
        pol_unsup_optimizer = tf.train.RMSPropOptimizer(learning_rate=pol_unsup_lr, centered=False, decay=0.8)
        pol_unsup_loss = val_loss #not correct
        pol_unsup_train_op = tgt_optimizer.minimize(pol_unsup_loss, global_step=pol_unsup_step, var_list=pol_variables)




sess = tf.Session()
summary_writer = tf.summary.FileWriter(G_logdir, sess.graph)