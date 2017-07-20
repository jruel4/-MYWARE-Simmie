# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 06:02:07 2017

@author: marzipan
"""

import tensorflow as tf
import numpy as np

G_logs = 'C:\\Users\\marzipan\\workspace\\Simmie\Experimental\Logs\\'
G_logdir = G_logs + 'A16\\'





'''

All input parameters the next receives

'''

shape_eeg_feat = [None,1,30 * 30]
shape_rpv = [None,2]
shape_act = [None,60]

with tf.name_scope("in"):
    in_eeg_features = tf.placeholder(tf.float32, shape=shape_eeg_feat, name="IN_EEG")
    in_rpv = tf.placeholder(tf.int32, shape=shape_rpv, name="IN_RPV")
    in_action = tf.placeholder(tf.int32, shape=shape_act, name="IN_ACT")

##
# Target parameters
##
tgt_output_units = shape_rpv[1]
tgt_layers = [50,50,tgt_output_units]
tgt_lr = 1e-6
with tf.name_scope("tgt") as tgt_scope:
    tgt_dense = tf.contrib.layers.stack(in_eeg_features, tf.contrib.layers.fully_connected, tgt_layers, weights_initializer=tf.constant_initializer(1), activation_fn=tf.sigmoid,scope='tgt_dense')
    with tf.name_scope("predict"):
        tgt_out_softmax = tf.nn.softmax(tgt_dense, name="TGT_Softmax")
        tgt_out_predict = tf.arg_max(tgt_out_softmax, 1, name="TGT_Prediction") # TODO: which dimension is this over?

    with tf.name_scope("loss"):
        tgt_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tgt_dense, labels=in_rpv, name = 'TGT_Loss'))
    tgt_step = tf.Variable(0, name='TGT_Step', trainable=False)
    tgt_optimizer = tf.train.RMSPropOptimizer(learning_rate=tgt_lr, centered=False, decay=0.8)
    tgt_train_op = tgt_optimizer.minimize(tgt_loss, global_step=tgt_step)
    
    with tf.name_scope('summaries'):
        tf.summary.scalar("tgt_step", tgt_step)
        tf.summary.scalar("tgt_predict", tgt_out_predict)
        tf.summary.scalar("tgt_loss", tgt_loss)


##
# Shared PV parameters
##
pv_layers = [100,100, shape_act[1]]
pv_unroll_len = 1

with tf.name_scope("pv"):
    LSTMCellOps = [tf.contrib.rnn.BasicLSTMCell(pv_nunits, state_is_tuple=True,forget_bias=2.0) for pv_nunits in pv_layers]
    stackedLSTM = tf.contrib.rnn.MultiRNNCell(LSTMCellOps, state_is_tuple=True)
    unstackedInput = tf.unstack(in_eeg_features, axis=1, num=pv_unroll_len, name="PV_UnrolledInput")

    # cellOutputs corresponds to the output of each multicell in the unrolled LSTM
    # finalState maps to last cell in unrolled LSTM; has one entry for each cell in the multicell ([C0,...,Cn] for an n-celled multicell), each entry is tuple corresponding to (internalHiddenState,outputHiddenState)
    cellOutputs, multicellFinalState = tf.contrib.rnn.static_rnn(stackedLSTM, unstackedInput, dtype=tf.float32, scope='pv_rnn')

    pv_lstm_out = cellOutputs[-1]


##
# Value parameters
##
val_tgt_weights = tf.constant([-1.,1.],dtype=tf.float32) #weight the output of tgt_out_softmax in calculating value
val_discount_rate = tf.constant(0.9)
val_lr = 1e-6
val_output_units = 1

with tf.name_scope("val"):
    val_previous_predicted = tf.Variable(0.0, "VAL_PreviousPredicted", dtype=tf.float32)
    val_next_predicted = tf.contrib.layers.fully_connected(inputs=pv_lstm_out, num_outputs=val_output_units, activation_fn=None,scope='val_dense')
    val_actual_reward = tf.reduce_sum(val_tgt_weights * tgt_out_softmax)
    
    val_prediction_error = val_actual_reward - val_previous_predicted
    with tf.name_scope('loss'):
        val_loss = val_prediction_error + (val_discount_rate * val_next_predicted) # need to manage execution order here, this won't work...
    val_step = tf.Variable(0, name='VAL_Step', trainable=False)
    val_optimizer = tf.train.RMSPropOptimizer(learning_rate=val_lr, centered=False, decay=0.8)
    val_train_op = tgt_optimizer.minimize(val_loss, global_step=val_step)

    with tf.name_scope('summaries'):
        tf.summary.scalar("val_step", val_step)
        tf.summary.scalar("val_loss", val_loss)
        tf.summary.scalar("val_prediction_error", val_prediction_error)
        tf.summary.scalar("val_previous_predicted", val_previous_predicted)
        tf.summary.scalar("val_current_reward", val_actual_reward)
        tf.summary.scalar("val_next_predicted", val_next_predicted)

    
    with tf.control_dependencies([val_loss]):
        val_assgn_op0 = val_previous_predicted.assign(val_next_predicted[0,0])
    



##
# Policy parameters
##
pol_output_units = pv_layers[-1]
pol_imp_lr = 8e-7
pol_unsup_lr = 8e-7

with tf.name_scope("pol"):
    with tf.name_scope("predict"):
        pol_out_softmax = tf.nn.softmax(pv_lstm_out,name="POL_Softmax")
        pol_out_predict = tf.arg_max(pol_out_softmax, 1, "POL_Prediction")

    with tf.name_scope("summaries"):
        tf.summary.scalar("pol_prediction", pol_out_predict)
    
    with tf.name_scope("pol_imp"):
        # imprinting
        pol_imp_step = tf.Variable(0, name='POLIMP_Step', trainable=False)
        pol_imp_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pol_out_softmax, labels=in_action, name = 'POLIMP_Loss'))
        pol_imp_optimizer = tf.train.RMSPropOptimizer(learning_rate=pol_imp_lr, centered=False, decay=0.8)
        pol_imp_train_op = tgt_optimizer.minimize(pol_imp_loss, global_step=pol_imp_step)
        
        with tf.name_scope('summaries'):
            tf.summary.scalar("pol_imp_step", pol_imp_step)
            tf.summary.scalar("pol_imp_loss", pol_imp_loss)


    with tf.name_scope("pol_unsup"):
        pol_unsup_step = tf.Variable(0, name='POLUNSUP_Step', trainable=False)
        pol_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "pv") +\
                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "pol/pol_unsup")
        pol_unsup_optimizer = tf.train.RMSPropOptimizer(learning_rate=pol_unsup_lr, centered=False, decay=0.8)
        pol_unsup_loss = val_prediction_error * val_loss #not correct
        pol_unsup_train_op = tgt_optimizer.minimize(pol_unsup_loss, global_step=pol_unsup_step, var_list=pol_variables)
    
        with tf.name_scope('summaries'):
            tf.summary.scalar("pol_unsup_step", pol_unsup_step)
            tf.summary.scalar("pol_unsup_loss", pol_unsup_loss)


###
# Summaries
###
summary_op = tf.summary.merge_all()

# length of idx must match batch size
def one_hot(idx, total, batch_size=1):
    assert len(idx) == batch_size
    out = np.zeros([batch_size, total])
    out[np.arange(batch_size), idx] = 1
    return out.astype(np.int32)

def spoof_data(batch_size=1):
    return np.random.randn(batch_size,shape_eeg_feat[1],shape_eeg_feat[2])
def spoof_act(batch_size=1):
    acts = [ np.random.randint(0,60) for x in range(batch_size)]
    return one_hot(acts,60,batch_size)
def spoof_rpv(batch_size=1):
    rpvs = [ np.random.randint(0,2) for x in range(batch_size)]
    return one_hot(rpvs, shape_rpv[1], batch_size)


def predict(sess, writer, feed, fetch, sessrun_name=''):
    if sessrun_name != '':
        run_opt = tf.RunOptions(trace_level=tf.RunOptions.SOFTWARE_TRACE)
        run_md = tf.RunMetadata()
        out = sess.run( fetch, feed, run_opt, run_md)
        writer.add_run_metadata(run_md, sessrun_name)
    else:
        out = sess.run( fetch, feed )
    return out

def train(sess,writer,feed,fetch,sessrun_name=''):
    if sessrun_name != '':
        run_opt = tf.RunOptions(trace_level=tf.RunOptions.SOFTWARE_TRACE)
        run_md = tf.RunMetadata()
        out = sess.run( fetch, feed, run_opt, run_md)
        writer.add_run_metadata(run_md, sessrun_name)
    else:
        out = sess.run( fetch, feed )
    return out

def tgt_predict(sess, writer, eeg_data, sessrun_name=''):
    feed = {in_eeg_features : eeg_data}
    fetch = {
            'prediction'      : tgt_out_predict,
            'softmax'         : tgt_out_softmax }
    return predict(sess,writer,feed,fetch,sessrun_name)


def tgt_train(sess, writer, eeg_data, rpv, sessrun_name=''):
    print(rpv)
    print(in_rpv)
    feed = {
            in_eeg_features    : eeg_data,
            in_rpv             : rpv       }
    fetch = {
            'train_op'  : tgt_train_op,
            'loss'      : tgt_loss,
#            'summaries' : summary_op,
            'step'      : tgt_step
            }
    return train(sess,writer,feed,fetch,sessrun_name)


def val_train(sess, writer, eeg_data, sessrun_name=''):
    feed = { in_eeg_features : eeg_data}
    fetch = {
            'train_op'  : val_train_op,
            'loss'      : val_loss,
#            'summaries' : summary_op,
            'step'      : val_step,
            'assgn_op'  : val_assgn_op0.op # assigns the new predicted value to the old predicted value
            }
    return train(sess,writer,feed,fetch,sessrun_name)

    
def pol_predict(sess, writer, eeg_data, sessrun_name=''):
    feed = {in_eeg_features : eeg_data}
    fetch = {
            'prediction'      : pol_out_predict,
            'softmax'         : pol_out_softmax }
    return predict(sess,writer,feed,fetch,sessrun_name)
    

def pol_imp_train(sess, writer, eeg_data, proctor_action, sessrun_name=''):
    feed = {
            in_eeg_features : eeg_data,
            in_action       : proctor_action }
    fetch = {
            'train_op'  : pol_imp_train_op,
            'loss'      : pol_imp_loss,
#            'summaries' : summary_op,
            'step'      : pol_imp_step
            }
    return train(sess,writer,feed,fetch,sessrun_name)

def pol_unsup_train(sess, writer, eeg_data, sessrun_name=''):
    feed = { in_eeg_features : eeg_data}
    fetch = {
            'train_op'  : pol_unsup_train_op,
            'loss'      : pol_unsup_loss,
#            'summaries' : summary_op,
            'step'      : pol_unsup_step
            }
    return train(sess,writer,feed,fetch,sessrun_name)


sess = tf.Session()
summary_writer = tf.summary.FileWriter(G_logdir, sess.graph)
sess.run(tf.global_variables_initializer())

tgt_predict(sess,summary_writer,spoof_data(),'TGT_PRED_1TST')
pol_predict(sess,summary_writer,spoof_data(),'POL_PRED_1TST')
tgt_train(sess,summary_writer,spoof_data(),spoof_rpv(),'TGT_TRN_1TST')
val_train(sess,summary_writer,spoof_data(),'VAL_TRN_TST')
pol_imp_train(sess,summary_writer,spoof_data(),spoof_act(),'POL_IMP_TRN_TST')
pol_unsup_train(sess,summary_writer,spoof_data(),'POL_UNSUP_TRN_TST')


feed = {
            in_eeg_features    : spoof_data(),
            in_rpv             : spoof_rpv(),       
            in_action          : spoof_act()}
s = sess.run(summary_op, feed)
summary_writer.add_summary(s, global_step=0)