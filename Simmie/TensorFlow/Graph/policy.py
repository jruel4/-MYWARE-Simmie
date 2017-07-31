# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 16:48:13 2017

@author: marzipan
"""


##
# Policy parameters
##

import tensorflow as tf

class PolicyNet():

    '''
    Args
        pv_lstm_out:        shared net output
        pol_output_units:   number of output units (equivalent to number of possible actions)
        pol_imp_lr:         learning rate of policy net in imprint mode
        pol_rl_lr:          learning rate of policy net in RL mode
    '''
    def __init__(self, pv_lstm_out, pol_output_units, pol_imp_lr, pol_rl_lr):
        self.pv_lstm_out = pv_lstm_out
        self.pol_output_units = pol_output_units
    
        self.pol_imp_lr = pol_imp_lr
        self.pol_rl_lr = pol_rl_lr
    
    '''
    
    '''
    def build_inputs(self, in_eeg, in_action):
        self.in_eeg = in_eeg
        self.in_action = in_action
    
    def get_summaries(self):
        pass
    def get_prediction(self):
        pass
    def get_loss_op(self):
        pass
    def get_train_op(self):
        pass
    def get_step(self):
        pass
    
    '''
    Args
        graph:      the default graph
    '''
    def build_graph(self,default_graph):
        with default_graph.as_default():
            with tf.name_scope("pol"):
                with tf.name_scope("predict"):
                    self.pol_out_softmax = tf.nn.softmax(self.pv_lstm_out, name="POL_Softmax")
                    self.pol_out_predict = tf.arg_max(self.pol_out_softmax, 1, "POL_Prediction")
            
                with tf.name_scope("summaries"):
                    self.pol_summaries = tf.summary.merge([
                        tf.summary.scalar("pol_prediction", self.pol_out_predict[0])
                    ])
                with tf.name_scope("pol_imp"):
                    # imprinting
                    self.pol_imp_step = tf.Variable(0, name='POLIMP_Step', trainable=False)
                    self.pol_imp_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pol_out_softmax, labels=self.in_action, name = 'POLIMP_Loss'))
                    self.pol_imp_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.pol_imp_lr, centered=False, decay=0.8)
                    self.pol_imp_train_op = self.pol_imp_optimizer.minimize(self.pol_imp_loss, global_step=self.pol_imp_step)
                    
                    with tf.name_scope('summaries'):
                        self.pol_imp_summaries = tf.summary.merge([
            #                tf.summary.scalar("polimp_step", pol_imp_step),
                            tf.summary.scalar("polimp_loss", self.pol_imp_loss)
                        ])
            
                with tf.name_scope("pol_rl"):
                    self.pol_rl_step = tf.Variable(0, name='POLRL_Step', trainable=False)
                    self.pol_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "pv") +\
                                    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "pol/pol_rl")
                    self.pol_rl_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.pol_rl_lr, centered=False, decay=0.8)
                    self.pol_rl_loss = val_prediction_error * val_loss #not correct
                    self.pol_rl_train_op = pol_rl_optimizer.minimize(self.pol_rl_loss, global_step=self.pol_rl_step, var_list=self.pol_variables)
                
                    with tf.name_scope('summaries'):
                        self.pol_rl_summaries = tf.summary.merge([
            #                tf.summary.scalar("polrl_step", pol_rl_step),
                            tf.summary.scalar("polrl_loss", self.pol_rl_loss[0,0])
                        ])

    
class PolicyImp:
    with default_graph.as_default():
        with tf.name_scope("pv"):
            LSTMCellOps = [tf.contrib.rnn.BasicLSTMCell(pv_nunits, state_is_tuple=True,forget_bias=2.0) for pv_nunits in self.pv_layers]
            stackedLSTM = tf.contrib.rnn.MultiRNNCell(LSTMCellOps, state_is_tuple=True)
            unstackedInput = tf.unstack(self.in_eeg, axis=1, num=self.pv_unroll_len, name="PV_UnrolledInput")
        
            # cellOutputs corresponds to the output of each multicell in the unrolled LSTM
            # finalState maps to last cell in unrolled LSTM; has one entry for each cell in the multicell ([C0,...,Cn] for an n-celled multicell), each entry is tuple corresponding to (internalHiddenState,outputHiddenState)
            cellOutputs, multicellFinalState = tf.contrib.rnn.static_rnn(stackedLSTM, unstackedInput, dtype=tf.float32, scope='pv_rnn')
        
            self.pv_lstm_out = cellOutputs[-1]
        
        return {
                'out' : [self.pv_lstm_out],
                }