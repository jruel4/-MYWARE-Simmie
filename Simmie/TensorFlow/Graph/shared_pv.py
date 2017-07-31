# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 16:47:57 2017

@author: marzipan
"""

import tensorflow as tf

##
# Shared PV parameters
##


class SharedNet:
    def __init__(self, layers, unroll_length):
        self.pv_layers = layers #[100,100, shape_act[1]]
        self.pv_unroll_len = unroll_length

    def build_inputs(self, in_eeg, in_action):
        self.in_eeg = in_eeg
        self.in_action = in_action
    

    def build_graph(self,default_graph):
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