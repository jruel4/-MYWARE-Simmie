# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 17:15:30 2017

@author: marzipan
"""

import tensorflow as tf

'''
Member variables beginning with "_" indicate internal variables
Member variables begnning with "m" indicate public facing variables
'''
class Shared:
    def __init__(self, _CELLS, _UNROLL_LENGTH, _SCOPE="shared"):
        
        # Parameters
        self.mLSTMCellSizes = _CELLS
        self.mLSTMUnrollLength = _UNROLL_LENGTH
        self.mScope = _SCOPE
        
        # "Public" variables
        self.mOutputSize = self.mLSTMCellSizes[-1]
        
        # Declare inputs and outputs, useful for debugging
        self.mOutputLayer = None
        self.mInputData = None
        self.mInputHState = None


    '''
    buildGraph
    Input:
        - EEG state placeholder
        - (!)LIST OF hidden states, NOT TENSOR!
    Output:
        - LSTM output layer
        - LSTM state
    '''
    def buildGraph(self, _phINPUT_EEG, _INPUT_HSTATES):

        self.mInputData = _phINPUT_EEG
        self.mInputHState = _INPUT_HSTATES
        
        with tf.variable_scope(self.mScope):

            # Create individual LSTM cells
            self.mLSTMCells = list()
            for cell_size in self.mLSTMCellSizes:
                self.mLSTMCells.append(tf.contrib.rnn.BasicLSTMCell(cell_size, state_is_tuple=True,forget_bias=2.0))
        
            # Connect all of the LSTM cells together
            self._StackedLSTM = tf.contrib.rnn.MultiRNNCell(self.mLSTMCells, state_is_tuple=True)
        
            # Unroll the input (assuming we're getting a static sequence length in, TODO make dynamic...)
            self._UnstackedInput = tf.unstack(self.mInputData, axis=1, num=self.mLSTMUnrollLength, name="unrolled_input")

            # Load in the LSTM state
            self._RNNTupleState = tuple(
                [tf.nn.rnn_cell.LSTMStateTuple(self.mInputHState[idx][0], self.mInputHState[idx][1])
                 for idx in range(len(self.mLSTMCells))]
            )
        
            # Generate RNN
            # mLSTMSequenceOutputs is a list of length mLSTMUnrollLength where each element is h(t) of the last cell (i.e. the output)
            # mLSTMFinalState is final states (h(t) and c(t)) for all cells
            self._LSTMSequenceOutputs, self._LSTMFinalState = tf.contrib.rnn.static_rnn(
                    cell = self._StackedLSTM,
                    inputs = self._UnstackedInput,
                    dtype = tf.float32,
                    initial_state = self._RNNTupleState,
                    scope = 'rnn')
        
            self.mOutputLayer = self._LSTMSequenceOutputs[-1]
            
        return self.mOutputLayer, self._LSTMFinalState
    
    
    
if __name__ == "__main__":
    import numpy as np
    from Simmie.TensorFlow.Utilities import *

    logdir = pwd(__file__) + '\\log__' + __file__.split('/')[-1] + '\\'
    print("\nLogdir: " + logdir + "\n")


    tf.reset_default_graph()

    cells = [30,40,50]

    in_eeg = tf.placeholder(tf.float32, [None,1,250*1], name='InputEEG')
 
    # NOTE: for each cell the hidden state is [2, unroll_len,cell_size]
    in_ht = [ tf.placeholder(tf.float32, (2, 1, cell), name="InputHt") for cell in cells ]
    
    # As long
    s = Shared(_CELLS=cells)
    
    out_l0 = s.buildGraph(in_eeg, in_ht)
    
    sess = tf.Session()
    summary_writer = tf.summary.FileWriter(logdir, sess.graph)
    sess.run(tf.global_variables_initializer())

    # Simple run test
    ht = [np.ones((2,1,cell)) for cell in cells]

    feed = {in_eeg:np.ones((1,1,250))}
    for idx in range(len(ht)):
        feed.update({in_ht[idx]:ht[idx]})

    _ = sess.run([out_l0], feed)

    sess.close()
    
    tb = yes_no("Tensorboard?")
    br =  yes_no("Browser?")
    if tb: start_tensorboard(logdir)
    if br: start_browser(logdir)