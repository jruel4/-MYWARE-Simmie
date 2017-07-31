# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 17:13:25 2017

@author: marzipan
"""

import tensorflow as tf

class Policy:
    def __init__(self, _LEARNING_RATE, _ACTIONSPACE_SIZE, _SCOPE="policy", _PARENT_SCOPE="shared"):
        # Parameters
        self.mLearningRate = _LEARNING_RATE
        self.mActionspaceSize = _ACTIONSPACE_SIZE
        self.mScope = _SCOPE
        self.mParentScope = _PARENT_SCOPE

        # Variables
        self.mStep = tf.get_variable(self.mScope + '/policy_step', shape=[], dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False)
        self.mStepPredict = tf.get_variable(self.mScope + '/policy_predict_step', shape=[], dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False)
        
        # Declare inputs and outputs, useful for debugging
        self.mOutputValue = None
        self.mInputLayer = None
        self.mInputTDError = None

    '''
    Inputs:
        - Input to this graph (usually output from shared, but can be a placeholder for testing)
        - TD error (usually from value function)
    Outputs:
        - training op
        - action taken
        - assign op
        YOU MUST INCLUDE assign op IN FETCH DICT WHEN RUNNING A PREDICTION!
    '''
    def buildGraph(self, _INPUT_LAYER, _INPUT_TDERR):
        
        self.mInputLayer = _INPUT_LAYER
        self.mInputTDError = _INPUT_TDERR
        
        # Predictions / softmax
        with tf.variable_scope(self.mScope + "/prediction"):
            self._L0 = tf.contrib.layers.fully_connected(
                    inputs = self.mInputLayer,
                    num_outputs = self.mActionspaceSize,
                    activation_fn = None,
                    scope = 'dense0')


            self.mSoftmax = tf.nn.softmax(self._L0, name="softmax")
            self.mAt0Greedy = tf.arg_max(self.mSoftmax, 1, "A_t0_greedy")
        
            # Choose our output action based on the softmax probabilities
            distribution = tf.contrib.distributions.Multinomial(total_count=1., probs=self.mSoftmax, name="multinomial_sample")
            self.mAt0OneHot = distribution.sample()
            self.mAt0 = tf.arg_max(self.mAt0OneHot, 1, "A_t0")
                                
        with tf.variable_scope(self.mScope + "/train"):
            
            # Get valid trainable variables
            self.mTrainableVariables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.mScope) +\
                                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.mParentScope)
            
            # Calculate policy gradient
            with tf.variable_scope("calculate_gradients"):

                self.mGradients = list()
                for v in self.mTrainableVariables:
                    varname = "gradient_vars/" + "/".join(v.name.replace(':','').split('/')[-3:])
                    self.mGradients.append( tf.get_variable(varname, shape=v.shape, dtype=v.dtype, trainable=False, initializer=tf.constant_initializer(0.0)) )
                
                # Reduce softmax output to only actions we have chosen
                self.mAt0Probability = tf.reduce_max(self.mSoftmax * tf.cast(self.mAt0OneHot, tf.float32), axis=1)
                gradients_tmp = [ grad / self.mAt0Probability for grad in tf.gradients(self.mAt0Probability, self.mTrainableVariables) ]
                # CAUTION: The above statement will break if the batch size is not 1!
                
                # Create gradient save op
                assgn_gradients = list()
                for idx in range(len(self.mGradients)):
                    assgn_gradients.append( tf.assign( self.mGradients[idx], gradients_tmp[idx]))
                    
                '''
                YOU MUST INCLUDE assgn_gradients IN FETCH DICT WHEN RUNNING A PREDICTION
                '''
                
            
            # Generate the training op
            with tf.name_scope("apply_gradients"):
                self.mTrainOp = list()
                for idx in range(len(self.mTrainableVariables)):
                    delta = self.mGradients[idx] * self.mLearningRate * tf.reduce_sum(self.mInputTDError, axis=0)
                    self.mTrainOp.append(tf.assign(self.mTrainableVariables[idx], self.mTrainableVariables[idx] + delta))
            
                # Is this how we're supposed to update the step??
                self.mTrainOp.append(tf.assign(self.mStep, self.mStep + 1))
                
        return self.mTrainOp, self.mAt0, [assgn_gradients]


if __name__ == "__main__":

    import numpy as np
    from Simmie.TensorFlow.Utilities import *

    logdir = pwd(__file__) + '\\log__' + __file__.split('/')[-1] + '\\'
    print("\nLogdir: " + logdir + "\n")

    tf.reset_default_graph()

    in_l0 = tf.placeholder(tf.float32, [None, 10], name='InputL0')
    in_td = tf.placeholder(tf.float32, [None, 1], name="InputTd")
    
    p = Policy(_LEARNING_RATE=0.1, _ACTIONSPACE_SIZE=67)
    
    t_op, at0, assgn = p.buildGraph(in_l0, in_td)
    
    sess = tf.Session()
    summary_writer = tf.summary.FileWriter(logdir, sess.graph)
    sess.run(tf.global_variables_initializer())

    # Simple run test
    feed = {in_l0:np.ones((1,10)), in_td:np.ones([1,1])}
    _ = sess.run([t_op, at0, assgn], feed)

    sess.close()
    
    tb = yes_no("Tensorboard?")
    br =  yes_no("Browser?")
    if tb: start_tensorboard(logdir)
    if br: start_browser(logdir)