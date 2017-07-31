# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 17:12:07 2017

@author: marzipan
"""

import tensorflow as tf

class Value:
    def __init__(self, _LEARNING_RATE, _DISCOUNT_RATE, _SCOPE="value", _PARENT_SCOPE="shared"):
        # Parameters
        self.mLearningRate = _LEARNING_RATE
        self.mDiscountRate = _DISCOUNT_RATE
        self.mScope = _SCOPE
        self.mParentScope = _PARENT_SCOPE

        # Variables
        self.mVt0 = tf.get_variable(self.mScope + "/V_t0", shape=(1,1), dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=False)
        self.mStep = tf.get_variable(self.mScope + '/value_step', shape=[], dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False)
        
        # Declare inputs and outputs, useful for debugging
        self.mVt1 = None
        self.mTDError = None
        self.mTrainOp = None
        
        self.mInputLayer = None
        self.mInputReward = None

    '''
    Inputs:
        - Input to this graph (usually output from shared, but can be a placeholder for testing)
        - Reward for current state
    Outputs:
        - training op
        - tderror for V(t)
        - assign op, evaluate during / after training(!)
    '''
    def buildGraph(self, _INPUT_LAYER, _INPUT_REWARD):

        self.mInputLayer = _INPUT_LAYER
        self.mInputReward = _INPUT_REWARD
        
        with tf.variable_scope(self.mScope):
            
            # Generate V(t+1)
            self.mVt1 = tf.contrib.layers.fully_connected(
                    inputs=self.mInputLayer,
                    num_outputs=1,
                    activation_fn=None,
                    scope='V_t1')
            
            self.mActualReward = self.mInputReward
            
            # Value prediction error is (R(T) + future_discount*V(T+1)) - V(T)
            #   V(T) represents the expected value of this state
            #   (R(T) + discount*V(T+1)) represents a more accurate value (since we know R(T))
            #
            # Note, that we are now using average reward, so discount rate = 1.0 and mActualReward is the immediate reward - average reward
            #
            with tf.name_scope('loss'):
                self.mTDError = (self.mActualReward + self.mDiscountRate * self.mVt1) - self.mVt0
                self.mTDError = tf.identity(self.mTDError, name="tderr")
                self.mLoss = tf.abs(tf.reduce_mean(self.mTDError))
            
            with tf.name_scope('train'):
                self.mTrainableVariables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.mScope) +\
                                            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.mParentScope)

                self._Optimizer = tf.train.AdamOptimizer(learning_rate=self.mLearningRate)
                self.mTrainOp = self._Optimizer.minimize(self.mLoss, var_list=self.mTrainableVariables, global_step=self.mStep)
            
            with tf.control_dependencies([self.mLoss, self.mTrainOp]):
                vt0_assgn = self.mVt0.assign([self.mVt1[0]])
                
        return self.mTrainOp, self.mTDError, [vt0_assgn]
                
                
if __name__ == "__main__":

    import os,glob 
    import numpy as np
    from Simmie.TensorFlow.Utilities import *

    logdir = pwd(__file__) + '\\log__' + __file__.split('/')[-1] + '\\'
    print("\nLogdir: " + logdir + "\n")    

    tf.reset_default_graph()

    in_l0 = tf.placeholder(tf.float32, [None,10], name='InputL0')
    in_rt = tf.placeholder(tf.float32, [1], name="InputRt")
    
    # As long
    v = Value(_LEARNING_RATE=0.1)
    
    train, value, err, assgn = v.buildGraph(in_l0, in_rt)
    
    sess = tf.Session()
    summary_writer = tf.summary.FileWriter(logdir, sess.graph)
    sess.run(tf.global_variables_initializer())

    # Simple run test
    feed = {in_l0:np.ones((1,10)), in_rt:np.ones([1])}
    _, v, td, _ = sess.run([value, err, train, assgn], feed)

    sess.close()
    
    tb = yes_no("Tensorboard?")
    br =  yes_no("Browser?")
    if tb: start_tensorboard(logdir)
    if br: start_browser(logdir)