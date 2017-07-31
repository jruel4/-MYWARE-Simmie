# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 17:08:25 2017

@author: marzipan
"""

import tensorflow as tf

class Reward:
    def __init__(self, _INPUT_SHAPE, _MA_LENGTH, _SCOPE="reward"):
        
        # Parameters
        self.mInputShape = _INPUT_SHAPE
        self.mMALength = _MA_LENGTH
        self.mScope = _SCOPE

        # Get variables
        self._TargetState = tf.get_variable(_SCOPE + "/target_profile/values", shape = _INPUT_SHAPE, initializer=tf.constant_initializer(value=1.0), trainable=False)
        self._TargetWeights = tf.get_variable(_SCOPE + "/target_profile/weights", shape = _INPUT_SHAPE, initializer=tf.constant_initializer(value=1.0), trainable=False)
        self._RewardRingbuf = tf.get_variable(_SCOPE + "/reward_ringbuf", shape=[_MA_LENGTH], initializer=tf.constant_initializer(1.0), trainable=False)
    
        # Declare inputs and outputs, useful for debugging
        self.mInputData = None
    
    def _reward_euclidean(self):
        with tf.name_scope("euclidean_distance"):
            diff = self.mInputData - self._TargetState
            sq = tf.square(diff)
            sq_weighted = sq * self._TargetWeights
            summed = tf.reduce_sum(sq_weighted,axis=(1,2))
            sqrt = tf.sqrt(summed)
            distance = sqrt
        return distance
    
    '''
    buildGraph
    Input:
        - Input EEG data placeholder
    Output:
        - Reward
        - (!) assign op, RUN AT THE END OF EVERY STEP!
    '''
    def buildGraph(self, _phINPUT_EEG):
        
        self.mInputData = _phINPUT_EEG
        
        with tf.variable_scope(self.mScope):
            with tf.name_scope("calculate_reward"):
                self.mRewardInstant = self._reward_euclidean()
                self.mRewardMA = tf.reduce_mean(self._RewardRingbuf)
                self.mRewardActual = self.mRewardInstant - self.mRewardMA
            
            # Rotate the buffer
            with tf.name_scope("rotate_buffer"):
                with tf.control_dependencies([self.mRewardActual]):
                    rb_rot = tf.concat([self._RewardRingbuf[1:], tf.reduce_sum(self.mRewardInstant,keep_dims=True)], axis=0)
                    reward_ringbuf_assgn = tf.assign(self._RewardRingbuf,rb_rot)
        
        return self.mRewardActual, [reward_ringbuf_assgn]
        
    '''
    Pass in instance of session to automagically update variables,
    alternatively pass _SESS=None to just get back assgn's
    '''
    def updateTarget(self, _SESS, _TARGET_STATE, _WEIGHTS):

        state_assgn = tf.assign(self._TargetState, _TARGET_STATE)
        weight_assgn = tf.assign(self._TargetWeights, _WEIGHTS)

        if _SESS == None:
            return (state_assgn, weight_assgn)
        else:
            _SESS.run([state_assgn.op, weight_assgn.op])
            
            
if __name__ == "__main__":
    import numpy as np
    from Simmie.TensorFlow.Utilities import *

    logdir = pwd(__file__) + '\\log__' + __file__.split('/')[-1] + '\\'
    print("\nLogdir: " + logdir + "\n")


    tf.reset_default_graph()

    tgt_shape = (1,1,250)
    in_eeg = tf.placeholder(tf.float32, [None,1,250*1], name='InputEEG')
    
    # As long
    r = Reward(_INPUT_SHAPE=tgt_shape, _MA_LENGTH=100)
    
    rt0, assgn = r.buildGraph(in_eeg)
    
    sess = tf.Session()
    summary_writer = tf.summary.FileWriter(logdir, sess.graph)
    sess.run(tf.global_variables_initializer())

    feed = {in_eeg:np.ones((1,1,250))}

    _ = sess.run([rt0, assgn], feed)
    
    r.updateTarget(_SESS=sess,_TARGET_STATE=np.ones(tgt_shape),_WEIGHTS=np.ones(tgt_shape))

    sess.close()
    
    tb = yes_no("Tensorboard?")
    br =  yes_no("Browser?")
    if tb: start_tensorboard(logdir)
    if br: start_browser(logdir)