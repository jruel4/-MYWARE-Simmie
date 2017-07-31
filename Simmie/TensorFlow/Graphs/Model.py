# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 23:01:59 2017

@author: marzipan
"""

import tensorflow as tf
import numpy as np
import time

#==============================================================================
# if __name__ == "__main__":
#     from Policy import Policy
#     from Value import Value
#     from Shared import Shared
#     from Reward import Reward
# else:
#==============================================================================
from Policy import Policy
from Value import Value
from Shared import Shared
from Reward import Reward

class ActorCritic:
    def __init__(
            self,
            _NAUDIO_COMMANDS, #scalar, number of possible audio commands
            _EEG_INPUT_SHAPE, #shape, (ntimepoints, nchan, nfreqs)
            _LOGDIR, #pass in directory to write summaries and whatnot
            _POLICY_LR = 1e-4, #scalar, policy learning rate
            _VALUE_LR = 1e-3, #scalar, value learning rate
            _REWARD_MA_LEN = 100, #scalar
            _LSTM_CELLS = [30,30,30] #lstm dimensions, (cell0_size, cell1_size, ...) when total length is number of cells

            ):
        

        # These should not be changed by user but may change later in architechture
        self._InputShape = list(_EEG_INPUT_SHAPE)
        self._LSTMCells = list(_LSTM_CELLS)
        self._LSTMUnrollLength = 1
        self._ValueDiscount = 1.0
        
        self._Policy = Policy(_LEARNING_RATE = _POLICY_LR, _ACTIONSPACE_SIZE = _NAUDIO_COMMANDS)
        self._Value = Value(_LEARNING_RATE = _VALUE_LR, _DISCOUNT_RATE = self._ValueDiscount)
        self._Reward = Reward(_INPUT_SHAPE = _EEG_INPUT_SHAPE, _MA_LENGTH = _REWARD_MA_LEN)
        self._Shared = Shared(_CELLS=_LSTM_CELLS, _UNROLL_LENGTH = self._LSTMUnrollLength)

        # We store a version of the hidden state which we pass in every iteration
        self._HiddenStateShape = (len(_LSTM_CELLS), 2, self._LSTMUnrollLength, _LSTM_CELLS[-1])        
        self._LocalHiddenState = np.zeros(self._HiddenStateShape)
        
        # Save the logdir
        self.mLogdir = _LOGDIR
        
        self._buildModel()
        self._buildSummaries()
        self._buildFeedDicts()
        self._initSession()

    def _buildModel(self):
        # Inputs (from the outpside world)
        self._phInputEEG = tf.placeholder(tf.float32, shape=list([None] + self._InputShape), name="St0")
        self._phHiddenState = [ tf.placeholder(tf.float32, (2, self._LSTMUnrollLength, self._LSTMCells[idx]), name="Cell" + str(idx) +"_Ht0_Ct0") for idx in range(len(self._LSTMCells)) ]

        '''
        =========
        REFERENCE
        =========
        
        Graph inputs:
            - self._phInputEEG
            - self._phHiddenState
        Graph Outputs:
            - self._Action
            - self._LSTMState
        Graph Train:
            - self._ValueTrain
            - self._PolicyTrain
        Graph Assigns:
            - self._RewardAssgn (after training)
            - self._ValueAssgn (after training)
            - self._PolicyGradientsAssgn (!! with prediction)

        Build order is as follows:
            0. Reshape EEG input
            1. Shared, must be build before policy and before value
            2. Reward, must be built before value
            3. Value, must be built before policy
            4. Policy
        '''
        
        # St
        self._netInputEEG = self._reshapeEEGInput(self._phInputEEG)
        
        # shared LSTM
        self._LSTMOutput, self._LSTMState = self._Shared.buildGraph(_phINPUT_EEG=self._netInputEEG, _INPUT_HSTATES = self._phHiddenState)

        # R(St)
        self._RewardOutput, self._RewardAssgn = self._Reward.buildGraph(_phINPUT_EEG = self._netInputEEG)
        
        # V(St)
        self._ValueTrain, self._TDError, self._ValueAssgn = self._Value.buildGraph(_INPUT_LAYER=self._LSTMOutput, _INPUT_REWARD=self._RewardOutput)
        
        # Pi(St)
        self._PolicyTrain, self._Action, self._PolicyGradientsAssgn = self._Policy.buildGraph(_INPUT_LAYER=self._LSTMOutput, _INPUT_TDERR=self._TDError)
    
    

        
        
    def _buildSummaries(self):
        
        weights_policy = tf.get_collection(tf.GraphKeys.WEIGHTS, self._Policy.mScope)
        weights_value = tf.get_collection(tf.GraphKeys.WEIGHTS, self._Value.mScope)
        biases_policy = tf.get_collection(tf.GraphKeys.BIASES, self._Policy.mScope)
        biases_value = tf.get_collection(tf.GraphKeys.BIASES, self._Value.mScope)


        # Value
        self.mValueTrainSummaries = tf.summary.merge([
            tf.summary.scalar("val_loss", self._Value.mLoss),
            tf.summary.scalar("td_error", tf.reduce_mean(self._TDError)),
            tf.summary.scalar("R_t1_instant", tf.reduce_mean(self._Reward.mRewardInstant)),
            tf.summary.scalar("R_t1_average", self._Reward.mRewardMA),
            tf.summary.scalar("R_t1_actual", tf.reduce_mean(self._Reward.mRewardActual)),
            tf.summary.scalar("V_t0", self._Value.mVt0[0,0]),
            tf.summary.scalar("V_t1", self._Value.mVt1[0,0])] +\
            [tf.summary.histogram(w.name.split(':')[0], tf.reshape(w.value(), [-1])) for w in weights_value] +
            [tf.summary.histogram(b.name.split(':')[0], tf.reshape(b.value(), [-1])) for b in biases_value] +
            [tf.summary.histogram(w.name.split(':')[0], tf.reshape(w.value(), [1,1,self._LSTMCells[-1], 1]), max_outputs=10) for w in weights_value]
        )
        # Policy
        self.mPolicyTrainSummaries = tf.summary.merge([
            tf.summary.scalar("action_greedy", self._Policy.mAt0Greedy[0]),
            tf.summary.scalar("action_taken", self._Action[0]),
            tf.summary.histogram("softmax", self._Policy.mSoftmax),
            tf.summary.histogram("action_taken", self._Action)] +
            [tf.summary.histogram(w.name.split(':')[0], tf.reshape(w.value(), [-1])) for w in weights_policy] +
            [tf.summary.histogram(b.name.split(':')[0], tf.reshape(b.value(), [-1])) for b in biases_policy] +
            [tf.summary.histogram(w.name.split(':')[0], tf.reshape(w.value(), [1,-1,self._LSTMCells[-1], 1]), max_outputs=10) for w in weights_policy]
            )
            #[tf.summary.histogram(g.name.split("pv_rnn/")[-1].split(':')[0], g) for g in pol_gradients] )
        
#==============================================================================
#         self.mInputSummary = tf.summary.merge([
#                 tf.summary.image('spect', tf.reshape(in_raw_eeg_features, shape_eeg_input[1:] + [1]), max_outputs=nchan)
#                 ])
#         
#         tgt_prof_summaries = tf.summary.merge([ 
#                 tf.summary.image('tgt_prof', tf.reshape(in_raw_tgt_profile , [ntimepoints,nchan,nfreqs,1]), max_outputs=nchan),
#                 tf.summary.image('tgt_weighting', tf.reshape(in_raw_tgt_weighting , [ntimepoints,nchan,nfreqs,1]), max_outputs=nchan)
#                 ])
#==============================================================================

    def _buildFeedDicts(self):
        #Generate the fetch dictionaries for various actions
        self.mPolicyActFetch = {
                'rnn_output_state': self._LSTMState,
                'action'          : self._Action,
                'step'            : self._Policy.mStepPredict,
                'gradient_save'   : self._PolicyGradientsAssgn,
                }
        self.mPolicyActFetch_Summaries = self.mPolicyActFetch #no summaries
        
        self.mPolicyTrainFetch = {
                'rnn_output_state': self._LSTMState,
                'train_op'  : self._PolicyTrain,
                'step'      : self._Policy.mStep
                }
        self.mPolicyTrainFetch_Summaries = dict(summaries=self.mPolicyTrainSummaries, **self.mPolicyTrainFetch)
        
        self.mValueTrainFetch = {
                'rnn_output_state': self._LSTMState,
                'train_op'  : self._ValueTrain,
                'loss'      : self._TDError,
                'step'      : self._Value.mStep,
                'assgn_ops' : [self._RewardAssgn, self._ValueAssgn] #these will be evaluated last (using control_dependencies)
                }
        self.mValueTrainFetch_Summaries = dict(summaries=self.mValueTrainSummaries, **self.mValueTrainFetch)
    
    def _initSession(self):
        # Tensorflow Init
        self.mSaver = tf.train.Saver()
        self.mSess = tf.Session()
        self.mWriter = tf.summary.FileWriter(self.mLogdir, self.mSess.graph)
        self.mSess.run(tf.global_variables_initializer())
        
        self.idx = 0
    
    # Provide a sessrun_name to do a full trace
    def _runSession(self, _FETCH, _FEED, _SESSRUN_NAME=''):
        _SESSRUN_NAME = str(self.idx)
        self.idx +=1
        if _SESSRUN_NAME != '':
            metadata = tf.RunMetadata()
            out = self.mSess.run(
                    fetches = _FETCH,
                    feed_dict = _FEED,
                    options = tf.RunOptions(trace_level=tf.RunOptions.SOFTWARE_TRACE),
                    run_metadata = tf.RunMetadata())
            self.mWriter.add_run_metadata(metadata, _SESSRUN_NAME)
        else:
            out = self.mSess.run( _FETCH, _FEED )
        return out
    
    def _reshapeEEGInput(self, _IN):        
        shape_spectrogram_flat = np.prod(self._InputShape)
        reshape_shape = [-1, self._LSTMUnrollLength, shape_spectrogram_flat]
        reshaped_input = tf.reshape(_IN, reshape_shape)
        return reshaped_input
    
    def _addHiddenState(self,feed):
        [ feed.update({self._phHiddenState[idx] : self._LocalHiddenState[idx]}) for idx in range(len(self._LocalHiddenState))]

    def trainPolicy(self, _INPUT, _DO_SUMMARIES=False, _UPDATE_HIDDEN_STATE=False):
        feed = { self._phInputEEG : _INPUT }
        self._addHiddenState(feed)
        
        if _DO_SUMMARIES:
            fetch = self.mPolicyTrainFetch_Summaries
        else:
            fetch = self.mPolicyTrainFetch
        
        out = self._runSession(fetch, feed)
        if _UPDATE_HIDDEN_STATE: self._LocalHiddenState = out['rnn_output_state']
        return out

    def trainValue(self, _INPUT, _DO_SUMMARIES=False, _UPDATE_HIDDEN_STATE=False):
        feed = { self._phInputEEG : _INPUT }
        self._addHiddenState(feed)

        if _DO_SUMMARIES:
            fetch = self.mValueTrainFetch_Summaries
        else:
            fetch = self.mValueTrainFetch

        out = self._runSession(fetch,feed)
        if _UPDATE_HIDDEN_STATE: self._LocalHiddenState = out['rnn_output_state']
        return out

    def chooseAction(self, _INPUT, _DO_SUMMARIES=False, _UPDATE_HIDDEN_STATE=False):
        feed = { self._phInputEEG : _INPUT }
        self._addHiddenState(feed)

        if _DO_SUMMARIES:
            fetch = self.mPolicyActFetch_Summaries
        else:
            fetch = self.mPolicyActFetch

        out = self._runSession(fetch,feed)
        if _UPDATE_HIDDEN_STATE: self._LocalHiddenState = out['rnn_output_state']
        return out
    
    def run(self, _INPUT, _DO_SUMMARIES=False):
        t0 = self.trainPolicy(_INPUT, _DO_SUMMARIES)
        t1 = self.trainValue(_INPUT, _DO_SUMMARIES)
        act0 = self.chooseAction(_INPUT, _DO_SUMMARIES, _UPDATE_HIDDEN_STATE=True)
        
        if _DO_SUMMARIES:
            self.mWriter.add_summary(t0['summaries'], global_step=t0['step'])
            self.mWriter.add_summary(t1['summaries'], global_step=t1['step'])
            self.mWriter.flush()

        return act0['action']

    def updateTargetState(self):
#        reshape_tf_tgt = [1, self._LSTMUnrollLength, shape_spectrogram_flat]
        pass
    
    
if __name__ == "__main__":
        # Used for input reshaping
#    np_shape_eeg_input = [-1, ntimepoints, nchan, nfreqs]
#    np_shape_rnn_input = (len(pv_layers), 2, 1, pv_layers[-1])

    shape_eeg = [1,1,250] # [ntp, nchan, nfreq]
    naudio = 67

    import numpy as np
    import time
    from Simmie.TensorFlow.Utilities import *

    logdir = pwd(__file__) + '\\log__' + __file__.split('/')[-1] + '\\'
    print("\nLogdir: " + logdir + "\n")    

    tf.reset_default_graph()

    Simmie = ActorCritic(naudio, shape_eeg, logdir)
    
    print("Begin: ", time.time())
    for i in range(1):
        Simmie.run(np.ones([1] + shape_eeg), _DO_SUMMARIES=True)
    print("End: ", time.time())

    tb = yes_no("Tensorboard?")
    br =  yes_no("Browser?")
    if tb: start_tensorboard(logdir)
    if br: start_browser(logdir)