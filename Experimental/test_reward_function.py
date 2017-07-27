# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 13:54:29 2017

@author: marzipan
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 13:53:26 2017

@author: marzipan
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 06:02:07 2017

@author: marzipan
"""

import time
import tensorflow as tf
import numpy as np
import pickle

from Simmie.InterfaceAdapters.Output_Adapters.Audio_Command_Adapter import AudioCommandAdapter
from Simmie.InterfaceAdapters.Input_Adapters.EEG_State_Adapter import EEGStateAdapter

from DataUtilities.LSLUtils.TargetProfile import TargetProfile

nchan = 1
nfreqs = 125
ntimepoints = 250
sps=100


# Directories
G_logs = 'C:\\Users\\marzipan\\workspace\\Simmie\\Experimental\\Logs\\RewardFunctionTest\\'
G_logdir = G_logs + 'A1\\'

### CONFIG

# All
spectrogram_size = nfreqs * ntimepoints * nchan

shape_eeg_input = [None, ntimepoints, nchan, nfreqs]
shape_tgt_profile_input = [ntimepoints, nchan, nfreqs]

shape_eeg_feat = [-1,1,spectrogram_size]
shape_tgt_profile = [1,1,spectrogram_size]

tf.reset_default_graph()

# INPUTS
with tf.name_scope("in"):
    in_raw_eeg_features = tf.placeholder(tf.float32, shape=shape_eeg_input, name="IN_EEG")
    in_raw_tgt_profile = tf.placeholder(tf.float32, shape=shape_tgt_profile_input, name="IN_TGT_PROF")
    in_raw_tgt_weighting = tf.placeholder(tf.float32, shape=shape_tgt_profile_input, name="IN_TGT_WEIGHTING")

    in_eeg_features = tf.reshape(in_raw_eeg_features, shape_eeg_feat)
    in_tgt_profile = tf.reshape(in_raw_tgt_profile, shape_tgt_profile)
    in_tgt_weighting = tf.reshape(in_raw_tgt_weighting, shape_tgt_profile)

'''
REWARD
Input:
    in_eeg_features
Args:
    shape_tgt_profile - should be [1,1,specgram_size] (subject to change)
Out:
    reward - the reward (supports batching)
        1D tensor of length "batch_size"
'''
with tf.variable_scope("rwrd"):
    tgt = tf.get_variable("TargetProfile", shape = shape_tgt_profile, initializer=tf.constant_initializer(value=1.0), trainable=False)
    tgt_weighting = tf.get_variable("TargetProfileWeighting", shape = shape_tgt_profile, initializer=tf.constant_initializer(value=1.0), trainable=False)
    
    # Assign op is used when changing the target profile
    tgt_assgn = tf.assign(tgt, in_tgt_profile)
    tgt_weighting_assgn = tf.assign(tgt_weighting, in_tgt_weighting)

    # Take the euclidean distance between the two tensors
    diff = in_eeg_features - tgt
    sq = tf.square(diff)
    sq_weighted = sq * tgt_weighting
    summed = tf.reduce_sum(sq_weighted,axis=(1,2))
    sqrt = tf.sqrt(summed)
    reward = -1 * sqrt



# Target profile input
#   [ntimepoints, nchan, nfreqs]


with tf.name_scope('summaries'):

    input_summaries = tf.summary.merge([
            tf.summary.image('spect', tf.reshape(in_raw_eeg_features, shape_eeg_input[1:] + [1]), max_outputs=nchan)
            ])
    
    tgt_prof_summaries = tf.summary.merge([ 
            tf.summary.image('tgt_prof', tf.reshape(in_raw_tgt_profile , [ntimepoints,nchan,nfreqs,1]), max_outputs=nchan),
            tf.summary.image('tgt_weighting', tf.reshape(in_raw_tgt_weighting , [ntimepoints,nchan,nfreqs,1]), max_outputs=nchan)
            ])

def set_target_profile(sess,writer,tgt_profile, tgt_weighting, step):
    feed = {
            in_raw_tgt_profile : tgt_profile,
            in_raw_tgt_weighting : tgt_weighting
            }
    fetch = {
            'target_profile_assign_op' : tgt_assgn,
            'target_weighting_assign_op' : tgt_weighting_assgn, 
            'target_profile_image' : tgt_prof_summaries
            }
    out = sess.run( fetch, feed )
    writer.add_summary(out['target_profile_image'], global_step=step)


sum_op = tf.summary.merge_all()

# Tensorflow Init
saver = tf.train.Saver()
sess = tf.Session()
summary_writer = tf.summary.FileWriter(G_logdir, sess.graph)
sess.run(tf.global_variables_initializer())


tgt1 = TargetProfile()
f,w = tgt1.create_tgt_profile(range(18,23),[1,1,1,1,1],nchan,nfreqs, ntimepoints)
w2 = w / np.max(w)
set_target_profile(sess,summary_writer,w,w2,0)

a_l=list()
a_=list()
for i in range(nfreqs):
    a=np.asarray([[[1. if ((x >= (i - 2)) and (x <= (i + 2))) else 0. for x in range(nfreqs)]]*nchan]*ntimepoints)
    a_.append(a)
    a=a[None,:,:,:]
    a_l.append(sess.run(reward,{in_raw_eeg_features:a}))
    s = sess.run(input_summaries,{in_raw_eeg_features:a})
    summary_writer.add_summary(s, global_step=i)