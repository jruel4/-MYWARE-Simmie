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

from DataUtilities.TargetProfile import TargetProfile

nchan = 8
nfreqs = 50
ntimepoints = 30
sps=100

OUTPUT = AudioCommandAdapter(
        name="Simmie",
        uid=np.random.randint(0,1e4),
        relaxed_enabled=True)

INPUT = EEGStateAdapter(
        n_freq=nfreqs, #should match with pipeline
        n_chan=nchan, #should match with pipeline
        eeg_feed_rate=sps, #sps
        samples_per_output=1, # 
        spectrogram_timespan=10, #assumed to be in seconds
        n_spectrogram_timepoints=ntimepoints)


_ = input("\nACA started, now start up audio control GUI...\nPress enter when ready.")
print("Continuing.")


# Directories
G_logs = 'C:\\Users\\marzipan\\workspace\\Simmie\\Experimental\\Logs\\'
G_logdir = G_logs + 'S25\\'
G_tgtdir = 'C:\\Users\\marzipan\\workspace\\Simmie\\Experimental\\Logs\\'

# File locations
checkpoint_file = G_logdir + 'model.ckpt'
tgt_file = G_tgtdir + 'alpha10hz.tgt'

### CONFIG

# All
naudio_commands = len(OUTPUT.get_valid_audio_commands())
spectrogram_size = nfreqs * ntimepoints * nchan
shape_eeg_feat = [None,1,spectrogram_size]
shape_tgt_profile = [1,1,spectrogram_size]
shape_rpv = [None,2]
shape_act = [None,naudio_commands]

#==============================================================================
# # Target
# tgt_output_units = shape_rpv[1]
# tgt_layers = [50,50,tgt_output_units]
# tgt_lr = 1e-3
#==============================================================================

# LEARNING RATES
pol_imp_lr = 5e-4
pol_rl_lr = 5e-4
val_lr = 1e-2

# RL CONSTANTS
val_discount_rate = tf.constant(0.9)

# Shared Net Structure
pv_layers = [100,100, shape_act[1]]
pv_unroll_len = 1

# Policy Net Structure
pol_output_units = pv_layers[-1]

# Value Net Structure
val_tgt_weights = tf.constant([-1.,1.],dtype=tf.float32) #weight the output of tgt_out_softmax in calculating value
val_output_units = 1





# INPUTS
with tf.name_scope("in"):
    in_eeg_features = tf.placeholder(tf.float32, shape=shape_eeg_feat, name="IN_EEG")
    in_rpv = tf.placeholder(tf.int32, shape=shape_rpv, name="IN_RPV")
    in_action = tf.placeholder(tf.int32, shape=shape_act, name="IN_ACT")
    in_tgt_profile = tf.placeholder(tf.float32, shape=shape_tgt_profile, name="IN_TGT_PROF")
    in_tgt_weighting = tf.placeholder(tf.float32, shape=shape_tgt_profile, name="IN_TGT_WEIGHTING")

with tf.name_scope("optimizers"):
    val_optimizer = tf.train.RMSPropOptimizer(learning_rate=val_lr, centered=False, decay=0.8)
    pol_imp_optimizer = tf.train.RMSPropOptimizer(learning_rate=pol_imp_lr, centered=False, decay=0.8)
    pol_rl_optimizer = tf.train.RMSPropOptimizer(learning_rate=pol_rl_lr, centered=False, decay=0.8)



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




# SHARED NET
with tf.variable_scope("pv"):
    LSTMCellOps = [tf.contrib.rnn.BasicLSTMCell(pv_nunits, state_is_tuple=True,forget_bias=2.0) for pv_nunits in pv_layers]
    stackedLSTM = tf.contrib.rnn.MultiRNNCell(LSTMCellOps, state_is_tuple=True)
    unstackedInput = tf.unstack(in_eeg_features, axis=1, num=pv_unroll_len, name="PV_UnrolledInput")

    # cellOutputs corresponds to the output of each multicell in the unrolled LSTM
    # finalState maps to last cell in unrolled LSTM; has one entry for each cell in the multicell ([C0,...,Cn] for an n-celled multicell), each entry is tuple corresponding to (internalHiddenState,outputHiddenState)
    cellOutputs, multicellFinalState = tf.contrib.rnn.static_rnn(stackedLSTM, unstackedInput, dtype=tf.float32, scope='pv_rnn')

    pv_lstm_out = cellOutputs[-1]

'''
VALUE NET
Input:
    pv_lstm_out - output of shared LSTM net

Parameters
    val_output_units - number of output units (usually just 1)
    val_tgt_weights - 
'''
with tf.variable_scope("val"):
    val_previous_predicted = tf.Variable(0.0, "VAL_PreviousPredicted", dtype=tf.float32)
    val_next_predicted = tf.contrib.layers.fully_connected(inputs=pv_lstm_out, num_outputs=val_output_units, activation_fn=None,scope='val_dense')
    val_actual_reward = reward
    
    val_prediction_error = val_actual_reward - val_previous_predicted
    with tf.name_scope('loss'):
        val_loss = tf.reduce_mean(val_prediction_error + (val_discount_rate * val_next_predicted)) # need to manage execution order here, this won't work...
    val_step = tf.Variable(0, name='VAL_Step', trainable=False)
    val_train_op = val_optimizer.minimize(val_loss, global_step=val_step)
    
    with tf.control_dependencies([val_loss]):
        val_assgn_op0 = val_previous_predicted.assign(val_next_predicted[0,0])
    

# POLICY
with tf.variable_scope("pol_predict"):
        pol_out_softmax = tf.nn.softmax(pv_lstm_out,name="POL_Softmax")
        pol_out_predict = tf.arg_max(pol_out_softmax, 1, "POL_Prediction")

with tf.variable_scope("pol_imp"):
    # imprinting
    pol_imp_step = tf.Variable(0, name='POLIMP_Step', trainable=False)
    pol_imp_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pol_out_softmax, labels=in_action, name = 'POLIMP_Loss'))
    pol_imp_train_op = pol_imp_optimizer.minimize(pol_imp_loss, global_step=pol_imp_step)
    
with tf.variable_scope("pol_rl"):
    pol_rl_step = tf.Variable(0, name='POLRL_Step', trainable=False)
    pol_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "pv") +\
                    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "pol_rl")
    pol_rl_loss = tf.reduce_mean(val_prediction_error * val_loss) #not correct
    pol_rl_train_op = pol_rl_optimizer.minimize(pol_rl_loss, global_step=pol_rl_step, var_list=pol_variables)

with tf.name_scope('summaries'):

#==============================================================================
#     # Target
#     tgt_summaries =     tf.summary.merge([ tf.summary.scalar("tgt_loss", tgt_loss) ])
#==============================================================================
    
    # Value
    val_summaries = tf.summary.merge([
        tf.summary.scalar("val_loss", val_loss),
        tf.summary.scalar("val_prediction_error", val_prediction_error[0]),
        tf.summary.scalar("val_previous_predicted", val_previous_predicted),
        tf.summary.scalar("val_current_reward", val_actual_reward[0]),
        tf.summary.scalar("val_next_predicted", val_next_predicted[0,0]),
    ])
    
    # Policy
    pol_summaries =     tf.summary.merge([ tf.summary.scalar("pol_prediction", pol_out_predict[0]) ])
    pol_rl_summaries =  tf.summary.merge([ tf.summary.scalar("polrl_loss", pol_rl_loss) ])
    pol_imp_summaries = tf.summary.merge([ tf.summary.scalar("polimp_loss", pol_imp_loss) ])
    
    input_summaries = tf.summary.merge([ tf.summary.image('spect', tf.transpose(tf.reshape(in_tgt_profile, [-1,nchan*ntimepoints,nfreqs,1]),perm=[0,2,1,3]) ) ])
    
    tgt_prof_summaries = tf.summary.merge([ 
            tf.summary.image('tgt_prof', tf.transpose(tf.reshape(in_tgt_profile, [nchan,ntimepoints,nfreqs,1]),perm=[0,2,1,3]), max_outputs=nchan)
            ])
    
#==============================================================================
# # This gets the initial layer's weights and creates an image summary
# with tf.name_scope("weight_images"):
#     with tf.variable_scope("tgt_dense0",reuse=True):
#         x=tf.get_variable("weights")
#     y=tf.reshape(x, [-1,900,50,1])
#     z=tf.summary.merge([ tf.summary.image('spect',y) ])
#     q=sess.run(z)
#     summary_writer.add_summary(q,global_step = 1000)
#     print(tgt_dense0)
#==============================================================================

# length of idx must match batch size
def one_hot(idx, total, batch_size=1):
    assert len(idx) == batch_size
    out = np.zeros([batch_size, total])
    out[np.arange(batch_size), idx] = 1
    return out.astype(np.int32)

def spoof_data(batch_size=1):
    return np.random.randn(batch_size,shape_eeg_feat[1],shape_eeg_feat[2])
def spoof_act(batch_size=1):
    acts = [ np.random.randint(0,naudio_commands) for x in range(batch_size)]
    return one_hot(acts,naudio_commands,batch_size)
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

def val_train(sess, writer, eeg_data, sessrun_name=''):
    feed = { in_eeg_features : eeg_data}
    fetch = {
            'train_op'  : val_train_op,
            'loss'      : val_loss,
            'summaries' : val_summaries,
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
            'summaries' : pol_imp_summaries,
            'step'      : pol_imp_step
            }
    return train(sess,writer,feed,fetch,sessrun_name)

def pol_rl_train(sess, writer, eeg_data, sessrun_name=''):
    feed = { in_eeg_features : eeg_data}
    fetch = {
            'train_op'  : pol_rl_train_op,
            'loss'      : pol_rl_loss,
            'summaries' : pol_rl_summaries,
            'step'      : pol_rl_step
            }
    return train(sess,writer,feed,fetch,sessrun_name)


def set_target_profile(sess,writer,tgt_profile, tgt_weighting, step):
    feed = {
            in_tgt_profile : tgt_profile,
            in_tgt_weighting : tgt_weighting
            }
    fetch = {
            'target_profile_assign_op' : tgt_assgn,
            'target_weighting_assign_op' : tgt_weighting_assgn, 
            'target_profile_image' : tgt_prof_summaries
            }
    out = sess.run( fetch, feed )
    writer.add_summary(out['target_profile_image'], global_step=step)

#def input_summaries(sess,writer,eeg_data):
    

# Tensorflow Init
saver = tf.train.Saver()
sess = tf.Session()
summary_writer = tf.summary.FileWriter(G_logdir, sess.graph)
sess.run(tf.global_variables_initializer())

#==============================================================================
# tgt_predict(sess,summary_writer,spoof_data(1000),'TGT_PRED_TST')
# pol_predict(sess,summary_writer,spoof_data(1000),'POL_PRED_TST')
# tgt_train(sess,summary_writer,spoof_data(1000),spoof_rpv(1000),'TGT_TRN_TST')
# val_train(sess,summary_writer,spoof_data(1000),'VAL_TRN_TST')
# pol_imp_train(sess,summary_writer,spoof_data(1000),spoof_act(1000),'POL_IMP_TRN_TST')
# pol_rl_train(sess,summary_writer,spoof_data(1000),'POL_RL_TRN_TST')
#==============================================================================

# Empty data structures
empty_data = np.empty([0,1,nfreqs * ntimepoints * nchan])
empty_rpv = np.empty([0,shape_rpv[1]])
empty_act = np.empty([0,naudio_commands])

# Batching
imp_training_data = empty_data
imp_training_lbls = empty_act
#==============================================================================
# tgt_training_data = empty_data
# tgt_training_lbls = empty_rpv
#==============================================================================
rl_training_data = empty_data
#==============================================================================
# tgt_minimum_batch_size = 5
#==============================================================================
imp_minimum_batch_size = 100
rl_minimum_batch_size = 100

# Intervals - all in seconds!
interval_send_command = 0.1 # time to send all commands
interval_console_output = 1.0
interval_summary_writer = 0.1
interval_graph_saver = 60.0

# Timerkeepers / timers
timekeep_send_command = time.time()
timekeep_console_output = time.time()
timekeep_summary_writer = time.time()
timekeep_summary_writer1 = time.time()
timekeep_summary_writer2 = time.time()
timekeep_graph_saver = time.time()

timekeep_beginning = time.time()

# Bools
do_sendcommand = True
do_console = True
do_summaries = True
do_summaries1 = True
do_summaries2 = True
do_save = False

# Logging
cmds_sent = 0
loop_idx = 0

policy_rl_train_info = None
policy_imp_train_info = None
value_train_info = None
#==============================================================================
# tgt_train_info = None
#==============================================================================


# Start the input threads
INPUT.launch_eeg_adapter()

# input timekeep and interval, return (is_beeping, new_timekeep)
def check_timekeep(timekeep, interval):
    is_beeping = ((time.time() - timekeep) > interval)
    if is_beeping: timekeep = time.time()
    return timekeep, is_beeping


#==============================================================================
# !!!!!!!!!
# HUGE TODO
# Freeze LSTM state, restore at the end!
# Could also just freeze during predictions and do one training per 
# !!!!!!!!!
#==============================================================================


# LOAD TARGET STATE

tgt = TargetProfile()
f,w = tgt.create_tgt_profile(range(18,23),[200,200,200,200,200],nchan,nfreqs, ntimepoints)
w=w.reshape((1,1,-1))
w2 = w / np.max(w)
set_target_profile(sess,summary_writer,w,w2,0)

once = False
try:
    while(True):
        loop_idx += 1
        
        # Get new data from adapters
        raw_data, tgt_data, tgt_lbls, imp_data, imp_lbls  = INPUT.retrieve_latest_data()
        # Check if actually got any data
        if len(raw_data) > 0:
            raw_data = np.reshape(raw_data,(-1,1, ntimepoints * nchan * nfreqs))
            rl_training_data = np.concatenate((rl_training_data,raw_data))
        else:
            # imp_data / rpv_data should NEVER contain any data if raw_data is empty
            # should be safe to skip
            continue
    
        # Send output commands if they have not been sent in this interval
        if do_sendcommand:
            for i in range(4):
                act_out = pol_predict(sess, summary_writer, raw_data[-1,None])['prediction'][0] # TODO, Ok to send states like this?
                OUTPUT.submit_command_relaxed(act_out)
            cmds_sent += 1
    
        # Load training data if any is present
        if len(imp_data) > 0:
            imp_data = np.reshape(imp_data,(-1,1, ntimepoints * nchan * nfreqs))
            imp_lbls = one_hot(imp_lbls, naudio_commands, len(imp_lbls))#[:,None,:]
            imp_training_data = np.concatenate((imp_training_data,imp_data))
            imp_training_lbls = np.concatenate((imp_training_lbls,imp_lbls))


        # Check to see if we have enough data to train
        if len(rl_training_data) >= rl_minimum_batch_size:
            policy_rl_train_info = pol_rl_train(sess, summary_writer, rl_training_data)
            value_train_info = val_train(sess, summary_writer, rl_training_data)
            rl_training_data = empty_data
            timekeep_summary_writer ,   do_summaries     = check_timekeep(timekeep_summary_writer,  interval_summary_writer)
            if do_summaries:
                summary_writer.add_summary(policy_rl_train_info['summaries'], global_step=policy_rl_train_info['step'])
                summary_writer.add_summary(value_train_info['summaries'], global_step=value_train_info['step'])
    
    
        if len(imp_training_data) >= imp_minimum_batch_size:
            policy_imp_train_info = pol_imp_train(sess, summary_writer, imp_training_data,imp_training_lbls)
            imp_training_data = empty_data
            imp_training_lbls = empty_act
            timekeep_summary_writer1 ,   do_summaries1     = check_timekeep(timekeep_summary_writer1,  interval_summary_writer)
            if do_summaries1:
                summary_writer.add_summary(policy_imp_train_info['summaries'], global_step=policy_imp_train_info['step'])

        if do_console:
            print("Single loop took: %.5f" % (time.time() - timekeep_console_output),
                  ", loop number: ", loop_idx,
                  ", cmds sent: ", cmds_sent,
                  ", t-0: %.4f" % (time.time() - timekeep_beginning))
            summary_writer.flush()

        if do_save:
            print("Saving checkpoint")
            saver.save(sess, checkpoint_file, global_step=loop_idx)

        timekeep_graph_saver    ,   do_save          = check_timekeep(timekeep_graph_saver,     interval_graph_saver)
        timekeep_send_command   ,   do_sendcommand   = check_timekeep(timekeep_send_command,    interval_send_command)
        timekeep_console_output ,   do_console       = check_timekeep(timekeep_console_output,  interval_console_output)


except KeyboardInterrupt:
    print("Receieved interrupt")
    print("Saving checkpoint")
    saver.save(sess, checkpoint_file, global_step=loop_idx)
    summary_writer.flush()
    INPUT.stop_eeg_thread()
    OUTPUT.close()