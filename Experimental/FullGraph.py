# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 06:02:07 2017

@author: marzipan
"""

import time
import tensorflow as tf
import numpy as np

from Simmie.InterfaceAdapters.Output_Adapters.Audio_Command_Adapter import AudioCommandAdapter
from Simmie.InterfaceAdapters.Input_Adapters.EEG_State_Adapter import EEGStateAdapter

from LSLUtils.TargetProfile import TargetProfile

nchan = 1
nfreqs = 250
ntimepoints = 5
sps=250

OUTPUT = AudioCommandAdapter(
        name="Simmie",
        uid=np.random.randint(0,1e4),
        relaxed_enabled=True)

INPUT = EEGStateAdapter(
        n_freq=nfreqs, #should match with pipeline
        n_chan=nchan, #should match with pipeline
        eeg_feed_rate=sps, #sps
        samples_per_output=1, # 
        spectrogram_timespan=1, #assumed to be in seconds
        n_spectrogram_timepoints=ntimepoints)


_ = input("\nACA started, now start up audio control GUI...\nPress enter when ready.")
print("Continuing.")


# Directories
G_logs = 'C:\\Users\\marzipan\\workspace\\Simmie\\Experimental\\Logs\\'
G_logdir = G_logs + 'S48\\'
G_tgtdir = 'C:\\Users\\marzipan\\workspace\\Simmie\\Experimental\\Logs\\'

# File locations
checkpoint_file = G_logdir + 'model.ckpt'
tgt_file = G_tgtdir + 'alpha10hz.tgt'


tf.reset_default_graph()

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
pol_imp_lr = 5e-2
pol_rl_lr = 5e-2
val_lr = 2e-2

# RL CONSTANTS
val_discount_rate = tf.constant(0.1)

# Shared Net Structure
pv_layers = [10,10,10]
pv_unroll_len = 1

# Policy Net Structure
pol_output_units = shape_act[-1]

# Value Net Structure
val_tgt_weights = tf.constant([-1.,1.],dtype=tf.float32) #weight the output of tgt_out_softmax in calculating value
val_output_units = 1


# Used for input reshaping
np_shape_eeg_input = [-1, ntimepoints, nchan, nfreqs]

# All
spectrogram_size = nfreqs * ntimepoints * nchan

shape_eeg_input = [None, ntimepoints, nchan, nfreqs]
shape_tgt_profile_input = [ntimepoints, nchan, nfreqs]

shape_eeg_feat = [-1,1,spectrogram_size]
shape_tgt_profile = [1,1,spectrogram_size]

# INPUTS
with tf.name_scope("in"):
    in_raw_eeg_features = tf.placeholder(tf.float32, shape=shape_eeg_input, name="IN_EEG")
    in_raw_tgt_profile = tf.placeholder(tf.float32, shape=shape_tgt_profile_input, name="IN_TGT_PROF")
    in_raw_tgt_weighting = tf.placeholder(tf.float32, shape=shape_tgt_profile_input, name="IN_TGT_WEIGHTING")

    in_action = tf.placeholder(tf.int32, shape=shape_act, name="IN_ACT")
    in_rpv = tf.placeholder(tf.int32, shape=shape_rpv, name="IN_RPV")

    # Reshape the inputs
    in_eeg_features = tf.reshape(in_raw_eeg_features, shape_eeg_feat)
    in_tgt_profile = tf.reshape(in_raw_tgt_profile, shape_tgt_profile)
    in_tgt_weighting = tf.reshape(in_raw_tgt_weighting, shape_tgt_profile)


with tf.name_scope("optimizers"):
    val_optimizer = tf.train.AdamOptimizer(learning_rate=val_lr)
    pol_imp_optimizer = tf.train.RMSPropOptimizer(learning_rate=pol_imp_lr, centered=False, decay=0.8)
#    pol_rl_optimizer = tf.train.RMSPropOptimizer(learning_rate=pol_rl_lr, centered=False, decay=0.8)

'''
REWARD _ GOOD
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
    reward = sqrt


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
        reward - actual reward
    
    Parameters
        val_output_units - number of output units (usually just 1)
        val_tgt_weights - 
    '''
    with tf.name_scope("val"):
        # Carryover is the predicted value for state t0 using data from t-1
        val_carryover_previous_predicted = tf.get_variable("VAL_PreviousPredicted", shape=(1,1), dtype=tf.float32, initializer=tf.constant_initializer(1.0), trainable=False)
    
        # Next predicted is v(t0) -> v(tN)
        val_next_predicted = tf.contrib.layers.fully_connected(inputs=pv_lstm_out, num_outputs=val_output_units, activation_fn=None,scope='val_dense')
        val_previous_predicted = tf.concat([val_carryover_previous_predicted, val_next_predicted[:-1]],axis=0)
        val_actual_reward = reward
        
        # Value prediction error is (R(T) + future_discount*V(T+1)) - V(T)
        #   V(T) represents the expected value of this state
        #   (R(T) + future_discount*V(T+1)) represents a more accurate value (since we know R(T))
        val_prediction_error = (val_actual_reward + val_discount_rate * val_next_predicted) - val_previous_predicted

        with tf.name_scope('loss'):
            val_loss = tf.abs(tf.reduce_mean(val_prediction_error)) # need to manage execution order here, this won't work...
        val_step = tf.Variable(0, name='VAL_Step', trainable=False)
        val_train_op = val_optimizer.minimize(val_loss, global_step=val_step)
        
        with tf.control_dependencies([val_loss]):
            val_assgn_op0 = val_carryover_previous_predicted.assign([val_next_predicted[-1]])
        
    
    # POLICY
    with tf.name_scope("pol_predict"):
        pol_dense0 = tf.contrib.layers.fully_connected(inputs=pv_lstm_out, num_outputs=pol_output_units, activation_fn=None,scope='pol_dense')
        pol_out_softmax = tf.nn.softmax(pol_dense0,name="POL_Softmax")
        pol_out_predict = tf.arg_max(pol_out_softmax, 1, "POL_Prediction")
    
    with tf.name_scope("pol_imp"):
        pol_imp_step = tf.Variable(0, name='POLIMP_Step', trainable=False)
        pol_imp_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pol_out_softmax, labels=in_action, name = 'POLIMP_Loss'))
        pol_imp_train_op = pol_imp_optimizer.minimize(pol_imp_loss, global_step=pol_imp_step)
        
    with tf.name_scope("pol_rl"):
        pol_rl_step = tf.Variable(0, name='POLRL_Step', trainable=False)
        pol_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "pv/pv_rnn") +\
                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "pv/pol_dense")


        # Reduce softmax output to only the chosen actions (A(t) with highest probability)
        pol_chosen_acts = tf.reduce_max(pol_out_softmax, axis=1)
        
        # Scale actions by the value function error (and step size!)
        #   positive error means that the state's value was worth more than we expected
        #   negative error means that the state was worth less than we expected)
        pol_chosen_acts_scaled = (pol_chosen_acts * tf.stop_gradient(val_prediction_error)) / tf.stop_gradient(pol_chosen_acts)

        # Normalize w/ respect to the probability of the chosen action
#        pol_grads_norm = pol_grads / pol_chosen_acts

        # Take the gradients of the chosen action with respect to the parameterization of the function
        pol_grads = tf.gradients(pol_chosen_acts_scaled, pol_variables)

        # Generate the training op
        pol_rl_train_op = list()
        for idx in range(len(pol_variables)):
            pol_rl_train_op.append(tf.assign(pol_variables[idx], pol_variables[idx] + pol_grads[idx] * pol_rl_lr))

        # Is this how we're supposed to update the step??
        pol_rl_train_op.append(tf.assign(pol_rl_step, pol_rl_step+1))

#        pol_rl_loss = tf.reduce_mean(val_prediction_error * val_loss) #not correct
#        pol_rl_train_op = pol_rl_optimizer.minimize(pol_rl_loss, global_step=pol_rl_step, var_list=pol_variables)

with tf.name_scope('summaries'):

    # Value
    val_summaries = tf.summary.merge([
        tf.summary.scalar("val_loss", val_loss),
        tf.summary.scalar("val_mean_prediction_error", tf.reduce_mean(val_prediction_error)),
        tf.summary.scalar("val_predicted_t0", val_carryover_previous_predicted[0,0]),
        tf.summary.scalar("val_reward_t0", val_actual_reward[0]),
        tf.summary.scalar("val_predicted_t1", val_next_predicted[0,0])
    ])
    
    # Policy
    pol_summaries =     tf.summary.merge([ tf.summary.scalar("pol_prediction", pol_out_predict[0]) ])
    pol_rl_summaries =  tf.summary.merge([ tf.summary.scalar("polrl_step", pol_rl_step) ])
    pol_imp_summaries = tf.summary.merge([ tf.summary.scalar("polimp_loss", pol_imp_loss) ])
    
    input_summaries = tf.summary.merge([
            tf.summary.image('spect', tf.reshape(in_raw_eeg_features, shape_eeg_input[1:] + [1]), max_outputs=nchan)
            ])
    
    tgt_prof_summaries = tf.summary.merge([ 
            tf.summary.image('tgt_prof', tf.reshape(in_raw_tgt_profile , [ntimepoints,nchan,nfreqs,1]), max_outputs=nchan),
            tf.summary.image('tgt_weighting', tf.reshape(in_raw_tgt_weighting , [ntimepoints,nchan,nfreqs,1]), max_outputs=nchan)
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
    feed = { in_raw_eeg_features : eeg_data}
    fetch = {
            'train_op'  : val_train_op,
            'loss'      : val_loss,
            'summaries' : val_summaries,
            'step'      : val_step,
            'assgn_op'  : val_assgn_op0.op # assigns the new predicted value to the old predicted value
            }
    return train(sess,writer,feed,fetch,sessrun_name)

    
def pol_predict(sess, writer, eeg_data, sessrun_name=''):
    feed = {in_raw_eeg_features : eeg_data}
    fetch = {
            'prediction'      : pol_out_predict,
            'softmax'         : pol_out_softmax }
    return predict(sess,writer,feed,fetch,sessrun_name)
    

def pol_imp_train(sess, writer, eeg_data, proctor_action, sessrun_name=''):
    feed = {
            in_raw_eeg_features : eeg_data,
            in_action       : proctor_action }
    fetch = {
            'train_op'  : pol_imp_train_op,
            'loss'      : pol_imp_loss,
            'summaries' : pol_imp_summaries,
            'step'      : pol_imp_step
            }
    return train(sess,writer,feed,fetch,sessrun_name)

def pol_rl_train(sess, writer, eeg_data, sessrun_name=''):
    feed = { in_raw_eeg_features : eeg_data}
    fetch = {
            'train_op'  : pol_rl_train_op,
            'loss'      : val_loss,
            'summaries' : pol_rl_summaries,
            'step'      : pol_rl_step
            }
    return train(sess,writer,feed,fetch,sessrun_name)


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
empty_data = np.empty([0,ntimepoints,nchan,nfreqs])
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
rl_minimum_batch_size = 5

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
f,w = tgt.create_tgt_profile(range(18,23),[5e2,10e2,20e2,10e2,5e2],nchan, nfreqs, ntimepoints)
w2 = w / np.max(w)
set_target_profile(sess,summary_writer,w,w2,0)

try:
    while(True):
        loop_idx += 1
        
        # Get new data from adapters
        raw_data, tgt_data, tgt_lbls, imp_data, imp_lbls  = INPUT.retrieve_latest_data()
        # Check if actually got any data
        if len(raw_data) > 0:
            raw_data = raw_data.reshape(np_shape_eeg_input)
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
            imp_data = imp_data.reshape(np_shape_eeg_input)
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