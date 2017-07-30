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
ntimepoints = 1
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
        spectrogram_timespan=1., #assumed to be in seconds
        n_spectrogram_timepoints=ntimepoints)

_ = input("ACA started, now start up audio control GUI...\nPress enter when ready.")
print("Continuing.\n")


# LEARNING RATES
pol_imp_lr = 1e-50
pol_rl_lr = 4e-5
val_lr = 1e-3

# Directories
G_logs = 'C:\\Users\\marzipan\\workspace\\Simmie\\Experimental\\Logs\\LR\\'
G_logdir = G_logs + 'S17\\'
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
np_shape_rnn_input = (len(pv_layers), 2, 1, 10)

# All
spectrogram_size = nfreqs * ntimepoints * nchan

shape_eeg_input = [None, ntimepoints, nchan, nfreqs]
shape_tgt_profile_input = [ntimepoints, nchan, nfreqs]
shape_rnn_input = (len(pv_layers), 2, None, 10)

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

    # This represents the RNNs current state
    in_rnn_state = tf.placeholder(tf.float32, shape_rnn_input)

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
    
    # For each "layer" create an LSTM cell
    LSTMCellOps = list()
    for pv_units in pv_layers:
        LSTMCellOps.append(tf.contrib.rnn.BasicLSTMCell(pv_units, state_is_tuple=True,forget_bias=2.0))

    # Connect all of the LSTM cells together
    stackedLSTM = tf.contrib.rnn.MultiRNNCell(LSTMCellOps, state_is_tuple=True)

    # Unroll the input (assuming we're getting a static sequence length in, TODO make dynamic...)
    unstackedInput = tf.unstack(in_eeg_features, axis=1, num=pv_unroll_len, name="PV_UnrolledInput")

    #JCR0
    # Load in the LSTM state
    l = tf.unstack(in_rnn_state, axis=0)
    rnn_tuple_state = tuple(
        [tf.nn.rnn_cell.LSTMStateTuple(l[idx][0], l[idx][1])
         for idx in range(len(pv_layers))]
    )
    
#    batch_size = tf.concat((tf.shape(in_eeg_features)[0,None], np.ones([3])), 0)
#    rnn_states_batch = tf.tile(rnn_tuple_state, batch_size)

    # Generate RNN, multicellFinalState is the final state
    cellOutputs, multicellFinalState = tf.contrib.rnn.static_rnn(stackedLSTM, unstackedInput, dtype=tf.float32, initial_state=rnn_tuple_state,scope='pv_rnn')

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
        td_error = (val_actual_reward + val_discount_rate * val_next_predicted) - val_previous_predicted

        with tf.name_scope('loss'):
            val_loss = tf.abs(tf.reduce_mean(td_error)) # need to manage execution order here, this won't work...
        val_step = tf.Variable(0, name='VAL_Step', trainable=False)
        val_train_op = val_optimizer.minimize(val_loss, global_step=val_step)
        
        with tf.control_dependencies([val_loss]):
            val_assgn_op0 = val_carryover_previous_predicted.assign([val_next_predicted[-1]])
        
    
    # POLICY
    with tf.name_scope("pol_predict"):
        pol_dense0 = tf.contrib.layers.fully_connected(inputs=pv_lstm_out, num_outputs=pol_output_units, activation_fn=None,scope='pol_dense')
        pol_out_softmax = tf.nn.softmax(pol_dense0, name="POL_Softmax")
        pol_out_predict = tf.arg_max(pol_out_softmax, 1, "POL_Prediction")

        # Distribution
        dist = tf.contrib.distributions.Multinomial(total_count=1., probs=pol_out_softmax)
        dist_sample = dist.sample()
        pol_out_predict_dist = tf.arg_max(dist_sample, 1, "POL_PredictionDist")
        
        # This is used for summaries
        pol_step_var = tf.Variable(0, name='POL_Step', trainable=False)
        pol_step = tf.assign(pol_step_var, pol_step_var + 1)

    with tf.name_scope("pol_imp"):
        pol_imp_step = tf.Variable(0, name='POLIMP_Step', trainable=False)
        # TODO Does this need to be fixed since we are exploring, not just being greedy?
        pol_imp_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pol_out_softmax, labels=in_action, name = 'POLIMP_Loss'))
        pol_imp_train_op = pol_imp_optimizer.minimize(pol_imp_loss, global_step=pol_imp_step)
        
    with tf.name_scope("pol_rl"):
        pol_rl_step = tf.Variable(0, name='POLRL_Step', trainable=False)


    with tf.name_scope("calc_pol_grads"):
        # Get the variables which we can train on
        pol_trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "pv/pv_rnn") + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "pv/pol_dense")

        # Reduce softmax output to only actions we have chosen
        # NOTE: dist_sample is basically a mask, 0 if action was not chosen and 1 if action was chosen
        pol_chosen_acts = tf.cast(tf.reduce_max(pol_out_softmax * dist_sample, axis=1), tf.float32)
        pol_gradients_tmp = tf.gradients(pol_chosen_acts, pol_trainable_variables)
        #TODO divide the gradients 
        
        # Create gradient variables
        pol_gradient_vars = list()
        for v in pol_trainable_variables:
            pol_gradient_vars.append( tf.get_variable("policy_gradient/" + v.name, shape=v.shape, dtype=v.dtype, trainable=False, initializer=tf.constant_initializer(0.0)) )
        
    with tf.name_scope("policy_backprop"):
        
        # Generate the training op
        pol_rl_train_op = list()
        for idx in range(len(pol_trainable_variables)):
            gradient_scaled = pol_gradient_vars[idx] * pol_rl_lr * td_error
            pol_rl_train_op.append(tf.assign(pol_trainable_variables[idx], pol_trainable_variables[idx] + pol_gradients_raw[idx] * pol_rl_lr))
        
        # Scale actions by the value function error (and step size!)
        #   positive error means that the state's value was worth more than we expected
        #   negative error means that the state was worth less than we expected)

        # Take the gradients of the chosen action with respect to the parameterization of the function
        pol_grads = tf.gradients(pol_chosen_acts_scaled, pol_variables)



        # Is this how we're supposed to update the step??
        pol_rl_train_op.append(tf.assign(pol_rl_step, pol_rl_step+1))


with tf.name_scope('summaries'):

    weights_pol_dense = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "pv/pol_dense/weights")[0]
    weights_val_dense = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "pv/val_dense/weights")[0]
    biases_pol_dense = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "pv/pol_dense/biases")[0]
    
    # Value
    val_summaries = tf.summary.merge([
        tf.summary.scalar("val_loss", val_loss),
        tf.summary.scalar("val_mean_prediction_error", tf.reduce_mean(td_error)),
        tf.summary.scalar("val_predicted_t0", val_carryover_previous_predicted[0,0]),
        tf.summary.scalar("val_reward_t0", val_actual_reward[0]),
        tf.summary.scalar("val_predicted_t1", val_next_predicted[0,0]),
        tf.summary.histogram(weights_val_dense.name, tf.reshape(weights_pol_dense.value(), [-1])),
        tf.summary.image(weights_val_dense.name, tf.reshape(weights_val_dense.value(), [1,pv_layers[-1], 1, 1]), max_outputs=10, collections=None)
    ])

    # Policy
    pol_summaries =     tf.summary.merge([
            tf.summary.scalar("pol_prediction", pol_out_predict[0]),
            tf.summary.scalar("pol_prediction_dist", pol_out_predict_dist[0]),
            tf.summary.histogram("pol_softmax", pol_out_softmax),
            tf.summary.histogram("pol_predictions", pol_out_predict),
            tf.summary.histogram("pol_predictions_dist", pol_out_predict_dist),
            tf.summary.histogram(weights_pol_dense.name, tf.reshape(weights_pol_dense.value(), [-1])),
            tf.summary.histogram(biases_pol_dense.name, biases_pol_dense.value()),
            tf.summary.image(weights_pol_dense.name, tf.reshape(weights_pol_dense.value(), [1,pv_layers[-1], naudio_commands, 1]), max_outputs=10, collections=None)
            ])
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

def singlestate2batch(state, batchsize):
    return np.repeat(state, batchsize, axis=2)

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

def val_train(sess, writer, state, eeg_data, sessrun_name=''):
    feed = {
            in_raw_eeg_features : eeg_data,
            in_rnn_state : state
            }
    fetch = {
            'rnn_output_state': multicellFinalState,
            'train_op'  : val_train_op,
            'loss'      : val_loss,
            'summaries' : val_summaries,
            'step'      : val_step,
            'assgn_op'  : val_assgn_op0.op # assigns the new predicted value to the old predicted value
            }
    return train(sess,writer,feed,fetch,sessrun_name)

    
def choose_next_action(sess, writer, state, eeg_data, sessrun_name=''):
    feed = {
            in_raw_eeg_features : eeg_data,
            in_rnn_state : state
            }
    fetch = {
            'rnn_output_state': multicellFinalState,
            'prediction'      : pol_out_predict_dist,
            'prediction_old'  : pol_out_predict,
            'softmax'         : pol_out_softmax,
            'summaries'       : pol_summaries,
            'step'            : pol_step
            }

    return predict(sess,writer,feed,fetch,sessrun_name)
    

def pol_imp_train(sess, writer, state, eeg_data, proctor_action, sessrun_name=''):
    feed = {
            in_rnn_state : state,
            in_raw_eeg_features : eeg_data,
            in_action       : proctor_action
            }
    fetch = {
            'rnn_output_state': multicellFinalState,
            'train_op'  : pol_imp_train_op,
            'loss'      : pol_imp_loss,
            'summaries' : pol_imp_summaries,
            'step'      : pol_imp_step
            }
    return train(sess,writer,feed,fetch,sessrun_name)

def pol_rl_train(sess, writer, state, eeg_data, sessrun_name=''):
    feed = {
            in_raw_eeg_features : eeg_data,
            in_rnn_state : state
            }
    fetch = {
            'rnn_output_state': multicellFinalState,
            'train_op'  : pol_rl_train_op,
            'loss'      : val_loss,
            'summaries' : pol_rl_summaries,
            'step'      : pol_rl_step
            }
    return train(sess,writer,feed,fetch,sessrun_name)

def set_rnn_state(sess, writer, state):
    pass
    #    init_state = np.zeros((num_layers, 2, batch_size, state_size))
    # NOT DONE
    # Need to initialize the first state and then update the
    # local variable (containing the hidden state) every time we go through
    # a training session. Also need to pass in this hidden state every time we
    # run an op
    
    # Alternatively, just store a variable with the hidden state. Update this
    # just once per "run"

def rl_train(sess, writer, state, eeg_data, sessrun_name=''):
    feed = {
            in_raw_eeg_features : eeg_data,
            in_rnn_state : state
            }
    pol_fetch = {
            'pol_train_op'  : pol_rl_train_op,
            'pol_loss'      : val_loss,
            'pol_summaries' : pol_rl_summaries,
            'pol_step'      : pol_rl_step
            }
    
    val_fetch = {
            'val_train_op'  : val_train_op,
            'val_loss'      : val_loss,
            'val_summaries' : val_summaries,
            'val_step'      : val_step,
            'val_assgn_op'  : val_assgn_op0.op # assigns the new predicted value to the old predicted value
            }
    merged_fetch = {pol_fetch, val_fetch} # cool python 3.5 trick to merge dictionaries
    return train(sess,writer,feed,merged_fetch,sessrun_name)




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

# Empty data structures
empty_data = np.empty([0,ntimepoints,nchan,nfreqs])
empty_rpv = np.empty([0,shape_rpv[1]])
empty_act = np.empty([0,naudio_commands])

# Batching
imp_training_data = empty_data
imp_training_lbls = empty_act
rl_training_data = empty_data

imp_minimum_batch_size = 100
rl_minimum_batch_size = 1

# Intervals - all in seconds!
interval_send_command = 0.05 # time to send all commands
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

#peaks = range(16,25)
peaks = range(1,20)
#amps = [1.25e0,2.5e0,5e0,10e0,20e0,10e0,5e0,2.5e0,1.25e0]
amps = np.linspace(0,1,19)

f,w = tgt.create_tgt_profile(peaks,amps,nchan, nfreqs, ntimepoints)
w2 = w / np.max(w)
set_target_profile(sess,summary_writer,w,w2,0)

# The variable we use to hold the state
import copy
rnn_state = np.zeros(np_shape_rnn_input)
rnn_state_tmp = copy.copy(rnn_state)

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
            for i in range(1):
                policy_predict_info = pol_predict(sess, summary_writer, rnn_state, raw_data[-1,None]) # select the most recent data but make sure we still have a batch size of 1 (raw_data[-1].shape = [ntimepoints,nchan,nfreq] whereas raw_data[-1,None].shape = [1,ntimepoints,nchan,nfreq]
                act_out = policy_predict_info['prediction'][0]
                rnn_state_tmp = policy_predict_info['rnn_output_state']
                OUTPUT.submit_command_relaxed(act_out)

                # summaries
                summary_writer.add_summary(policy_predict_info['summaries'], global_step=policy_predict_info['step'])
                in_img_summary = sess.run(input_summaries, {in_raw_eeg_features: raw_data[-1,None]})
                summary_writer.add_summary(in_img_summary, global_step=policy_predict_info['step'])

            cmds_sent += 1
    
        # Load training data if any is present
        if len(imp_data) > 0:
            imp_data = imp_data.reshape(np_shape_eeg_input)
            imp_lbls = one_hot(imp_lbls, naudio_commands, len(imp_lbls))#[:,None,:]
            imp_training_data = np.concatenate((imp_training_data,imp_data))
            imp_training_lbls = np.concatenate((imp_training_lbls,imp_lbls))


        # Check to see if we have enough data to train
        if len(rl_training_data) >= rl_minimum_batch_size:
            policy_rl_train_info = pol_rl_train(sess, summary_writer,singlestate2batch(rnn_state, len(rl_training_data)), rl_training_data[-1,None])
            value_train_info = val_train(sess, summary_writer,singlestate2batch(rnn_state, len(rl_training_data)), rl_training_data[-1,None])
            rnn_state_tmp = policy_rl_train_info['rnn_output_state']
            rl_training_data = empty_data
            timekeep_summary_writer ,   do_summaries     = check_timekeep(timekeep_summary_writer,  interval_summary_writer)
            if do_summaries:
                summary_writer.add_summary(policy_rl_train_info['summaries'], global_step=policy_rl_train_info['step'])
                summary_writer.add_summary(value_train_info['summaries'], global_step=value_train_info['step'])
    
    
        if len(imp_training_data) >= imp_minimum_batch_size:
            policy_imp_train_info = pol_imp_train(sess, summary_writer,singlestate2batch(rnn_state, len(imp_training_data)), imp_training_data,imp_training_lbls)
            rnn_state_tmp = policy_imp_train_info['rnn_output_state'][-1]
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

        if (rnn_state != rnn_state_tmp).all():
            rnn_state = np.asarray(rnn_state_tmp)[:,:,-1,None,:]

except KeyboardInterrupt:
    print("Receieved interrupt")
    print("Saving checkpoint")
    saver.save(sess, checkpoint_file, global_step=loop_idx)
    summary_writer.flush()
    INPUT.stop_eeg_thread()
    OUTPUT.close()