# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 16:45:50 2017

@author: marzipan
"""


import time
import tensorflow as tf
import numpy as np

from Simmie.InterfaceAdapters.Output_Adapters.Audio_Command_Adapter import AudioCommandAdapter
from Simmie.InterfaceAdapters.Input_Adapters.EEG_State_Adapter import EEGStateAdapter


nchan = 8
nfreqs = 30
ntimepoints = 10

OUTPUT = AudioCommandAdapter(
        name="Simmie",
        uid=np.random.randint(0,1e4),
        relaxed_enabled=True)

INPUT = EEGStateAdapter(
        n_freq=nfreqs, #should match with pipeline
        n_chan=nchan, #should match with pipeline
        eeg_feed_rate=10, #sps
        samples_per_output=1, # 
        spectrogram_timespan=10, #assumed to be in seconds
        n_spectrogram_timepoints=ntimepoints)


_ = input("\nACA started, now start up audio control GUI...\nPress enter when ready.")
print("Continuing.")



G_logs = 'C:\\Users\\marzipan\\workspace\\Simmie\Experimental\Logs\\'
G_logdir = G_logs + 'S5\\'
#time.strftime("%m-%d_%H.%M_")

checkpoint_file = G_logdir + 'model.ckpt'



'''

All input parameters the next receives

'''
naudio_commands = len(OUTPUT.get_valid_audio_commands())
shape_eeg_feat = [None,1,nfreqs * ntimepoints * nchan]
shape_rpv = [None,2]
shape_act = [None,naudio_commands]

with tf.name_scope("in"):
    in_eeg_features = tf.placeholder(tf.float32, shape=shape_eeg_feat, name="IN_EEG")
    in_rpv = tf.placeholder(tf.int32, shape=shape_rpv, name="IN_RPV")
    in_action = tf.placeholder(tf.int32, shape=shape_act, name="IN_ACT")
    
    
    




###
# Summaries
###
summary_op = tf.summary.merge_all()

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

def tgt_predict(sess, writer, eeg_data, sessrun_name=''):
    feed = {in_eeg_features : eeg_data}
    fetch = {
            'prediction'      : tgt_out_predict,
            'softmax'         : tgt_out_softmax }
    return predict(sess,writer,feed,fetch,sessrun_name)


def tgt_train(sess, writer, eeg_data, rpv, sessrun_name=''):
    feed = {
            in_eeg_features    : eeg_data,
            in_rpv             : rpv       }
    fetch = {
            'train_op'  : tgt_train_op,
            'loss'      : tgt_loss,
            'summaries' : tgt_summaries,
            'step'      : tgt_step
            }
    return train(sess,writer,feed,fetch,sessrun_name)


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
tgt_training_data = empty_data
tgt_training_lbls = empty_rpv
rl_training_data = empty_data
tgt_minimum_batch_size = 5
imp_minimum_batch_size = 10
rl_minimum_batch_size = 10

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
tgt_train_info = None


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

        if len(tgt_data) > 0:
            tgt_data = np.reshape(tgt_data,(-1,1, ntimepoints * nchan * nfreqs))
            tgt_training_data = np.concatenate((tgt_training_data,tgt_data))
            tgt_training_lbls = np.concatenate((tgt_training_lbls,tgt_lbls))
    
        
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



        if len(tgt_training_data) >= tgt_minimum_batch_size:
            tgt_train_info = tgt_train(sess, summary_writer, tgt_training_data, tgt_training_lbls)
            tgt_training_data = empty_data
            tgt_training_lbls = empty_rpv
            timekeep_summary_writer2 ,   do_summaries2     = check_timekeep(timekeep_summary_writer2,  interval_summary_writer)
            if do_summaries2:
                summary_writer.add_summary(tgt_train_info['summaries'], global_step=tgt_train_info['step'])


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
