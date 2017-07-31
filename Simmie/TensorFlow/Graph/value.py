# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 16:48:36 2017

@author: marzipan
"""



##
# Value parameters
##
val_tgt_weights = tf.constant([-1.,1.],dtype=tf.float32) #weight the output of tgt_out_softmax in calculating value
val_discount_rate = tf.constant(0.9)
val_lr = 1e-1
val_output_units = 1

with tf.name_scope("val"):
    val_previous_predicted = tf.Variable(0.0, "VAL_PreviousPredicted", dtype=tf.float32)
    val_next_predicted = tf.contrib.layers.fully_connected(inputs=pv_lstm_out, num_outputs=val_output_units, activation_fn=None,scope='val_dense')
    val_actual_reward = tf.reduce_sum(val_tgt_weights * tgt_out_softmax)
    
    val_prediction_error = val_actual_reward - val_previous_predicted
    with tf.name_scope('loss'):
        val_loss = val_prediction_error + (val_discount_rate * val_next_predicted) # need to manage execution order here, this won't work...
    val_step = tf.Variable(0, name='VAL_Step', trainable=False)
    val_optimizer = tf.train.RMSPropOptimizer(learning_rate=val_lr, centered=False, decay=0.8)
    val_train_op = tgt_optimizer.minimize(val_loss, global_step=val_step)

    with tf.name_scope('summaries'):
        val_summaries = tf.summary.merge([
#            tf.summary.scalar("val_step", val_step),
            tf.summary.scalar("val_loss", val_loss[0,0]),
            tf.summary.scalar("val_prediction_error", val_prediction_error),
            tf.summary.scalar("val_previous_predicted", val_previous_predicted),
            tf.summary.scalar("val_current_reward", val_actual_reward),
            tf.summary.scalar("val_next_predicted", val_next_predicted[0,0]),
        ])

    
    with tf.control_dependencies([val_loss]):
        val_assgn_op0 = val_previous_predicted.assign(val_next_predicted[0,0])
    

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
