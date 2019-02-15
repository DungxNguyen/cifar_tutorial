"""A binary to train CIFAR-10 using multiple GPUs with synchronous updates.

Accuracy:
cifar10_multi_gpu_train.py achieves ~86% accuracy after 100K steps (256
epochs of data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
--------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)
2 Tesla K20m  | 0.13-0.20              | ~84% at 30K steps  (2.5 hours)
3 Tesla K20m  | 0.13-0.18              | ~84% at 30K steps
4 Tesla K20m  | ~0.10                  | ~84% at 30K steps

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import re
import time
import math

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import cifar10
import cifar10_input_dataset

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/scratch2/dungn/cifar10_train',
                           """Where to write checkpoints""")
tf.app.flags.DEFINE_integer('max_steps', 200000,
                            """Number of batches to run""")
tf.app.flags.DEFINE_integer('max_to_keep', 1000,
                            """Number of checkpoints to keep""")
tf.app.flags.DEFINE_integer('num_gpus', 2, """Number of gpus""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Log device placement""")
tf.app.flags.DEFINE_float('dropout', 1.0, """Dropout Keep Probability""")

tf.app.flags.DEFINE_float('init_lr', 0.03, """Initial LR""")
tf.app.flags.DEFINE_float('decay_lr', 0.1, """Decay rate LR""")
tf.app.flags.DEFINE_boolean('use_momentum', True, """Use momentum or not""")
tf.app.flags.DEFINE_float('momentum_alpha', 0.9, """Momentum alpha""")

# Define the sub-dir to store log and checkpoints
train_dir = os.path.normpath(
    os.path.join(FLAGS.train_dir,
                 'batch_{0}'.format(FLAGS.batch_size),
                 'momentum_{0}'.format(FLAGS.use_momentum),
                 'momentum_alpha_{0}'.format(FLAGS.momentum_alpha),
                 'init_lr_{0}'.format(FLAGS.init_lr),
                 'decay_lr_{0}'.format(FLAGS.decay_lr),
                 'dropout_{0}'.format(FLAGS.dropout)))


def tower_loss(scope, images, labels):
    """Calculate the total loss on a single tower running the CIFAR model.

    Args:
      scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
      images: Images. 4D tensor of shape [batch_size, height, width, 3].
      labels: Labels. 1D tensor of shape [batch_size].

    Returns:
       Tensor of shape [] containing the total loss for a batch of data
    """

    logits = cifar10.inference(images)

    _ = cifar10.loss(logits, labels)

    losses = tf.get_collection('losses', scope)

    total_loss = tf.add_n(losses, name='total_loss')

    for l in losses + [total_loss]:
        loss_name = re.sub('%s_[0-9]*/' % cifar10.TOWER_NAME, '', l.op.name)
        tf.summary.scalar(loss_name, l)

    return total_loss


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
    is over individual gradients. The inner list is over the gradient
    calculation for each tower.
    Returns:
    List of pairs of (gradient, variable) where the gradient has been averaged
    across all towers.
    """

    average_grads = []

    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(axis=0, values=grads)

        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def train():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0),
            trainable=False)

        # Calculate learning rate
        # num_batches_per_epoch = (cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
        #                         FLAGS.batch_size / FLAGS.num_gpus)
        num_batches_per_epoch = (
            cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size)
        decay_steps = int(num_batches_per_epoch * cifar10.NUM_EPOCHS_PER_DECAY)
        lr = tf.train.exponential_decay(
            FLAGS.init_lr,
            global_step,
            decay_steps,
            FLAGS.decay_lr,
            staircase=True)

        if (FLAGS.use_momentum):
            opt = tf.train.MomentumOptimizer(lr, FLAGS.momentum_alpha)
        else:
            opt = tf.train.GradientDescentOptimizer(lr)

        train_input = cifar10.distorted_inputs()
        iterator = train_input.make_one_shot_iterator()
        next_element = iterator.get_next()

        dropout = tf.placeholder(tf.float32, name='dropout')

        # Gradients each gpus
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope(
                            '%s_%d' % (cifar10.TOWER_NAME, i)) as scope:
                        # Dequeue 1 batch to the GPU #i
                        image_batch, label_batch = next_element
                        tf.summary.image('images', image_batch)
                        loss = tower_loss(scope, image_batch, label_batch)

                        # Reuse variables for the next tower
                        tf.get_variable_scope().reuse_variables()

                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES,
                                                      scope)

                        grads = opt.compute_gradients(loss)

                        tower_grads.append(grads)

        # Sync across all towers
        grads = average_gradients(tower_grads)

        summaries.append(tf.summary.scalar('learning_rate', lr))

        # Hist of gradients
        for grad, var in grads:
            if grad is not None:
                summaries.append(
                    tf.summary.histogram(var.op.name + '/gradients', grad))

        # Apply gradients to adjust the shared variables
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # hist of trainable vars
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        # track the moving averages of trainable vars
        variable_averages = tf.train.ExponentialMovingAverage(
            cifar10.MOVING_AVERAGE_DECAY, global_step)

        variable_averages_op = variable_averages.apply(
            tf.trainable_variables())

        train_op = tf.group(apply_gradient_op, variable_averages_op)

        saver = tf.train.Saver(
            tf.global_variables(), max_to_keep=FLAGS.max_to_keep)

        summary_op = tf.summary.merge(summaries)

        init = tf.global_variables_initializer()

        sess = tf.Session(
            config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=FLAGS.log_device_placement))

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            test_top_k, test_loss_op = test()

        sess.run(init)

        summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
        for step in xrange(FLAGS.max_steps + 1):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss],
                                     feed_dict={dropout: FLAGS.dropout})

            # print(labels)
            duration = time.time() - start_time

            if step % 100 == 0:
                num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / FLAGS.num_gpus

                format_str = (
                    '%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                    'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch))
            if step % 100 == 0:
                summary = tf.Summary()
                summary_str = sess.run(
                    summary_op, feed_dict={dropout: FLAGS.dropout})
                summary.ParseFromString(summary_str)
                summary_writer.add_summary(summary, step)

            # Save model checkpoint
            if step % 20000 == 0 or step == FLAGS.max_steps:
                checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

                if step != 0:
                    num_iter = int(math.ceil(cifar10_input_dataset.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL / FLAGS.batch_size))
                    true_count = 0
                    total_sample_count = num_iter * FLAGS.batch_size
                    test_step = 0
                    total_loss = 0
                    while test_step < num_iter:
                        predictions, test_loss = sess.run([test_top_k, test_loss_op],
                                                          feed_dict={dropout: 1.0})
                        true_count += np.sum(predictions)
                        test_step += 1
                        total_loss += test_loss

                    # Prec@1
                    precisions = true_count / total_sample_count
                    total_loss = total_loss / total_sample_count

                    print('%s: step %s precision @ 1 = %.3f' % (datetime.now(), step, precisions))
                    
                    precision_summary = tf.Summary(value=[tf.Summary.Value(tag='Precision @ 1',simple_value=precisions)])
                    loss_summary = tf.Summary(value=[tf.Summary.Value(tag='Test Loss', simple_value=total_loss)])
                    summary_writer.add_summary(precision_summary, step)
                    summary_writer.add_summary(loss_summary, step)


def test():
    eval_input = cifar10.inputs(eval_data=True)

    test_iterator = eval_input.make_one_shot_iterator()

    next_test_element = test_iterator.get_next()

    images, labels = next_test_element

    softmax = cifar10.inference(images)

    labels = tf.reshape(labels, [-1])

    top_k_op = tf.nn.in_top_k(softmax, labels, 1)

    loss = cifar10.loss(softmax, labels)

    return top_k_op, loss


def main(argv=None):  # pylint: disable=unused-argument
    cifar10.maybe_download_and_extract()
    if tf.gfile.Exists(train_dir):
        tf.gfile.DeleteRecursively(train_dir)
    tf.gfile.MakeDirs(train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
