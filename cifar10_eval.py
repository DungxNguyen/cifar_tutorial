"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/scratch2/dungn/cifar10_eval',
                           """To write eval logs""")
tf.app.flags.DEFINE_string('eval_data', 'test', """test or train_eval""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/scratch2/dungn/cifar10_train',
                           """Where are checkpoints""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 1,
                            """How often the eval""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run""")
tf.app.flags.DEFINE_boolean('run_once', False, """Run once?""")


def eval_once(saver, summary_writer, top_k_op, summary_op, dropout):
    """Run Eval once.

    Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
    """

    with tf.Session() as sess:
        checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            # restore from checkpoint
            saver.restore(sess, checkpoint.model_checkpoint_path)
            global_step = checkpoint.model_checkpoint_path.split(
                '/')[-1].split('-')[-1]
        else:
            print('No checkpoint found')
            return

        coord = tf.train.Coordinator()
        try:
            threads = []
            for queue_runner in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(
                    queue_runner.create_threads(
                        sess, coord=coord, daemon=True, start=True))
            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            true_count = 0
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([top_k_op], feed_dict={dropout: 1.0})
                true_count += np.sum(predictions)
                step += 1

            # Prec@1
            precisions = true_count / total_sample_count
            print('%s: step %s precision @ 1 = %.3f' % (datetime.now(), global_step, precisions))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op, feed_dict={dropout: 1.0}))
            summary.value.add(tag='Precision @ 1', simple_value=precisions)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    """Eval CIFAR-10 for a number of steps."""
    with tf.Graph().as_default() as graph:
        eval_data = FLAGS.eval_data == 'test'
        images, labels = cifar10.inputs(eval_data=eval_data)

        dropout = tf.placeholder(tf.float32, name='dropout')
        
        logits = cifar10.inference(images)

        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # Restore the moving average version of the learned variables for eval.
        # TODO: Explain why
        variable_averages = tf.train.ExponentialMovingAverage(
            cifar10.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        # Summary operations
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, graph)

        while True:
            eval_once(saver, summary_writer, top_k_op, summary_op, dropout)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):
    cifar10.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)

    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()


if __name__ == '__main__':
    tf.app.run()
