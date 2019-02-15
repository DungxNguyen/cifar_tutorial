"""A binary to train CIFAR-10 using a single GPU.
Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.
Speed: With batch_size 128.
System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)
Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.
http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf

import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/scratch2/dungn/cifar10_train',
                           """Dir to write logs and checkpoints""")
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Number of batches to run""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Log device placement""")
tf.app.flags.DEFINE_integer('log_frequency', 100,
                            """How ofter to write logs""")


def train():
    # Note that in graph, logits, loss, train_op are all operations
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()

        # Store data in CPU
        with tf.device('/cpu:0'):
            images, labels = cifar10.distorted_inputs()

        dropout = tf.placeholder(tf.float32, name='dropout')

        logits = cifar10.inference(images)

        loss = cifar10.loss(logits, labels)

        # build the train graph
        # return a train operations
        train_op = cifar10.train(loss, global_step)

        class _LoggerHook(tf.train.SessionRunHook):
            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss)

            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = (
                        '%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
                    print(format_str % (datetime.now(), self._step, loss_value,
                                        examples_per_sec, sec_per_batch))

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=FLAGS.train_dir,
                hooks=[
                    tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                    tf.train.NanTensorHook(loss),
                    _LoggerHook()
                ],
                save_checkpoint_steps=25000,
                config=tf.ConfigProto(log_device_placement=FLAGS.
                                      log_device_placement)) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op, feed_dict={dropout: 1.0})


def main(argv=None):
    cifar10.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
