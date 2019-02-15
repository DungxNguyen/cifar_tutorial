"""Builds the CIFAR-10 network.
Summary of available functions:
 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()
 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)
 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)
 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

import cifar10_input_dataset

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/scratch2/dungn/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

IMAGE_SIZE = cifar10_input_dataset.IMAGE_SIZE
NUM_CLASSES = cifar10_input_dataset.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input_dataset.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input_dataset.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN

MOVING_AVERAGE_DECAY = 0.9999
NUM_EPOCHS_PER_DECAY = 125.0

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    Args:
    x: Tensor
    Returns:
    nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
    Returns:
    Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(
            name, shape, initializer=initializer, dtype=dtype)

    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
    Returns:
    Variable Tensor
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name, shape, tf.truncated_normal_initializer(
            stddev=stddev, dtype=dtype))

    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)

    return var


def distorted_inputs():
    """Construct distorted input for CIFAR training using the Reader ops.
    Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
    Raises:
    ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')

    return cifar10_input_dataset.distorted_inputs(
        data_dir=data_dir, batch_size=FLAGS.batch_size)


def inputs(eval_data):
    """Construct input for CIFAR evaluation using the Reader ops.
    Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
    Raises:
    ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')

    return cifar10_input_dataset.inputs(
        eval_data=eval_data, data_dir=data_dir, batch_size=FLAGS.batch_size)


def inference(images):
    """Build the CIFAR-10 model.
    Args:
    images: Images returned from distorted_inputs() or inputs().
    Returns:
    Logits.
    """

    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #
    #print(images)
    images.set_shape([FLAGS.batch_size, 24, 24, 3])
    #print(images)

    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay(
            'weights', shape=[5, 5, 3, 64], stddev=5e-2, wd=None)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv1)

    # pool1
    pool1 = tf.nn.max_pool(
        conv1,
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding='SAME',
        name='pool1')

    # norm1
    norm1 = tf.nn.lrn(
        pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay(
            'weights', shape=[5, 5, 64, 64], stddev=5e-2, wd=None)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv2)

    norm2 = tf.nn.lrn(
        conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

    pool2 = tf.nn.max_pool(
        norm2,
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding='SAME',
        name='pool2')

    #with tf.variable_scope('fully3') as scope:
    #    # TODO: Change it from fully connected to locally connected
    #    # Move everything into depth so we can perform a single matrix multiply.
    #    reshape = tf.reshape(pool2, [images.get_shape().as_list()[0], -1])
    #    dim = reshape.get_shape()[1].value
    #    weights = _variable_with_weight_decay('weights', shape=[dim, 384], stddev=0.04, wd=0.004)
    #    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    #    fully3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    #    _activation_summary(fully3)

    #with tf.variable_scope('fully4') as scope:
    #    dim = fully3.get_shape()[1].value
    #    weights = _variable_with_weight_decay('weights', shape=[dim, 192], stddev=0.04, wd=0.004)
    #    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.0))
    #    fully4 = tf.nn.relu(tf.matmul(fully3, weights) + biases, name=scope.name)
    #    _activation_summary(fully4)

    #with tf.variable_scope('flat_local3') as scope:
    #    # Implement 1-D local layer
    #    local_connections = 32
    #    strides = 4
    #    reshape = tf.reshape(pool2, [images.get_shape().as_list()[0], -1])
    #    dim = reshape.get_shape()[1].value
    #    patches = tf.squeeze(tf.extract_image_patches(reshape[..., None, None],
    #                                       ksizes=[1, local_connections, 1, 1],
    #                                       strides=[1, strides, 1, 1],
    #                                       rates=[1, 1, 1, 1],
    #                                       padding='SAME'))
    #    dim_out = patches.get_shape()[1].value
    #    weights = _variable_with_weight_decay('weights', shape=[dim_out, local_connections], stddev=0.04, wd=0.004)
    #    biases = _variable_on_cpu('biases', [dim_out], tf.constant_initializer(0.1))
    #    local3 = tf.nn.relu(tf.reduce_sum(tf.multiply(patches, weights), 2) + biases, name=scope.name)
    #    _activation_summary(local3)

    #with tf.variable_scope('flat_local4') as scope:
    #    # Implement 1-D local layer
    #    local_connections = 32
    #    strides = 4
    #    dim = local3.get_shape()[1].value
    #    patches = tf.squeeze(tf.extract_image_patches(local3[..., None, None],
    #                                       ksizes=[1, local_connections, 1, 1],
    #                                       strides=[1, strides, 1, 1],
    #                                       rates=[1, 1, 1, 1],
    #                                       padding='SAME'))
    #    dim_out = patches.get_shape()[1].value
    #    weights = _variable_with_weight_decay('weights', shape=[dim_out, local_connections], stddev=0.04, wd=0.004)
    #    biases = _variable_on_cpu('biases', [dim_out], tf.constant_initializer(0.1))
    #    local4 = tf.nn.relu(tf.reduce_sum(tf.multiply(patches, weights), 2) + biases, name=scope.name)
    #    _activation_summary(local4)

    # with tf.variable_scope('local3') as scope:
    #     # Implement 2-D local layer
    #     patches = tf.extract_image_patches(
    #         pool2,
    #         ksizes=[1, 3, 3, 1],
    #         strides=[1, 1, 1, 1],
    #         rates=[1, 1, 1, 1],
    #         padding='SAME')
    #     print(patches.get_shape())
    #     dim_out_depth = patches.get_shape()[3].value
    #     dim_out_width = patches.get_shape()[2].value
    #     dim_out_height = patches.get_shape()[1].value
    #     weights = _variable_with_weight_decay(
    #         'weights',
    #         shape=[dim_out_height, dim_out_width, dim_out_depth],
    #         stddev=0.04,
    #         wd=0.004)
    #     biases = _variable_on_cpu('biases', [dim_out_height, dim_out_width],
    #                               tf.constant_initializer(0.1))
    #     local3 = tf.nn.relu(
    #         tf.reduce_sum(tf.multiply(patches, weights), 3) + biases,
    #         name=scope.name)
    #     _activation_summary(local3)

    with tf.variable_scope('fully4') as scope:
        reshape = tf.reshape(pool2, [pool2.get_shape().as_list()[0], -1])
        # reshape = tf.reshape(local3, [local3.get_shape().as_list()[0], -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay(
            'weights', shape=[dim, 384], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [384],
                                  tf.constant_initializer(0.0))
        fully4 = tf.nn.dropout(tf.nn.relu(
            tf.matmul(reshape, weights) + biases, name=scope.name),
                               keep_prob=tf.get_default_graph().get_tensor_by_name('dropout:0'))
        _activation_summary(fully4)

    with tf.variable_scope('fully5') as scope:
        dim = fully4.get_shape()[1].value
        weights = _variable_with_weight_decay(
            'weights', shape=[dim, 192], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [192],
                                  tf.constant_initializer(0.0))
        fully5 = tf.nn.dropout(tf.nn.relu(
            tf.matmul(fully4, weights) + biases, name=scope.name),
                               keep_prob=tf.get_default_graph().get_tensor_by_name('dropout:0'))
        _activation_summary(fully5)
    # linear layer(WX + b),
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.

    with tf.variable_scope('softmax') as scope:
        dim = fully5.get_shape()[1].value
        weights = _variable_with_weight_decay(
            'weights', [dim, NUM_CLASSES], stddev=1 / 192.0, wd=None)
        biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                  tf.constant_initializer(0.0))
        #softmax_linear = tf.add(tf.matmul(fully4, weights), biases, name=scope.name)
        softmax_linear = tf.add(
            tf.matmul(fully5, weights), biases, name=scope.name)
        softmax = tf.nn.softmax(softmax_linear, name=scope.name)
        _activation_summary(softmax)

    # TODO: Create softmax units
    # Done
    # TODO: Why all variables are in CPU but not GPU?
    # ANS: Hmm, then multiple GPUs can access to them more easily. If there is 1
    # GPU in training, we can put the variables to the GPU

    return softmax


def loss(softmax, labels):
    """Add L2Loss to all the trainable variables.
    Add summary for "Loss" and "Loss/avg".

    Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
    of shape [batch_size]

    Returns:
    Loss tensor of type float.
    """

    # TODO: What it the type of labels
    # ANS: A number. Then we should convert to one_hot vector to calculate softmax

    # labels = tf.cast(labels, tf.int64)
    #print("softmax shape", softmax)
    labels = tf.reshape(labels, [FLAGS.batch_size])
    #print("Label reshape new", labels)
    labels_one_hot = tf.one_hot(labels, cifar10_input_dataset.NUM_CLASSES)
    #print("One hot", labels_one_hot)
    # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy = -tf.reduce_sum(
        tf.multiply(labels_one_hot, tf.log(tf.clip_by_value(softmax, 1e-10, 1.0))), 1)
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # Add L2 weight regularization to the loss:
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
    total_loss: Total loss from loss().
    Returns:
    loss_averages_op: op for generating moving averages of losses.
    """
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # TODO: Revew this function
    for i in losses + [total_loss]:
        tf.summary.scalar(i.op.name + '(raw)', i)
        tf.summary.scalar(i.op.name, loss_averages.average(i))

    return loss_averages_op


def train(total_loss, global_step):
    """Train CIFAR-10 model.

    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.

    Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
    processed.
    Returns:
    train_op: op for training.
    # TODO Why does it return variable_averages_op?
    # ANS: Because we add apply_gradient_op as dependencies of variable_averages_op, so when the session executes variable_averages_op, it requires apply_gradient_op to be executed first
    """

    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    print('decay steps %d' % (decay_steps))

    # Decay the learning rate
    lr = tf.train.exponential_decay(
        FLAGS.init_lr,
        global_step,
        decay_steps,
        FLAGS.decay_lr,
        staircase=True)
    tf.summary.scalar('learning_rate', lr)

    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients
    # TODO: search about function control_dependencies
    # ANS: force computing gradients after summarizing all losses
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        # Compute gradients from total_loss
        grads = opt.compute_gradients(total_loss)

    #Apply gradients
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    #Histogram for gradients
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    #Track the moving average of all trainable variables
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    with tf.control_dependencies([apply_gradient_op]):
        variable_averages_op = variable_averages.apply(
            tf.trainable_variables())

    return variable_averages_op


def maybe_download_and_extract():
    """Download and extract the tarball from Alex's website."""
    dest_directory = FLAGS.data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)

    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):

        def _progress(count, block_size, total_size):
            sys.stdout.write(
                '\r>> Downloading %s %.1f%%' %
                (filename,
                 float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)
