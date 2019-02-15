from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange

import tensorflow as tf

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
NUM_PREPROCESS_THREADS = 40


def read_raw_data(data):
    record_bytes = tf.decode_raw(data, tf.uint8)

    label = tf.cast(tf.strided_slice(record_bytes, [0], [1]), tf.int32)

    depth_major = tf.reshape(
        tf.strided_slice(record_bytes, [1], [1 + 32 * 32 * 3]), [3, 32, 32])

    uint8image = tf.transpose(depth_major, [1, 2, 0])

    return uint8image, label


def distorted_transform(raw_data):
    uint8image, label = read_raw_data(raw_data)

    reshaped_input = tf.cast(uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Here we apply many different distortion to training images
    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.random_crop(reshaped_input, [height, width, 3])

    # flip
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # brightness, may be ineffective because image standardization
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    distorted_image = tf.image.random_contrast(
        distorted_image, lower=0.2, upper=1.8)

    float_image = tf.image.per_image_standardization(distorted_image)
    float_image.set_shape([height, width, 3])
    label.set_shape([1])

    return float_image, label


def input_transform(raw_data):
    uint8image, label = read_raw_data(raw_data)

    reshaped_image = tf.cast(uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    resized_image = tf.image.resize_image_with_crop_or_pad(
        reshaped_image, height, width)

    float_image = tf.image.per_image_standardization(resized_image)

    float_image.set_shape([height, width, 3])

    label.set_shape([1])

    return float_image, label


def read_cifar10_dataset(filenames):
    """Reads and parses examples from CIFAR10 data files.
    Recommendation: if you want N-way read parallelism, call this function
    N times.  This will give you N independent Readers reading different
    files & positions within those files, which will give better mixing of
    examples.
    Args:
    filename_queue: A queue of strings with the filenames to read from.
    Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
    """
    label_bytes = 1
    height = 32
    width = 32
    depth = 3
    image_bytes = height * width * depth

    record_bytes = label_bytes + image_bytes

    raw = tf.data.FixedLengthRecordDataset(
        filenames=filenames, record_bytes=record_bytes)

    return raw


def distorted_inputs(data_dir, batch_size):
    """Construct distorted input for CIFAR training using the Reader ops.
    Args:
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.
    Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
    """
    filenames = [
        os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in xrange(1, 6)
    ]

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Fail to find file ' + f)

    with tf.name_scope('data_augmentation'):
        read_input = read_cifar10_dataset(filenames)

        input_dataset = read_input.map(
            distorted_transform, num_parallel_calls=NUM_PREPROCESS_THREADS)

        min_fraction_of_examples_in_queue = 0.4

        min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                                 min_fraction_of_examples_in_queue)

    return input_dataset.repeat().batch(batch_size).shuffle(
        buffer_size=min_queue_examples).prefetch(
            buffer_size=min_queue_examples)


def inputs(eval_data, data_dir, batch_size):
    """Construct input for CIFAR evaluation using the Reader ops.
    Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.
    Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
    """

    if not eval_data:
        filenames = [
            os.path.join(data_dir, 'data_batch_%d.bin' % i)
            for i in xrange(1, 6)
        ]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = [os.path.join(data_dir, 'test_batch.bin')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    with tf.name_scope('input'):
        read_input = read_cifar10_dataset(filenames)

        input_dataset = read_input.map(
            input_transform, num_parallel_calls=NUM_PREPROCESS_THREADS)

        min_fraction_of_examples_in_queue = 0.4

        min_queue_examples = int(
            num_examples_per_epoch * min_fraction_of_examples_in_queue)

    return input_dataset.repeat().batch(batch_size).prefetch(
        buffer_size=min_queue_examples)
