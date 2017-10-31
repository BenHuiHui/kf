import numpy as np
import tensorflow as tf
import os
import csv
import math

DATASET_PATH = 'transferred_test/'

# Image Parameters
N_CLASSES = 132 # CHANGE HERE, total number of classes
IMG_HEIGHT = 64 # CHANGE HERE, the image height to be resized to
IMG_WIDTH = 64 # CHANGE HERE, the image width to be resized to
CHANNELS = 3 # The 3 color channels, change to 1 if grayscale

print tf.__version__
# TOTAL_IMG = 48871
TOTAL_IMG = 40


def read_images(batch_size):
    imagepaths, labels = list(), list()

    # for i in range(TOTAL_IMG):
    imagepaths.append(DATASET_PATH + "0.jpg")
    labels.append(0)

    # Convert to Tensor
    imagepaths = tf.convert_to_tensor(imagepaths, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)

    # Build a TF Queue, shuffle data
    image, label = tf.train.slice_input_producer([imagepaths, labels],
                                                 shuffle=False)

    # Read images from disk
    image = tf.read_file(image)
    image = tf.image.decode_jpeg(image, channels=CHANNELS)

    # Do all the image preprocessing here.

    # Resize images to a common size
    image = tf.image.resize_images(image, [IMG_HEIGHT, IMG_WIDTH])

    # Normalize
    image = image * 1.0 / 127.5 - 1.0

    # Create batches
    X, Y = tf.train.batch([image, label], batch_size=batch_size,
                          capacity=batch_size * 8,
                          num_threads=4)
    return X, Y

# Create model
def conv_net(x, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):

        #kernel = tf.get_variable('convo2d/kernel')
        #print kernel
        # kernel = tf.get_variable('ConvNet/conv2d/kernel:0')
        # print kernel.eval()

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 32 filters and a kernel size of 5
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)
        # Because 'softmax_cross_entropy_with_logits' already apply softmax,
        # we only apply softmax to testing network
        out = tf.nn.softmax(out) if not is_training else out

    return out

# Parameters
batch_size = 1
display_step = 100

# Network Parameters
dropout = 0.75 # Dropout, probability to keep units

# Build the data input
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
X, _ = read_images(batch_size)

# Because Dropout have different behavior at training and prediction time, we
# need to create 2 distinct computation graphs that share the same weights.

# Create another graph for testing that reuse the same weights
logits_test = conv_net(X, N_CLASSES, dropout, reuse=False, is_training=False)

# Evaluate model (with test logits, for dropout to be disabled)
pred = tf.argmax(logits_test, 1)

# Initialize the variables (i.e. assign their default value)
# init = tf.global_variables_initializer()

# tf.reset_default_graph()
# def main(argv=None):
x = tf.placeholder(tf.int32, [1, None], name='input_placeholder')
x = tf.placeholder(tf.int32, [None, 132], name='input_placeholder')

# Start training
with tf.Graph().as_default() as g:
    with tf.Session() as sess:

        saver = tf.train.import_meta_graph('my_tf_model.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        print([val.name for val in tf.global_variables()])
        # print(tf.global_variables()[0])

        # Start the data queue
        tf.train.start_queue_runners()

        # print 'Loaded kernel'
        # print sess.run('ConvNet/conv2d/kernel:0', feed_dict= {})

        final_res = list()

        for step in range(1, int(math.ceil(float(TOTAL_IMG)/batch_size))+1):
            res = sess.run(pred)
            # if step % display_step == 0:
            print(res[0])
            final_res.extend(res[0])
        print(final_res)
        writer = csv.writer(open("test.csv", "wb"))
        for idx, res in enumerate(final_res):
            writer.writerow([str(idx) + ".jpg", res])

# if __name__ == '__main__':
#    main()