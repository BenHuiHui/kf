import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.set_random_seed(42)
from tensorflow.python.lib.io import file_io
from keras_resnet import Resnet

# reset everything to rerun in jupyter
# tf.reset_default_graph()

batch_size = 100
num_classes = 132


def train_model(train_dir, eval_dir, output_dir, eval_size, epochs, **args):
    resnet = Resnet()

    train_batch = resnet.get_batches(train_dir, batch_size=batch_size)
    eval_batch = resnet.get_batches(eval_dir, shuffle=False, batch_size=eval_size)

    resnet.fit(train_batch, eval_batch, epochs)

    resnet.model.save('model.h5')
    
    # Save model.h5 on to google storage
    with file_io.FileIO('model.h5', mode='r') as input_f:
        with file_io.FileIO(output_dir + '/model.h5', mode='w+') as output_f:
            output_f.write(input_f.read())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
      '--train-dir',
      required=True
    )
    parser.add_argument(
      '--eval-dir',
      required=True
    )
    parser.add_argument(
        '--output-dir',
        required=True
    )
    parser.add_argument(
        '--eval-size',
        type=int,
        required=True
    )
    parser.add_argument(
        '--epochs',
        type=int,
        required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__
    
    train_model(**arguments)
