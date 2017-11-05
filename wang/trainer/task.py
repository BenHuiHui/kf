from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
import numpy as numpy
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.models import Model
from keras.optimizers import SGD, RMSprop, Adam

from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing import image
from keras.models import Model
from tensorflow.contrib.training.python.training import hparam
import csv
import numpy as np
import tensorflow as tf
import os
import argparse

class Resnet():
    def __init__(self):
        self.model = self.createResnet()

    def createResnet(self):
        base_model = ResNet50(weights = 'imagenet', include_top = False, input_shape = (224,224,3), input_tensor = None, pooling = None)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.3)(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.3)(x)
        # predictions = Dense(132, activation='softmax')(x)
        predictions = Dense(2, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        for layer in base_model.layers:
            layer.trainable = False

        model.compile(optimizer=Adam(lr=0.01),loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def get_batches(self, path, gen=image.ImageDataGenerator(), shuffle=True, batch_size=8, class_mode='categorical'):
        print(path)
        return gen.flow_from_directory(path, target_size=(224, 224),
                class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)

    def fit(self, batches, val_batches, nb_epoch=1):

        self.model.fit_generator(batches,
                             steps_per_epoch=int(batches.samples/batches.batch_size),
                             epochs=nb_epoch,
                             validation_data=val_batches,
                             validation_steps=int(val_batches.samples/val_batches.batch_size))

    def predict(self, batches):
        prediction = self.model.predict_generator(batches, 1)
        with open("prediction.csv", "w") as f:
            p_writer = csv.writer(f, delimiter=',', lineterminator='\n')
            for idx, p in prediction:
                p_writer.writerow([str(idx) + ".jpg", str(int(p))])

# TRAIN_DATA = "data/transferred_train2/"
# VALIDATION_DATA = "data/transferred_valid2/"
# TEST_DATA = "data/transferred_test2/"

def train_and_predict(train_data, eval_data, test_data, num_epochs):
    r = Resnet()
    r.createResnet()
    batches_train = r.get_batches(train_data)
    batches_eval = r.get_batches(eval_data)
    batches_test = r.get_batches(test_data)
    r.fit(batches_train, batches_eval, num_epochs)
    r.predict(batches_test)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # Input Arguments
  parser.add_argument(
      '--train-files',
      help='GCS or local paths to training data',
      nargs='+',
      required=True
  )
  parser.add_argument(
      '--num-epochs',
      help="""\
      Maximum number of training data epochs on which to train.
      If both --max-steps and --num-epochs are specified,
      the training job will run for --max-steps or --num-epochs,
      whichever occurs first. If unspecified will run for --max-steps.\
      """,
      type=int,
  )
  parser.add_argument(
      '--train-batch-size',
      help='Batch size for training steps',
      type=int,
      default=40
  )
  parser.add_argument(
      '--eval-batch-size',
      help='Batch size for evaluation steps',
      type=int,
      default=40
  )
  parser.add_argument(
      '--eval-files',
      help='GCS or local paths to evaluation data',
      nargs='+',
      required=True
  )
  parser.add_argument(
      '--test-files',
      help='GCS or local paths to evaluation data',
      nargs='+',
      required=True
  )
  # Training arguments
  parser.add_argument(
      '--embedding-size',
      help='Number of embedding dimensions for categorical columns',
      default=8,
      type=int
  )
  parser.add_argument(
      '--first-layer-size',
      help='Number of nodes in the first layer of the DNN',
      default=100,
      type=int
  )
  parser.add_argument(
      '--num-layers',
      help='Number of layers in the DNN',
      default=4,
      type=int
  )
  parser.add_argument(
      '--scale-factor',
      help='How quickly should the size of the layers in the DNN decay',
      default=0.7,
      type=float
  )
  parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
  )

  # Argument to turn on all logging
  parser.add_argument(
      '--verbosity',
      choices=[
          'DEBUG',
          'ERROR',
          'FATAL',
          'INFO',
          'WARN'
      ],
      default='INFO',
  )
  # Experiment arguments
  parser.add_argument(
      '--eval-delay-secs',
      help='How long to wait before running first evaluation',
      default=10,
      type=int
  )
  parser.add_argument(
      '--min-eval-frequency',
      help='Minimum number of training steps between evaluations',
      default=None,  # Use TensorFlow's default (currently, 1000 on GCS)
      type=int
  )
  parser.add_argument(
      '--train-steps',
      help="""\
      Steps to run the training job for. If --num-epochs is not specified,
      this must be. Otherwise the training job will run indefinitely.\
      """,
      type=int
  )
  parser.add_argument(
      '--eval-steps',
      help='Number of steps to run evalution for at each checkpoint',
      default=100,
      type=int
  )
  parser.add_argument(
      '--export-format',
      help='The input format of the exported SavedModel binary',
      choices=['JSON', 'CSV', 'EXAMPLE'],
      default='JSON'
  )

  args = parser.parse_args()

  hparams = hparam.HParams(**args.__dict__)
  train_and_predict(hparams.train_files[0], hparams.eval_files[0], hparams.test_files[0], hparams.num_epochs)
