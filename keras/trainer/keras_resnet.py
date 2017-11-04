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
		predictions = Dense(132, activation='softmax')(x)
		model = Model(inputs=base_model.input, outputs=predictions)

		for layer in base_model.layers:
			layer.trainable = False

		model.compile(optimizer=Adam(lr=0.01),loss='categorical_crossentropy', metrics=['accuracy'])

		return model


	def get_batches(self, path, gen=image.ImageDataGenerator(), shuffle=True, batch_size=8, class_mode='categorical'):
		return gen.flow_from_directory(path, target_size=(224, 224),
                class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)


	def fit(self, batches, val_batches, nb_epoch=1):
		self.model.fit_generator(batches, 
                             steps_per_epoch=int(batches.samples/batches.batch_size),
                             epochs=nb_epoch,
                             validation_data=val_batches, 
                             validation_steps=int(val_batches.samples/val_batches.batch_size))
