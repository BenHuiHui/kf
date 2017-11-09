from keras.models import load_model
import csv
import numpy as np
from collections import OrderedDict
import pandas as pd
import argparse
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model

def get_batches(path, gen=image.ImageDataGenerator(), shuffle=False, batch_size=64, class_mode='categorical'):
    return gen.flow_from_directory(path, target_size=(299, 299),
                                   class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)


def saveImageList(images, labels, save_to_file):
    dic = OrderedDict()
    dic['image_name'] = images
    dic['category'] = np.array(labels, dtype=np.int32)
    data = pd.DataFrame(dic)
    data.to_csv(save_to_file, index=None, header=['image_name', 'category'])

def createModel(model_path):
    base_model = InceptionV3(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(132, activation='softmax')(x)

    # this is the model we will train
    model = Model(input=base_model.input, output=predictions)
    model.load_weights(model_path)
    return model

def preprocess_input(x):
    x /= 255.
    return x

def predict(output_dir, test_dir, model_path):
    res_path = output_dir + '/submit.csv'


    model = createModel(model_path)

    # model = load_model(model_path)
    X = None
    for i in range(0, 8):
        img = image.load_img(test_dir + "/" + str(i)+ ".jpg", target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        if X is None:
            X =x
        else:
            X = np.concatenate([X, x])
    preds = model.predict(X)

    # datagen = image.ImageDataGenerator(
    #     rescale=1. / 255,
    # )
    # test_batches = get_batches(test_dir, datagen)
    # preds = model.predict_generator(test_batches, int(test_batches.samples/test_batches.batch_size)+1)

    # labels = [np.argmax(np.array(l)) for l in preds]
    labels = preds.argmax(axis=-1)
    images = np.array([(str(i) + ".jpg") for i in range(len(labels))])
    saveImageList(images, labels, res_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
      '--output-dir',
      required=True
    )
    parser.add_argument(
      '--test-dir',
      required=True
    )
    parser.add_argument(
      '--model-path',
      required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__

    predict(**arguments)