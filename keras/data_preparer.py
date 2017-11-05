import csv
import os

number_class = 132
DATA_FILE = 'train.csv'
IMG_DIR = 'transferred_train/'

# Create directory
for i in range(number_class):
    train_directory = 'keras_image/train/'+str(i)
    eval_directory = 'keras_image/eval/'+str(i)

    if not os.path.exists(train_directory):
        os.makedirs(train_directory)

    if not os.path.exists(eval_directory):
        os.makedirs(eval_directory)


with open(DATA_FILE) as f:
    reader = csv.reader(f)
    header = reader.next()

    i = 0

    for row in reader:
        img_path = row[0]
        label = row[1]

        if i % 10 == 0:
            # eval data
            new_img_path = 'keras_image/eval/'+label+'/'+img_path
            os.rename(IMG_DIR+img_path, new_img_path)
        else:
            new_img_path = 'keras_image/train/'+label+'/'+img_path
            os.rename(IMG_DIR+img_path, new_img_path)

        i = i + 1
