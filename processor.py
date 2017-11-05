import csv
import os
from shutil import copyfile, move
from os import path

DATA_FILE = 'train.csv'

def generate_train_data():
    with open(DATA_FILE, 'r') as f:
        reader = csv.reader(f)
        headers = reader.next()

        for row in reader:
            img_name = row[0]
            label = row[1]
            new_dir = "data/transferred_train/" + label
            if not os.access(new_dir, os.R_OK):
                os.mkdir(new_dir)
            copyfile("transferred_train/"+img_name, new_dir + "/" + img_name)

def generate_validation_data():
    for label in xrange(132):
        source_dir = path.join("data/transferred_train/", str(label))
        if not os.access(source_dir, os.R_OK):
            return
        files = os.listdir(source_dir)
        new_dir = path.join("data/transferred_valid/", str(label))
        os.mkdir(new_dir)
        for filename in files[-30:]:
            move(path.join(source_dir, filename), path.join(new_dir, filename))

generate_validation_data()