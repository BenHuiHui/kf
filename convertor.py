from collections import OrderedDict
import pandas as pd
import csv
import numpy as np

def saveImageList(images, labels, save_to_file):
     dic = OrderedDict()
     dic['image_name'] = images
     dic['category'] = np.array(labels, dtype=np.int32)
     data = pd.DataFrame(dic)
     data.to_csv(save_to_file, index=None, header=['image_name','category'])


def generate_eva_csv(split):
	images = []
	labels = []

	train_images = []
	train_labels = []

	validate_images = []
	validate_labels = []

	with open('train.csv', 'r') as f:
		reader = csv.reader(f)
		header = reader.next()

		for d in reader:
			images.append('gs://cs5242-bucket/train_img/' + d[0])
			labels.append(int(d[1]))

	for i in range(len(images)):
		if i % split == 0:
			validate_images.append(images[i])
			validate_labels.append(labels[i])
		else:
			train_images.append(images[i])
			train_labels.append(labels[i])

	with open('train_input.csv', 'w') as f:
		writer = csv.writer(f, delimiter=',')
		for i in range(len(train_images)):
			writer.writerow([train_images[i], train_labels[i]])

	with open('validate_input.csv', 'w') as f:
		writer = csv.writer(f, delimiter=',')
		for i in range(len(validate_images)):
			writer.writerow([validate_images[i], validate_labels[i]])


num_classes = 132
def generate_dict():
	with open('dict.txt', 'w') as f:
		for i in range(num_classes):
			f.write(str(i)+'\n')


# generate_dict()
generate_eva_csv(9)