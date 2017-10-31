from collections import OrderedDict
import pandas as pd
import csv
import numpy as np

input_filename = 'test_default_param.csv'

def saveImageList(images, labels, save_to_file):
     dic = OrderedDict()
     dic['image_name'] = images
     dic['category'] = np.array(labels, dtype=np.int32)
     data = pd.DataFrame(dic)
     data.to_csv(save_to_file, index=None, header=['image_name','category'])


images = []
labels = []

with open(input_filename, 'r') as f:
	reader = csv.reader(f)
	for d in reader:
		images.append(d[0])
		labels.append(int(d[1]))

saveImageList(images, labels, 'submission.csv')