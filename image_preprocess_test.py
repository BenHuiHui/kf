from PIL import Image
import os

train_imgdir = 'transferred_test/'
train_images = os.listdir(train_imgdir)

im_width=224
im_height=224

for img_path in train_images:
	img = Image.open(os.path.join(train_imgdir, img_path))
	img = img.resize((im_width,im_height), Image.ANTIALIAS)
	img.save('test/'+img_path)
