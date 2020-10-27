import os
from PIL import Image
from six import BytesIO
from lxml import etree
import tensorflow as tf
import numpy as np

from object_detection.utils import dataset_util


dataset = '../data/ImageSets/Main'
annotations_dir = '../data/Annotations'
ImageDir = '../data/JPEGImages'
classes =  ['Background','Presser Foot', 'Zipper Foot', 'Sliding Button Foot', 'Blind Hem Foot', 'Bobon', 'Seam Ripper', 'Seam Ripper Lid']
num_classes = 8
label_id_offset = 1
train_image_np = []
train_image_tensors = []
bx_tensors = []

def load_image_into_numpy_array(path):
	img_data = tf.io.gfile.GFile(path, 'rb').read()
	image = Image.open(BytesIO(img_data))
	(im_width, im_height) = image.size
	return np.array(image.getdata()).reshape(
		(im_height, im_width, 3)).astype(np.uint8)


class_dict =  {}
for ind, c in enumerate(classes):
	class_dict[c + '_id'] = {'id': ind + 1, 'name': c}

	

image_list = []
label_list = []
classes_one_hot_tensors = []

with open(dataset + '/train.txt', 'r') as f:
	for line in f.readlines():
		image_list.append(line[:-1])

for i in image_list:
	print(i)
	train_image_tensors.append(tf.expand_dims(tf.convert_to_tensor(load_image_into_numpy_array(ImageDir + '/' + i + '.JPG'), dtype=tf.float32), axis=0))
	
	root = etree.parse(annotations_dir + '/' + i + '.xml').getroot()
	
	img_size = root.find('size')
	img_width = int(img_size.find('width').text)
	img_height = int(img_size.find('height').text)

	for obj in root.findall('object'):
		class_name = obj.find('name').text
		class_ind = class_dict[class_name + '_id']['id']
		bnd_box = obj.find('bndbox')
		xmin = int(bnd_box.find('xmin').text)
		xmax = int(bnd_box.find('xmax').text)
		ymin = int(bnd_box.find('ymin').text)
		ymax = int(bnd_box.find('ymax').text)

		f_x = xmin/float(img_width)
		f_y = ymin/float(img_height)

		f_xmax = xmax/float(img_width)
		f_ymax = ymax/float(img_height)

		#f_height = f_ymax - f_y
		#f_width = f_xmax - f_x

		bx_tensors.append(tf.convert_to_tensor([[f_x, f_y, f_xmax, f_ymax]],dtype=np.float32))
		zero_indexed_classes = tf.convert_to_tensor(np.array([class_ind], dtype=np.int) - label_id_offset)
		classes_one_hot_tensors.append(tf.one_hot(zero_indexed_classes, num_classes))
