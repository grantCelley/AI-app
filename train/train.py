import os
from PIL import Image
from six import BytesIO
from lxml import etree
import tensorflow as tf
import numpy as np

from object_detection.utils import dataset_util, config_util
from object_detection.builders import model_builder


dataset = '../data/ImageSets/Main'
annotations_dir = '../data/Annotations'
ImageDir = '../data/JPEGImages'
classes =  ['Background','Presser Foot', 'Zipper Foot', 'Sliding Button Foot', 'Blind Hem Foot', 'Bobon','Screw Driver', 'Seam Ripper', 'Seam Ripper Lid']
num_classes = len(classes)
print(num_classes)
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
		zero_indexed_classes = tf.convert_to_tensor(class_ind - label_id_offset, dtype=tf.int32)
		classes_one_hot_tensors.append(tf.one_hot(zero_indexed_classes, num_classes))

tf.keras.backend.clear_session()
pipline_config = '/models/research/object_detection/configs/tf2/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.config'
cheackpoint_path = './ckpt-0'

configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
model_config.ssd.num_classes = num_classes
model_config.ssd.freeze_batchnorm = True
detection_model = model_builder.build(
	model_config=model_config, is_training=True)

fake_box_predictor = tf.compat.v2.train.Checkpoint(
    _base_tower_layers_for_heads=detection_model._box_predictor._base_tower_layers_for_heads,
     _box_prediction_head=detection_model._box_predictor._box_prediction_head,
    )

fake_model = tf.compat.v2.train.Checkpoint(
          _feature_extractor=detection_model._feature_extractor,
          _box_predictor=fake_box_predictor)
ckpt = tf.compat.v2.train.Checkpoint(model=fake_model)
ckpt.restore(checkpoint_path).expect_partial()

image, shapes = detection_model.preprocess(tf.zeros([1, 640, 640, 3]))
prediction_dict = detection_model.predict(image, shapes)
_ = detection_model.postprocess(prediction_dict, shapes)
print('Weights restored!')

tf.keras.backend.set_learning_phase(True)
