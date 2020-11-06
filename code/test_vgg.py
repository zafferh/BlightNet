from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from config import *

import numpy as np
import os

train_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()
mean = np.array([123.68, 116.779, 103.939], dtype="float32")
train_datagen.mean = mean
val_datagen.mean = mean
test_datagen.mean = mean

base_dir = 'data'
base_train_dir = os.path.join(base_dir, 'train')
base_val_dir = os.path.join(base_dir, 'val')
base_test_dir = os.path.join(base_dir, 'test')

train_generator = train_datagen.flow_from_directory(
	base_train_dir,
	target_size=(IM_SIZE, IM_SIZE),
	batch_size=BATCH_SIZE_BASE,
	class_mode='categorical',
	color_mode='rgb'
)

val_generator = val_datagen.flow_from_directory(
	base_val_dir,
	target_size=(IM_SIZE, IM_SIZE),
	batch_size=BATCH_SIZE_BASE,
	class_mode='categorical',
	color_mode='rgb'
)

test_generator = val_datagen.flow_from_directory(
	base_test_dir,
	target_size=(IM_SIZE, IM_SIZE),
	batch_size=8,
	class_mode='categorical',
	color_mode='rgb'
)

json_file = open('model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(MODEL_DIR_BASE)

opt = SGD(lr=LR_BASE, momentum=MOMENTUM_BASE)
loaded_model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])

_, train_acc = loaded_model.evaluate_generator(train_generator)
_, val_acc = loaded_model.evaluate_generator(val_generator)
_, test_acc = loaded_model.evaluate_generator(test_generator)

print('Train accuracy: {}'.format(train_acc))
print('Val accuracy: {}'.format(val_acc))
print('Test accuracy: {}'.format(test_acc))