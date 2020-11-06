from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg16
from keras.layers import Dense, Input, Dropout, Flatten
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from config import *
from imutils import paths

import pickle
import numpy as np
import os

base_dir = 'data'
base_train_dir = os.path.join(base_dir, 'train')
base_test_dir = os.path.join(base_dir, 'val')

total_train = len(list(paths.list_images(base_train_dir)))
total_val = len(list(paths.list_images(base_test_dir)))

train_datagen = ImageDataGenerator(
    samplewise_center=False,  # Standardization
    samplewise_std_normalization=False,  # Divide each input by its std (Standardization)
    horizontal_flip=True,  # Randomly flip input horizontally
    vertical_flip=False,  # Randomly flip input vertically
    height_shift_range=0.2,  # random shifting by the range
    rotation_range=10,  # Degree range for random rotations,
    fill_mode='reflect',  # how to fill points outside of the boundaries
    shear_range=0.2,  # range for random shear intensity
    zoom_range=0.15
)

val_datagen = ImageDataGenerator()

mean = np.array([123.68, 116.779, 103.939], dtype="float32")
train_datagen.mean = mean
val_datagen.mean = mean

train_generator = train_datagen.flow_from_directory(
	base_train_dir,
	target_size=(IM_SIZE, IM_SIZE),
	batch_size=BATCH_SIZE_BASE,
	class_mode='categorical',
	color_mode='rgb'
)

val_generator = val_datagen.flow_from_directory(
	base_test_dir,
	target_size=(IM_SIZE, IM_SIZE),
	batch_size=BATCH_SIZE_BASE,
	class_mode='categorical',
	color_mode='rgb'
)


base_pretrained_model = vgg16.VGG16(input_tensor=Input(shape=(IM_SIZE,IM_SIZE,3)), include_top=False, weights='imagenet')

head_model = base_pretrained_model.output
head_model = Flatten(name="flatten")(head_model)
head_model = Dense(512, activation="relu")(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(2, activation="softmax")(head_model)

model = Model(inputs=base_pretrained_model.input, outputs=head_model)

for layer in base_pretrained_model.layers:
	layer.trainable = False

opt = SGD(lr=1e-4, momentum=0.9)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])

filepath="logs/weight-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

hist = model.fit_generator(
	train_generator,
	steps_per_epoch=total_train//BATCH_SIZE_BASE,
	validation_data=val_generator,
	validation_steps=total_val//BATCH_SIZE_BASE,
	epochs=EPOCHS_BASE,
	callbacks=callbacks_list
)


model_json = model.to_json()
with open("model/model.json", "w") as json_file:
    json_file.write(model_json)

with open('model/model_history.json', 'wb') as file_pi:
    pickle.dump(hist.history, file_pi)