from keras.layers import Cropping2D, Input, Lambda, Flatten, Dense, Dropout, Average
from keras.models import Sequential, Model
from keras.applications import vgg16

import tensorflow as tf

def Resize(shape):
    return Lambda(lambda image: tf.image.resize(image, shape))

def StandardVGG(x, im_size, name='vgg'):
    vgg = vgg16.VGG16(input_tensor=Input(shape=(im_size, im_size, 3)),
                        include_top=False, weights='imagenet')
    vgg.name = name
    for layer in vgg.layers:
        layer.trainable = False
    x = vgg(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    return Dense(2, activation="softmax")(x)

class CroppedVGG:

    def __init__(self, im_size, cropping):

        input = Input(shape=(im_size, im_size, 3))
        x = Cropping2D(cropping=cropping)(input)
        x = Resize((im_size, im_size))(x)
        x = StandardVGG(x, im_size)
        self.model = Model(inputs=[input], outputs=[x])