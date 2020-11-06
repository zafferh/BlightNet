from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg16
from sklearn.model_selection import train_test_split

import os
import shutil

def load_data(class0_path, class1_path):
        data = []
        label = []

        for f in os.listdir(class0_path):
            data.append(os.path.join(class0_path, f))
            label.append(0)

        for f in os.listdir(class1_path):
            data.append(os.path.join(class1_path, f))
            label.append(1)
	
        return data, label

def split_data_to_dir(data, label, base_train_dir, base_val_dir, base_test_dir):
        X_train, X_test, y_train, y_test = train_test_split(
            data, label, test_size=0.2, random_state=1)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.25, random_state=1)

        if os.path.isdir(base_train_dir):
            shutil.rmtree(base_train_dir)
        
        os.mkdir(base_train_dir)
        os.mkdir(os.path.join(base_train_dir, 'class0'))
        os.mkdir(os.path.join(base_train_dir, 'class1'))

        if os.path.isdir(base_val_dir):
            shutil.rmtree(base_val_dir)

        os.mkdir(base_val_dir)
        os.mkdir(os.path.join(base_val_dir, 'class0'))
        os.mkdir(os.path.join(base_val_dir, 'class1'))
            
        if os.path.isdir(base_test_dir):
            shutil.rmtree(base_test_dir)
        
        os.mkdir(base_test_dir)
        os.mkdir(os.path.join(base_test_dir, 'class0'))
        os.mkdir(os.path.join(base_test_dir, 'class1'))

        for i in range(len(X_train)):
            shutil.copyfile(X_train[i], 
                    os.path.join(base_train_dir,'class{}'.format(y_train[i]), 
                    os.path.basename(X_train[i])))
        for i in range(len(X_val)):
            shutil.copyfile(X_val[i],
                    os.path.join(base_val_dir,'class{}'.format(y_val[i]),
                    os.path.basename(X_val[i])))
        for i in range(len(X_test)):
            shutil.copyfile(X_test[i],
                    os.path.join(base_test_dir,'class{}'.format(y_test[i]),
                    os.path.basename(X_test[i])))
        print("Added {} train images, {} val images and {} test images.".format(len(X_train), len(X_val), len(X_test)))
		

if __name__ == '__main__':

    base_dir = 'data'
    blighted_path = os.path.join(base_dir, 'blighted')
    unblighted_path = os.path.join(base_dir, 'unblighted')

    base_train_dir = os.path.join(base_dir, 'train')
    base_val_dir = os.path.join(base_dir, 'val')
    base_test_dir = os.path.join(base_dir, 'test')

    data, label = load_data(unblighted_path, blighted_path)
    split_data_to_dir(data, label, base_train_dir, base_val_dir, base_test_dir)
