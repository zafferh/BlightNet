from imutils import paths

import cv2
import os
import shutil
import matplotlib.pyplot as plt

base_dir = 'data'

annot_dir = os.path.join(base_dir, 'annotation')
data_dir = os.path.join(base_dir, 'raw_data')
blighted_dir = os.path.join(base_dir, 'blighted')
unblighted_dir = os.path.join(base_dir, 'unblighted')

if os.path.isdir(unblighted_dir):
    shutil.rmtree(unblighted_dir)

if os.path.isdir(blighted_dir):
    shutil.rmtree(blighted_dir)

os.mkdir(blighted_dir)
os.mkdir(unblighted_dir)

class_file = os.path.join(annot_dir, 'classes.txt')

with open(class_file, 'r') as f:
    classes = f.readlines()

classes = [x.strip() for x in classes]

blighted_count=0
unblighted_count=0

for file in list(paths.list_images(data_dir)):
    basename = os.path.basename(file)
    name = os.path.splitext(basename)[0]

    annot_file = os.path.join(annot_dir, name+'.txt')

    if os.path.isfile(annot_file):
        img = cv2.imread(file)
        height, width, _ = img.shape
        with open(annot_file, 'r') as f:
            boxes = f.readlines()
        boxes = [x.split(' ') for x in boxes]
        for i, b in enumerate(boxes):
            center_x = round(float(b[1])*width)
            center_y = round(float(b[2])*height)
            box_width = round(float(b[3])*width)
            box_height = round(float(b[4])*height)
            crop_img = img[(center_y - box_height//2):(center_y+box_height//2),
                       (center_x - box_width//2):(center_x+box_width//2), :]
            if int(b[0]) == 0:
                unblighted_count+=1
                cv2.imwrite(os.path.join(unblighted_dir, '{}_{}.png'.format(i, name)), crop_img)
            elif int(b[0])==1:
                blighted_count+=1
                cv2.imwrite(os.path.join(blighted_dir, '{}_{}.png'.format(i, name)), crop_img)
    else:
        print('Could not find annotation file for {}.'.format(basename))

print('Added a total of {} images: {} blighted and {} unblighted'.format(blighted_count+unblighted_count, blighted_count, unblighted_count))

