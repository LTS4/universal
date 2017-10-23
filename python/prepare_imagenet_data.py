import numpy as np
import os
from scipy.misc import imread, imresize

CLASS_INDEX = None
CLASS_INDEX_PATH = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'


def preprocess_image_batch(image_paths, img_size=None, crop_size=None, color_mode="rgb", out=None):
    img_list = []

    for im_path in image_paths:
        img = imread(im_path, mode='RGB')
        if img_size:
            img = imresize(img,img_size)

        img = img.astype('float32')
        # We normalize the colors (in RGB space) with the empirical means on the training set
        img[:, :, 0] -= 123.68
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 103.939
        # We permute the colors to get them in the BGR order
        # if color_mode=="bgr":
        #    img[:,:,[0,1,2]] = img[:,:,[2,1,0]]

        if crop_size:
            img = img[(img_size[0] - crop_size[0]) // 2:(img_size[0] + crop_size[0]) // 2, (img_size[1]-crop_size[1])//2:(img_size[1]+crop_size[1])//2, :];

        img_list.append(img)

    try:
        img_batch = np.stack(img_list, axis=0)
    except:
        raise ValueError('when img_size and crop_size are None, images'
                ' in image_paths must have the same shapes.')

    if out is not None and hasattr(out, 'append'):
        out.append(img_batch)
    else:
        return img_batch

def undo_image_avg(img):
    img_copy = np.copy(img)
    img_copy[:, :, 0] = img_copy[:, :, 0] + 123.68
    img_copy[:, :, 1] = img_copy[:, :, 1] + 116.779
    img_copy[:, :, 2] = img_copy[:, :, 2] + 103.939
    return img_copy

def create_imagenet_npy(path_train_imagenet, len_batch=10000):

    # path_train_imagenet = '/datasets2/ILSVRC2012/train';

    sz_img = [224, 224]
    num_channels = 3
    num_classes = 1000

    im_array = np.zeros([len_batch] + sz_img + [num_channels], dtype=np.float32)
    num_imgs_per_batch = int(len_batch / num_classes)

    dirs = [x[0] for x in os.walk(path_train_imagenet)]
    dirs = dirs[1:]

    # Sort the directory in alphabetical order (same as synset_words.txt)
    dirs = sorted(dirs)

    it = 0
    Matrix = [0 for x in range(1000)]

    for d in dirs:
        for _, _, filename in os.walk(os.path.join(path_train_imagenet, d)):
            Matrix[it] = filename
        it = it+1


    it = 0
    # Load images, pre-process, and save
    for k in range(num_classes):
        for u in range(num_imgs_per_batch):
            print('Processing image number ', it)
            path_img = os.path.join(dirs[k], Matrix[k][u])
            image = preprocess_image_batch([path_img],img_size=(256,256), crop_size=(224,224), color_mode="rgb")
            im_array[it:(it+1), :, :, :] = image
            it = it + 1

    return im_array
