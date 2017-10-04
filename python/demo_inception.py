import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
import os.path
from prepare_imagenet_data import preprocess_image_batch, create_imagenet_npy, undo_image_avg
import matplotlib.pyplot as plt
import sys, getopt
import zipfile
from timeit import time

if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve


from universal_pert import universal_perturbation
device = '/gpu:0'
num_classes = 10

def jacobian(y_flat, x, inds):
    n = num_classes # Not really necessary, just a quick fix.
    loop_vars = [
         tf.constant(0, tf.int32),
         tf.TensorArray(tf.float32, size=n),
    ]
    _, jacobian = tf.while_loop(
        lambda j,_: j < n,
        lambda j,result: (j+1, result.write(j, tf.gradients(y_flat[inds[j]], x))),
        loop_vars)
    return jacobian.stack()

if __name__ == '__main__':

    # Parse arguments
    argv = sys.argv[1:]

    # Default values
    path_train_imagenet = '/datasets2/ILSVRC2012/train'
    path_test_image = 'data/test_img.png'
    
    try:
        opts, args = getopt.getopt(argv,"i:t:",["test_image=","training_path="])
    except getopt.GetoptError:
        print ('python ' + sys.argv[0] + ' -i <test image> -t <imagenet training path>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-t':
            path_train_imagenet = arg
        if opt == '-i':
            path_test_image = arg

    with tf.device(device):
        persisted_sess = tf.Session()
        inception_model_path = os.path.join('data', 'tensorflow_inception_graph.pb')

        if os.path.isfile(inception_model_path) == 0:
            print('Downloading Inception model...')
            urlretrieve ("https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip", os.path.join('data', 'inception5h.zip'))
            # Unzipping the file
            zip_ref = zipfile.ZipFile(os.path.join('data', 'inception5h.zip'), 'r')
            zip_ref.extract('tensorflow_inception_graph.pb', 'data')
            zip_ref.close()

        model = os.path.join(inception_model_path)

        # Load the Inception model
        with gfile.FastGFile(model, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            persisted_sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')

        persisted_sess.graph.get_operations()

        persisted_input = persisted_sess.graph.get_tensor_by_name("input:0")
        persisted_output = persisted_sess.graph.get_tensor_by_name("softmax2_pre_activation:0")

        print('>> Computing feedforward function...')
        def f(image_inp): return persisted_sess.run(persisted_output, feed_dict={persisted_input: np.reshape(image_inp, (-1, 224, 224, 3))})

        file_perturbation = os.path.join('data', 'universal.npy')

        if os.path.isfile(file_perturbation) == 0:

            # TODO: Optimize this construction part!
            print('>> Compiling the gradient tensorflow functions. This might take some time...')
            y_flat = tf.reshape(persisted_output, (-1,))
            inds = tf.placeholder(tf.int32, shape=(num_classes,))
            dydx = jacobian(y_flat,persisted_input,inds)

            print('>> Computing gradient function...')
            def grad_fs(image_inp, indices): return persisted_sess.run(dydx, feed_dict={persisted_input: image_inp, inds: indices}).squeeze(axis=1)

            # Load/Create data
            datafile = os.path.join('data', 'imagenet_data.npy')
            if os.path.isfile(datafile) == 0:
                print('>> Creating pre-processed imagenet data...')
                X = create_imagenet_npy(path_train_imagenet)

                print('>> Saving the pre-processed imagenet data')
                if not os.path.exists('data'):
                    os.makedirs('data')

                # Save the pre-processed images
                # Caution: This can take take a lot of space. Comment this part to discard saving.
                np.save(os.path.join('data', 'imagenet_data.npy'), X)

            else:
                print('>> Pre-processed imagenet data detected')
                X = np.load(datafile)

            # Running universal perturbation
            v = universal_perturbation(X, f, grad_fs, delta=0.2,num_classes=num_classes)

            # Saving the universal perturbation
            np.save(os.path.join(file_perturbation), v)

        else:
            print('>> Found a pre-computed universal perturbation! Retrieving it from ", file_perturbation')
            v = np.load(file_perturbation)

        print('>> Testing the universal perturbation on an image')

        # Test the perturbation on the image
        labels = open(os.path.join('data', 'labels.txt'), 'r').read().split('\n')

        image_original = preprocess_image_batch([path_test_image], img_size=(256, 256), crop_size=(224, 224), color_mode="rgb")
        label_original = np.argmax(f(image_original), axis=1).flatten()
        str_label_original = labels[np.int(label_original)-1].split(',')[0]

        # Clip the perturbation to make sure images fit in uint8
        clipped_v = np.clip(undo_image_avg(image_original[0,:,:,:]+v[0,:,:,:]), 0, 255) - np.clip(undo_image_avg(image_original[0,:,:,:]), 0, 255)

        image_perturbed = image_original + clipped_v[None, :, :, :]
        label_perturbed = np.argmax(f(image_perturbed), axis=1).flatten()
        str_label_perturbed = labels[np.int(label_perturbed)-1].split(',')[0]

        # Show original and perturbed image
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(undo_image_avg(image_original[0, :, :, :]).astype(dtype='uint8'), interpolation=None)
        plt.title(str_label_original)

        plt.subplot(1, 2, 2)
        plt.imshow(undo_image_avg(image_perturbed[0, :, :, :]).astype(dtype='uint8'), interpolation=None)
        plt.title(str_label_perturbed)

        plt.show()
