#MATLAB

###universal_perturbation.m

This function implements the algorithm proposed in [[1]](http://arxiv.org/pdf/1610.08401) to find a universal perturbation.

###Inputs
- `dataset`: images in `W*H*C*N` format, where `W`:width, `H`:height, `C`:channels (RGB), and `N`:number of images.
- `opts`: A structure containing the following optionals and non-optionals parameters:
  - `library`: framework to be used: Caffe or MatConvNet. __(Obligatory)__
  - `net`: corresponding network structure. __(Obligatory)__
  - `delta`: controls the desired fooling rate (default = 80% fooling rate).
  - `MAX_ITER_UNIV`: termination criterion (maximum number of iterations, default = Inf).
  - `xi`: controls the l_p magnitude of the perturbation (default = 10).
  - `p`: norm to be used (FOR NOW, ONLY p = 2, and p = Inf ARE ACCEPTED!) (default = Inf).
  - `df`: DeepFool's parameters (see [DeepFool](http://github.com/lts4/deepfool) for more information).
  
###Output
- `v`: a `W*H*C` perturbation image.

##Usage
First of all, you need to setup the paths to [DeepFool](http://github.com/lts4/deepfool) and [Caffe](http://caffe.berkeleyvision.org) (or [MatConvNet](http://www.vlfeat.org/matconvnet/)), e.g.:
```
addpath('./DeepFool');
addpath('./caffe/matlab');
```
Now, you have to load your pre-trained model. __Note that the last layer (usually softmax layer) should have been removed (see [DeepFool](http://github.com/lts4/deepfool) for more information).__ For example, in [Caffe](http://caffe.berkeleyvision.org) you can do something like:
```
caffe.set_mode_gpu(); % initialize Caffe in gpu mode
caffe.set_device(0); % choose the first gpu

net_model = 'deploy_googlenet.prototxt'; % GoogLeNet without softmax layer
net_weights = 'googlenet.caffemodel'; % weights
net = caffe.Net(net_model, net_weights, 'test'); % run with phase test (so that dropout isn't applied)
```
After loading your pre-trained model using your preferred framework, you have to load the set of images to compute the universal perturbation. __Be careful that the images should be pre-processed in the same way that the training images are pre-processed.__ As an example, one can load the first 500 ImageNet's training images from an HDF5 file:
```
dataset = h5read('/datasets/ImageNet_training.h5',['/batch',num2str(1)]);
dataset = dataset(:,:,:,1:500);
```
Finally, you have to set the options of the algorithm and run it:
```
opts.library = 'caffe';
opts.net = net;
v = universal_perturbation(dataset, opts);
```

##Reference
[1] S. Moosavi-Dezfooli\*, A. Fawzi\*, O. Fawzi, P. Frossard:
*Universal adversarial perturbations*.  [CoRR abs/1610.08401](http://arxiv.org/pdf/1610.08401) (2016)
