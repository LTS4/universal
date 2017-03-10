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

A demo using Caffe is provided in `demo_caffe.m`.

First of all, you need to setup the paths to [DeepFool](http://github.com/lts4/deepfool) and [Caffe](http://caffe.berkeleyvision.org) (or [MatConvNet](http://www.vlfeat.org/matconvnet/)), e.g.:
```
addpath('./DeepFool');
addpath('./caffe/matlab');
```
Next, you have to load your pre-trained model. __Note that the last layer (usually softmax layer) should have been removed (see [DeepFool](http://github.com/lts4/deepfool) for more information).__ We provide in the `data` folder the prototxt files for the common networks, without the softmax layer. The pre-trained model can be loaded as follows, using Caffe:
```
caffe.set_mode_gpu(); % initialize Caffe in gpu mode
caffe.set_device(0); % choose the first gpu

net_model = 'deploy_googlenet.prototxt'; % GoogLeNet without softmax layer
net_weights = 'googlenet.caffemodel'; % weights
net = caffe.Net(net_model, net_weights, 'test'); % run with phase test
```
After loading your pre-trained model using your preferred framework, you have to load the set of images to compute the universal perturbation. __Be careful that the images should be pre-processed in the same way that the training images are pre-processed.__ We provide an example script in `makeImagenetData.m`, where 10,000 training images are used. Once you have your data, you can load it and compute a universal perturbation as follows:

```
% Load data
dataset = h5read(fullfile('data', 'ImageNet.h5'), ['/batch1']);
% Set parameters
opts.library = 'caffe' % or 'matconvnet';
opts.net = net;
% Compute universal perturbation
v = universal_perturbation(dataset, opts);
```

##Reference
[1] S. Moosavi-Dezfooli\*, A. Fawzi\*, O. Fawzi, P. Frossard:
[*Universal adversarial perturbations*](http://arxiv.org/pdf/1610.08401), CVPR 2017.
