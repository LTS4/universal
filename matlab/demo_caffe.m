clear; clc; close all;

DEVICE_ID = 0; %set gpu id starting from 0

% set paths to Caffe and DeepFool
PATH_CAFFE = '/path/to/caffe/matlab';
PATH_DEEPFOOL = '/path/to/DeepFool';
PATH_IMAGENET_TRAIN = '/path/to/ILSVRC2012/train';

addpath(PATH_CAFFE);
addpath(PATH_DEEPFOOL);

caffe.set_mode_gpu();
caffe.set_device(DEVICE_ID);

model = 'caffenet';

if (strcmp(model, 'vgg_16'))
    fprintf('Loading VGG 16\n');
    net_model_path = fullfile('data', 'deploy_vgg_16.prototxt');
    net_weights_path = fullfile('data', 'vgg_16.caffemodel');
    net_url = 'http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel';
elseif (strcmp(model, 'vgg_19'))
    fprintf('Loading VGG 19\n');
    net_model_path = fullfile('data', 'deploy_vgg_19.prototxt');
    net_weights_path = fullfile('data', 'vgg_19.caffemodel');
    net_url = 'http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel';
elseif (strcmp(model, 'googlenet'))
    fprintf('Loading GoogLeNet\n');
    net_model_path = fullfile('data', 'deploy_googlenet.prototxt');
    net_weights_path = fullfile('data', 'googlenet.caffemodel');
    net_url = 'http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel';
elseif (strcmp(model, 'vgg_f'))
    fprintf('Loading VGG-F\n');
    net_model_path = fullfile('data', 'deploy_vgg_f.prototxt');
    net_weights_path = fullfile('data', 'googlenet.caffemodel');
    net_url = 'http://dl.caffe.berkeleyvision.org/bvlc_vgg_f.caffemodel';
elseif (strcmp(model, 'caffenet'))
    fprintf('Loading CaffeNet\n');
    net_model_path = fullfile('data', 'deploy_caffenet.prototxt');
    net_weights_path = fullfile('data', 'caffenet.caffemodel');
    net_url = 'http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel';
elseif (strcmp(model, 'resnet-152'))
    fprintf('Loading ResNet-152\n');
    net_model_path = fullfile('data', 'deploy_resnet.prototxt');
    net_weights_path = fullfile('data', 'caffenet.caffemodel');
    net_url = 'https://deepdetect.com/models/resnet/ResNet-152-model.caffemodel';
else
    error('Model is not recognized!');
end

if (~exist(net_weights_path, 'file'))
    fprintf('Downloading model...\n');
    websave(net_weights_path, net_url);
end
    

net = caffe.Net(net_model_path, net_weights_path, 'test'); % run with phase test (so that dropout isn't applied)
fprintf('Network is loaded\n');

% Loading the data
fprintf('Loading the data...\n');
im_array = makeImagenetData(PATH_IMAGENET_TRAIN);
fprintf('Data is loaded\n');

% set options
opts.library = 'caffe';
opts.net = net;

v = universal_perturbation(im_array, opts);

% Test perturbation on a sample image

d = load(fullfile('data', 'ilsvrc_2012_mean.mat'));
mean_data = d.mean_data;

labels_file = fullfile('data', 'synset_words.txt');
fileID = fopen(labels_file);
C = textscan(fileID, '%c %d %s', 'Delimiter', '\n');
synset_names = C{3};
synset_names_short = synset_names;
for i = 1:1000
    synset_names_short(i) = strtok(synset_names(i), ',');
end

im_test = imread(fullfile('data', 'test_img.png'));
im_test = preprocess_img(im_test, mean_data);

original_label = predict_caffe(im_test, net, 1);
perturbed_label = predict_caffe(im_test+v, net, 1);

% Show original and perturbed images
im_ = im_test + mean_data(1:224, 1:224, :);
im_ = permute(im_,[2,1,3]);
im_ = im_(:,:,[3,2,1],:);

figure; imshow(uint8(im_));
xlabel(synset_names_short(original_label));
set(gca,'xtick',[],'ytick',[])
set(gcf, 'Color', 'white');
set(gca, 'FontSize', 15);
title('Original image');

v_ = permute(v,[2,1,3]);
v_ = v_(:,:,[3,2,1],:);
figure; imshow(uint8(im_ + v_));
xlabel(synset_names_short(perturbed_label));
set(gca,'xtick',[],'ytick',[])
set(gcf, 'Color', 'white');
set(gca, 'FontSize', 15);
title('Perturbed image');
