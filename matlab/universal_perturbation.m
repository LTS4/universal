%%
%   MATLAB code for Universal Perturbation
%
%   universal_perturbation(dataset, varargin):
%   computes a universal perturbation for a Caffe's or MatConvNet's model
%
%   INPUTS 
%   dataset: images in W*H*C*N format  
%   opts: A struct containing the following parameters:
%     delta: controls the desired fooling rate (default = 80% fooling rate)
%     MAX_ITER_UNIV: optional other termination criterion (maximum number of iteration, default = Inf)
%     xi: controls the l_p magnitude of the perturbation (default = 10)
%     p: norm to be used (FOR NOW, ONLY p = 2, and p = Inf ARE ACCEPTED!) (default = Inf)
%     df.labels_limit: DeepFool's labels_limit (limits the number of classes to test against, by default = 10)
%     df.overshoot: DeepFool's overshoot (used as a termination criterion to prevent vanishing updates, by default = 0.02)
%     df.MAX_ITER: DeepFool's MAX_ITER (maximum number of iterations for DeepFool , by default = 10)
%
%   OUTPUT
%   v: universal perturbation
%
%   S. Moosavi-Dezfooli, A. Fawzi, O.Fawzi, P. Frossard: Universal Adversarial Perturbations, arXiv:1610.08401.
%%
function v = universal_perturbation(dataset, varargin)

%Options
opts.delta=0.2;
opts.MAX_ITER_UNIV = 10;
opts.xi = 10;
opts.p = Inf;
opts.BATCH_SIZE = 100;
opts.df.labels_limit=10;
opts.df.overshoot=0.02;
opts.df.MAX_ITER=10;
opts.library = [];
opts.net = [];

opts = vl_argparse(opts, varargin);

v = 0;
fooling_rate = 0;
num_images = size(dataset,4);

if isempty(opts.net) % check the network
    fprintf('Please specify the network!\nopts.net = net');
    return;
end

if(strcmp(opts.library,'caffe')) % check the framework: only Caffe and MatConvNet are supported.
    adversarial_DF = @(x) adversarial_DeepFool_caffe(x,opts.net,opts.df);
    predict = @(x)predict_caffe(x,opts.net,opts.BATCH_SIZE);
elseif(strcmp(opts.library,'matconvnet'))
    adversarial_DF = @(x) adversarial_DeepFool_matconvnet(x,opts.net,opts.df);
    predict = @(x)predict_matconvnet(x,opts.net,opts.BATCH_SIZE);
else
    fprintf('Library is not supported!\n');
    return;
end

itr = 0;
while(fooling_rate<(1-opts.delta) && itr<opts.MAX_ITER_UNIV)
    itr = itr + 1;    
    dataset = dataset(:,:,:,randperm(num_images)); % shuffle the dataset
    disp('Data randomly permuted');
    for cnt = 1:num_images % go through the data set and compute the perturbation increments sequentially
        if (predict(dataset(:,:,:,cnt))==predict(dataset(:,:,:,cnt)+v))
            [dv,~,~,iter] = adversarial_DF(dataset(:,:,:,cnt)+v); % compute adversarial perturbation
            if(iter<opts.df.MAX_ITER-1) % check for convergence
                v = v + dv; % update v
                v = proj_lp(v, opts.xi, opts.p); % project v on l_p ball
            end
        end
        clc;
        fprintf('Fooling rate for pass %d = %.3f\n', itr-1, fooling_rate);
        fprintf('Iteration %d/%d\n', cnt, num_images);
        
    end
    fprintf('Computing the new fooling rate...\n');
    est_labels_orig = predict(dataset(:,:,:,1:num_images)); % compute the original labels
    est_labels_pert = predict(bsxfun(@plus,dataset(:,:,:,1:num_images),v)); % compute the perturbed labels
    fooling_rate = sum(est_labels_orig~=est_labels_pert)/num_images; % compute the fooling rate
    fprintf('New fooling rate = %f\n', fooling_rate);
end

function v_projected = proj_lp(v, xi, p)
% project on the l_p ball centered at 0 and of radius xi
% SUPPORTS only p = 2 and p = Inf for now

if p==Inf
    v_projected = sign(v).*min(abs(v),xi);
elseif p==2
    v_projected = v * min(1, xi/norm(v(:)));
end
