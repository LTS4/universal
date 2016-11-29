function [ys, scores] = predict_matconvnet(ims, net, batchsize)

ys = zeros(size(ims, 4), 1);
scores = zeros(1000, size(ims, 4)); % Number of classes
for b = 1:batchsize:size(ims, 4)
    res = vl_simplenn(net, gpuArray(ims(:, :, :, b:(b+batchsize-1))), [], [], 'Mode', 'test');
    
    cc = squeeze(gather(res(end).x));
    scores(:, b:(b+batchsize-1)) = cc;
    [~, ys(b:(b+batchsize-1))] = max(squeeze(gather(res(end).x)));
end

end