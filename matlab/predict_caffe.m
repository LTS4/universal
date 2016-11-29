function [l,out] = predict_caffe(im,net,BATCH_SIZE)
% Computes the labels of im predicted by net

blob_shape = net.blobs('data').shape(); % store the original input size
n = size(im,4);

change_input_blob(net,BATCH_SIZE); % rehsape the input batch size

K = floor(n/BATCH_SIZE);
R = rem(n,BATCH_SIZE);

for k=1:K
    res = net.forward({im(:,:,:,(k-1)*BATCH_SIZE+(1:BATCH_SIZE))});
    res2(:,((k-1)*BATCH_SIZE+1):k*BATCH_SIZE)=res{1};
end

if(R~=0)
    change_input_blob(net,R);
    res = net.forward({im(:,:,:,K*BATCH_SIZE+(1:R))});
    res2(:,(K*BATCH_SIZE+1):K*BATCH_SIZE+R)=res{1};
end
[~,l] = max(res2,[],1);
out = res2;

net.blobs('data').reshape(blob_shape); % reshape to the original input size
net.reshape();
end
