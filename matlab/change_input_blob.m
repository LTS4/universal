function change_input_blob(net,n)
blob_shape = net.blobs('data').shape();
blob_shape(4) = n;
net.blobs('data').reshape(blob_shape);

net.reshape();
end
