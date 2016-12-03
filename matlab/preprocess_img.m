function imgs_out = preprocess_img(imgs, mean_data)

    IMAGE_DIM = 256;

    if size(imgs,3)==1
        imgs = repmat(imgs,1,1,3);
    end
    im_ = imgs(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
    im_ = permute(im_, [2, 1, 3]);  % flip width and height
    im_ = single(im_);  % convert from uint8 to single
    im_ = imresize(im_, [IMAGE_DIM IMAGE_DIM], 'bilinear');  % resize im_data
    im_ = im_ - mean_data;  % subtract mean_data (already in W x H x C, BGR)
    imgs_out = im_(1:224,1:224,:);
    
end