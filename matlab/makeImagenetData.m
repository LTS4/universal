function im_array = makeImagenetData(imagenet_path)
    
    if (exist(fullfile('data', 'ImageNet.h5'), 'file'))
        im_array = h5read(fullfile('data', 'ImageNet.h5'), ['/batch',num2str(1)]);
        return;
    end
    
    d = load(fullfile('data', 'ilsvrc_2012_mean.mat'));
    mean_data = d.mean_data;
    
    num_class = 1000;
    num_images = 10000;
    num_per_class = num_images/num_class;

    allLabels = dir( [imagenet_path] );
    class_name = {allLabels(3:end).name};

    im_array = zeros(224,224,3,num_images,'single');

    for j=1:num_class
        fprintf('Processing data %d\n', j);
        ff = fullfile(imagenet_path, class_name{j});
        allFiles = dir( ff );
        fullfile(imagenet_path, class_name{j})
        allNames = { allFiles.name };
        for i=1:num_per_class
            im = imread(char(fullfile(ff, allNames(i+3))));
          if size(im,3)==1
            im = repmat(im,1,1,3);
          end
          im_array_par(:,:,:,i) = preprocess_img(im, mean_data);
        end
        im_array(:,:,:,(j-1)*num_per_class+(1:num_per_class)) = im_array_par;
    end
    
    % Save data
    % This can take a significant amount of space. Comment if needed.
    fprintf('Saving the data...\n');
    h5create(fullfile('data', 'ImageNet.h5'),['/batch',num2str(1)],[224 224 3 10000],'DataType','single');
    h5write(fullfile('data', 'ImageNet.h5'), ['/batch',num2str(1)], im_array);