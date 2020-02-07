clc; clear all;

standard_size = [720,1280];
val_mat_dir='./val_mat.output/'
image_dir='image/image/'
save_label_path='val_label/'

img_list = dir(fullfile(val_mat_dir));

num_images=size(img_list,1);
for idx = 3:num_images
    filename=img_list(idx).name(1:end-4)
    i = idx;
    if (mod(idx,10)==0)
        fprintf(1,'Processing %3d/%d files\n', idx, num_images);
    end

    load(strcat(val_mat_dir,filename,'.mat'))
    input_img_name = strcat(image_dir,filename,'.jpg');
    im = imread(input_img_name);
    [h, w, c] = size(im);
    annPoints =  image_info.location;


    rate = standard_size(1)/h;
    rate_w = w*rate;
    if rate_w>standard_size(2)
        rate = standard_size(2)/w;
    end
    rate_h = double(int16(h*rate))/h;
    rate_w = double(int16(w*rate))/w;
    im = imresize(im,[int16(h*rate),int16(w*rate)]);
    annPoints(:,1) = annPoints(:,1)*double(rate_w);
    annPoints(:,2) = annPoints(:,2)*double(rate_h);

    im_density = get_density_map_gaussian(im,annPoints,15,4);
    im_density = im_density(:,:,1);

%     imwrite(im, [ 'val_img/' filename '.jpg']);
    csvwrite([ save_label_path, filename '.csv'], im_density);

end