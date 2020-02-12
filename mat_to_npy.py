import os,cv2
import scipy.io as scio
import numpy as np
from tqdm import tqdm
# from scipy.misc import imresize
 
standard_size = [720,1280];
val_mat_dir='/input0/train_mat'
image_dir='/input0/image/'
save_label_path='train_label'
os.mkdir(save_label_path)
def main():
    img_list=os.listdir(val_mat_dir)
 
    for idx in tqdm(range(len(img_list))):
        filename=img_list[idx].replace('.mat','')
        if not filename:
            continue
 
        i = idx
    #     if (mod(idx,10)==0)
    #         fprintf(1,'Processing %3d/%d files\n', idx, num_images);
    #     end
 
        mat=scio.loadmat(os.path.join(val_mat_dir,filename+'.mat'))
        input_img_name=os.path.join(image_dir,filename+'.jpg')
        im = cv2.imread(input_img_name,cv2.IMREAD_GRAYSCALE)
        h,w=im.shape
        annPoints =  mat['image_info']
 
        rate = standard_size[0]/h;
        rate_w = w*rate;
        if rate_w>standard_size[1]:
            rate = standard_size[1]/w;
        rate_h = float(int(h*rate))/h;
        rate_w = float(int(w*rate))/w;
        im = cv2.resize(im,(int(w*rate),int(h*rate)));
#         print(annPoints[0, 0][0])
        annPoints=annPoints[0,0][0]
        annPoints[:,0] = annPoints[:,0]*float(rate_w)
        annPoints[:,1] = annPoints[:,1]*float(rate_h)
 
        im_density=get_density_map_gaussian(im,annPoints)
#         print(im_density.shape,im_density,type(im_density))
#         np.savetxt("%s/%s.csv"%(save_label_path,filename), im_density)
        np.save('%s/%s.npy'%(save_label_path,filename),im_density)
 
 
import numpy as np
import math
 
 
def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
 
 
def get_density_map_gaussian(im, points):
    im_density = np.zeros(im.shape)
    [h, w] = im_density.shape
 
    for j in range(0, len(points)):
        f_sz = 15
        sigma = 4.0
        # H = matlab.fspecial('Gaussian', [f_sz, f_sz], sigma)
        H = matlab_style_gauss2D([f_sz, f_sz], sigma)
        x = min(w, max(1, abs(int(math.floor(points[j, 0])))))
        y = min(h, max(1, abs(int(math.floor(points[j, 1])))))
 
        if x > w or y > h:
            continue
        x1 = x - int(np.floor(f_sz / 2))
        y1 = y - int(np.floor(f_sz / 2))
        x2 = x + int(np.floor(f_sz / 2))
        y2 = y + int(np.floor(f_sz / 2))
        dfx1 = 0
        dfy1 = 0
        dfx2 = 0
        dfy2 = 0
        change_H = False
        if x1 < 1:
            dfx1 = abs(x1) + 1
            x1 = 1
            change_H = True
        if y1 < 1:
            dfy1 = abs(y1) + 1
            y1 = 1
            change_H = True
        if x2 > w:
            dfx2 = x2 - w
            x2 = w
            change_H = True
        if y2 > h:
            dfy2 = y2 - h
            y2 = h
            change_H = True
        x1h = 1 + dfx1
        y1h = 1 + dfy1
        x2h = f_sz - dfx2
        y2h = f_sz - dfy2
        if change_H:
            # H = matlab.fspecial('Gaussian', [double(y2h - y1h + 1), double(x2h - x1h + 1)], sigma)
            H = matlab_style_gauss2D([float(y2h - y1h + 1), float(x2h - x1h + 1)], sigma)
        im_density[y1-1: y2, x1-1: x2] = im_density[y1-1: y2, x1-1: x2] + H
    return im_density
 
if __name__=='__main__':
    main()