import scipy
from tensorlayer.prepro import *

def get_imgs_fn(file_name, path):
    return scipy.misc.imread(path + file_name, mode='RGB')

def crop_sub_img_fn_96(x, is_random=True):
    x = crop(x , wrg=96, hrg=96, is_random=is_random)
    x = x / (255. / 2.)
    x = x - 1
    return x
'''
def downsample_fn(x):
    x = imresize(x , size=[96, 96], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1
    return x
'''

