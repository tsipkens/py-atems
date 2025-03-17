
import numpy as np

from PIL import Image

# Import custom modules
import tools
import agg, pp.pp as pp

# Load images from the 'images' directory
imgs, pixsizes, fns, _ = tools.load_imgs('images')  # OPTION 3: load all images in 'images' folder

imgs_binary = agg.carboseg.seg_cnn(imgs, pixsizes)

tools.imshow_binary2(imgs, imgs_binary)
