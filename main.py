
import numpy as np

# Import custom modules
import tools
import agg, pp

# Load images from the 'images' directory
imgs, pixsizes = tools.load_imgs('images')  # OPTION 3: load all images in 'images' folder

# Run K-MEANS for all images
imgs_binary, img_kmeans, feature_set = agg.seg_kmeans(imgs, pixsizes)

tools.imshow_binary2(imgs, imgs_binary)

Aggs = agg.analyze_binary(imgs_binary, pixsizes, imgs, None, 1)

Aggs = pp.pcm(Aggs)
