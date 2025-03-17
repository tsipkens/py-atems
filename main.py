
import numpy as np

import matplotlib.pyplot as plt

# Import custom modules
import tools
import agg, pp.pp as pp

# Load images from the 'images' directory
imgs, pixsizes, fns, Imgs = tools.load_imgs('images')  # OPTION 3: load all images in 'images' folder

# Run K-MEANS for all images
imgs_binary, _, _ = agg.seg_kmeans(imgs, pixsizes)

Aggs = agg.analyze_binary(imgs_binary, pixsizes, imgs, None)

Aggs = pp.pcm(Aggs, imgs_binary)

tools.imshow_agg(Aggs, imgs, imgs_binary)
plt.show()
