
import numpy as np

import matplotlib.pyplot as plt

# Import custom modules
import tools
import agg, pp.pp as pp

# Load images from the 'images' directory
imgs, pixsizes, fns = tools.load_imgs('images', detect=True)  # OPTION 3: load all images in 'images' folder

# Run K-MEANS for all images
imgs_binary, _, _ = agg.seg_kmeans(imgs, pixsizes, v='v6.1')

Aggs = agg.analyze_binary(imgs_binary, pixsizes, imgs, None)

Aggs = pp.pcm(Aggs, imgs_binary)

plt.figure(figsize=(15, 10))
tools.imshow_agg(Aggs, imgs, imgs_binary, f_scale=True)
plt.show()
