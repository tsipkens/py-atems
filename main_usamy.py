"""Run aggregate and primary-particle YGMAP together."""

from pathlib import Path

import agg
import config_usamy as config
import pp
import tools


# Change this path to select the images for this workflow.
# IMAGE_FOLDER = Path(__file__).resolve().parent / "images2"
# imgs, pixsizes, fns = tools.load_microscopy_images(IMAGE_FOLDER)
imgs, pixsizes, fns = tools.load_imgs('images', detect=True)

imgs_binary, imgs_detect = agg.seg_usamy(imgs, pixsizes, yolo_opts=config.YOLO, device=config.DEVICE, return_detections=True, **config.YGMAP,)

aggs = agg.Aggs.Aggs(imgs_binary, pixsizes,imgs)
aggs.imshow1(24)
plt.show()

imgs_binary_pp, particles, imgs_detect_pp = pp.seg_usamy(imgs, pixsizes, aggregate_masks=imgs_binary, yolo_opts=config.PP_YOLO, device=config.DEVICE, return_particles=True, return_detections=True, **config.YGMAP_PP,)

if config.SHOW_RESULTS:
    tools.imshow_usamy_combined(imgs, imgs_binary, particles, pixsizes=pixsizes)
