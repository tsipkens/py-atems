"""Run aggregate and primary-particle YGMAP together."""

from pathlib import Path

import agg
import config_ygmap as config
import pp
import tools


# Change this path to select the images for this workflow.
IMAGE_FOLDER = Path(__file__).resolve().parent / "images2"
imgs, pixsizes, fns = tools.load_microscopy_images(IMAGE_FOLDER)

imgs_binary, imgs_detect = agg.seg_ygmap(imgs, pixsizes, yolo_opts=config.YOLO, device=config.DEVICE, return_detections=True, **config.YGMAP,)

imgs_binary_pp, particles, imgs_detect_pp = pp.seg_ygmap_pp(imgs, pixsizes, aggregate_masks=imgs_binary, yolo_opts=config.PP_YOLO, device=config.DEVICE, return_particles=True, return_detections=True, **config.YGMAP_PP,)


if config.SHOW_RESULTS:
    tools.imshow_ygmap_combined(imgs, imgs_binary, particles, pixsizes=pixsizes)
