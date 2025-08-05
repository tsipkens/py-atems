
import numpy as np
import json as json
import cv2
from skimage import measure

from skimage.segmentation import clear_border


# SAM2 imports.
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2

# System and IO imports.
from tqdm import tqdm
import pickle
import warnings
import sys



def downscaler(image, max_side=512):

    original_h, original_w = image.shape[:2]

    # 1. Downsample the image (e.g., max side from input)
    scale = max_side / max(original_h, original_w)
    if scale < 1:
        small_img = cv2.resize(image, (int(original_w * scale), int(original_h * scale)), interpolation=cv2.INTER_LINEAR)
    else:
        small_img = image.copy()

    return small_img


def upscaler(mask, original_h, original_w):

    # Mask is boolean or binary np.ndarray (small_h, small_w)
    mask_uint8 = mask.astype(np.uint8) * 255
    up_mask = cv2.resize(mask_uint8, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    return up_mask.astype(bool)  # shape (N, original_h, original_w)


def segment(image):

    # Load image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # kernel = np.ones((5, 5), np.uint8)
    # image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    original_h, original_w = image.shape[:2]
    image_s = image # downscaler(image)

    # Inputs files for SAM2
    checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "./configs/sam2.1/sam2.1_hiera_l.yaml"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        sam2 = build_sam2(model_cfg, checkpoint, device='cuda', apply_postprocessing=False)

        # Create mask generator
        # Mask generator with fast settings
        mask_generator = SAM2AutomaticMaskGenerator(
            sam2,
            points_per_side=4,  # 4
            pred_iou_thresh=0.9,  # 0.9
            stability_score_thresh=0.85,  # 0.85
            min_mask_region_area=1000,  # 1000
        )

        # Generate masks
        masks = mask_generator.generate(image_s)

    # Create a foreground mask: union of top-N masks or masks above area threshold
    H, W = image_s.shape[:2]
    foreground = np.zeros((H, W), dtype=bool)

    # Combine all of the masks.
    for m in masks:
        foreground = np.logical_or(foreground, m["segmentation"])

    # Convert to binary class map (0: background, 1: foreground)
    binary_mask = np.zeros((H, W), dtype=np.uint8)
    binary_mask[foreground] = 1

    # binary_mask = upscaler(binary_mask, original_h, original_w)


    # PHASE 2: Initialize at foreground centriods and redo.
    # Label each connected object
    labels = measure.label(binary_mask, connectivity=1)

    # Compute region properties
    regions = measure.regionprops(labels)

    # Get centroid of each object
    centroids = np.array([list(map(int, region.centroid)) for region in regions])

    # Initialize predictor
    predictor = SAM2ImagePredictor(sam2)
    predictor.set_image(image)

    # Example point input (x, y), label = 1 for foreground
    input_points = centroids
    input_label = np.ones(len(centroids))

    binary_mask, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_label,
        multimask_output=True
    )

    best_idx = scores.argmax()
    binary_mask = binary_mask[best_idx]

    return binary_mask


# Unpickle the input list of arrays from stdin.
images = pickle.load(sys.stdin.buffer)

binaries = [[]] * len(images)

# Loop over images.
for ii in tqdm(range(len(images)), file=sys.stderr):
    img = images[ii]
    binaries[ii] = np.array(segment(img))

print(f"Completed {ii+1}/{len(images)}", file=sys.stderr)

pickle.dump(binaries, sys.stdout.buffer)

