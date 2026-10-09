"""YGMAP-pp: separate primary-particle masks and area-equivalent diameters.

  AUTHOR: Ethan Xiong
"""
from dataclasses import dataclass
from pathlib import Path
import os
import cv2
import numpy as np
from agg.microsam import (
    MicroSAM, _rgb, _detections, square_window, local_box, pad_box, box_area,
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = os.getenv("MODEL_DIR", BASE_DIR / "models")
DEFAULT_CHECKPOINT = MODEL_DIR / "MicroSAM-pp" / "PyTorch" / "MSAM_PP_V1.pt"

@dataclass(frozen=True)
class PPOptions:
    window_size: int = 128
    box_padding_fraction: float = 0.25
    box_padding_pixels: int = 4
    min_mask_area: int = 10
    max_mask_to_box_ratio: float = 6.0
    min_neighbors: int = 1
    neighbor_padding_fraction: float = 0.35
    neighbor_padding_pixels: int = 6


def _options(opts):
    opts = PPOptions(**(opts or {})) if isinstance(opts, (dict, type(None))) else opts
    if not isinstance(opts, PPOptions):
        raise TypeError('opts must be a dict or PPOptions.')
    for name in ('window_size', 'box_padding_pixels', 'min_mask_area',
                 'min_neighbors', 'neighbor_padding_pixels'):
        value = getattr(opts, name)
        if not isinstance(value, int) or value < (1 if name == 'window_size' else 0):
            raise ValueError(f'Invalid {name}: {value}')
    for name in ('box_padding_fraction', 'max_mask_to_box_ratio', 'neighbor_padding_fraction'):
        value = getattr(opts, name)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and nonnegative.')
    return opts


def _prompted_component(mask, prompt):
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if count <= 2:
        return mask.astype(bool)
    x1, y1, x2, y2 = np.rint(prompt).astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(mask.shape[1], x2), min(mask.shape[0], y2)
    overlaps = np.bincount(labels[y1:y2, x1:x2].ravel(), minlength=count)[1:]
    if overlaps.max(initial=0) > 0:
        selected = int(np.argmax(overlaps)) + 1
    else:
        selected = int(np.argmax(np.bincount(labels.ravel())[1:])) + 1
    return labels == selected


def _neighbor_counts(boxes, shape, opts):
    counts = np.zeros(len(boxes), dtype=int)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = pad_box(box, shape, opts.neighbor_padding_fraction, opts.neighbor_padding_pixels)
        touches = ((x1 <= boxes[:, 2]) & (x2 >= boxes[:, 0]) &
                   (y1 <= boxes[:, 3]) & (y2 >= boxes[:, 1]))
        touches[i] = False
        counts[i] = touches.sum()
    return counts


def segment(imgs, imgs_detect_pp, pixsizes=None, *, opts=None,
            segmenter=None, checkpoint=None, device=None, return_particles=False):
    """Segment PP YOLO boxes, returning original-size union masks per image.

    With return_particles=True also return per-image particle records. Each
    accepted record stores a LOCAL mask plus crop_box=[x1,y1,x2,y2] so individual
    overlapping particles remain separate without storing a full image per PP.
    area_nm2 and dp_nm use nm/pixel calibration; absent/NaN scales yield NaN.
    Original detection_index connects every record (including rejects) to YOLO.
    No box merging, parent-aggregate assignment or aggregate averaging is done.
    """
    opts = _options(opts)
    if not isinstance(imgs, (list, tuple)) or len(imgs_detect_pp) != len(imgs):
        raise ValueError('Pass an image list and one PP detection dictionary per image.')
    scales = [None] * len(imgs) if pixsizes is None else (
        [pixsizes] * len(imgs) if np.isscalar(pixsizes) else list(pixsizes))
    if len(scales) != len(imgs):
        raise ValueError('pixsizes must have one entry per image.')
    binaries, particles = [], []
    for image_index, (image, detection, scale) in enumerate(
        zip(imgs, imgs_detect_pp, scales), start=1
    ):
        rgb = _rgb(image)
        shape = rgb.shape[:2]
        scale = np.nan if scale is None else float(scale)
        if not np.isnan(scale) and (not np.isfinite(scale) or scale <= 0):
            raise ValueError('Pixel sizes must be positive nm/pixel or None/NaN.')
        boxes, scores, classes, names = _detections(detection, shape)
        print(
            f"YGMAP-pp image {image_index}/{len(imgs)}: "
            f"segmenting {len(boxes)} YOLO candidates...",
            flush=True,
        )
        source_indices = np.asarray(detection.get('source_indices', np.arange(len(boxes))))
        if source_indices.shape != (len(boxes),):
            raise ValueError('source_indices must align with detection boxes.')
        neighbors = _neighbor_counts(boxes, shape, opts)
        union = np.zeros(shape, bool)
        records = []
        initialized = False
        for index, (box, confidence, class_id) in enumerate(zip(boxes, scores, classes)):
            record = {'detection_index': index, 'box': box.copy(), 'class_id': int(class_id),
                      'source_detection_index': int(source_indices[index]),
                      'class_name': names[int(class_id)], 'yolo_confidence': float(confidence),
                      'neighbor_count': int(neighbors[index]), 'status': 'neighbor_filter',
                      'mask': None, 'crop_box': None, 'area_px': 0, 'area_nm2': np.nan,
                      'dp_px': np.nan, 'dp_nm': np.nan, 'sam_predicted_iou': np.nan}
            if neighbors[index] >= opts.min_neighbors:
                if segmenter is None:
                    segmenter = MicroSAM(
                        checkpoint=checkpoint or DEFAULT_CHECKPOINT,
                        device=device,
                    )
                if not initialized:
                    segmenter.begin_image()
                    initialized = True
                crop_box = square_window(box, shape, opts.window_size,
                                         opts.box_padding_fraction, opts.box_padding_pixels)
                x1, y1, x2, y2 = crop_box
                crop = rgb[y1:y2, x1:x2]
                prompt = local_box(box, crop_box, crop.shape[:2])
                masks, sam_scores = segmenter.candidates(crop, prompt, tuple(crop_box))
                valid = np.isfinite(sam_scores)
                record['status'] = 'no_valid_candidate'
                if valid.any():
                    best = int(np.argmax(np.where(valid, sam_scores, -np.inf)))
                    mask = _prompted_component(masks[best], prompt)
                    area = int(mask.sum())
                    record.update(crop_box=crop_box.copy(), sam_predicted_iou=float(sam_scores[best]))
                    if not area or area < opts.min_mask_area:
                        record['status'] = 'small_mask'
                    elif area / max(1, box_area(box)) > opts.max_mask_to_box_ratio:
                        record['status'] = 'mask_box_ratio'
                    else:
                        diameter = float(2 * np.sqrt(area / np.pi))
                        record.update(status='accepted', mask=mask, area_px=area,
                                      area_nm2=area * scale ** 2, dp_px=diameter, dp_nm=diameter * scale)
                        union[y1:y2, x1:x2] |= mask
            if return_particles:
                records.append(record)
            if (index + 1) % 25 == 0 or index + 1 == len(boxes):
                print(
                    f"  image {image_index}/{len(imgs)}: "
                    f"{index + 1}/{len(boxes)} candidates processed",
                    flush=True,
                )
        binaries.append(union)
        particles.append(records)
    return (binaries, particles) if return_particles else binaries




