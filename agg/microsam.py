"""Standalone and YOLO-guided aggregate MicroSAM segmentation.

AUTHOR: Ethan Xiong

"""
from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

__all__ = ["MicroSAM", "StandaloneMicroSAM", "segment_standalone", "segment_ygmap"]

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = os.getenv("MODEL_DIR", BASE_DIR / "models")
DEFAULT_CHECKPOINT = MODEL_DIR / "MicroSAM-seg\\PyTorch" / "MSAM_512_V2.safe.pt"
STANDALONE_CHECKPOINT = MODEL_DIR / "MicroSAM-seg\\PyTorch" / "microsam_standalone.pt"  # NOTE: currently not included

@dataclass(frozen=True)
class GuidedOptions:
    # Aggregate settings, with configurable overlap and box-size guards.
    # Fallback defaults for direct calls to the segmentation function.
    # For normal YGMAP runs, change settings in config_ygmap.py instead.
    window_size: int = 640
    large_box_fraction: float = 0.80
    large_roi_padding: float = 0.50
    prompt_padding: float = 0.20
    min_mask_area: int = 50
    min_component_area: int = 100
    min_component_fraction: float = 0.03
    max_mask_fraction: float = 0.45
    max_mask_to_prompt_ratio: float = 8.0
    min_prompt_overlap: float = 0.01
    apply_nmm: bool = True
    nmm_threshold: float = 0.90
    # 100% allows at most a 2:1 area ratio; None disables the size guard.
    nmm_max_size_difference_pct: float | None = 100.0
    preserve_contained_boxes: bool = False


@dataclass(frozen=True)
class StandaloneOptions:
    # Match the working package's main_microsam.py, including halo_fraction=1.
    mode: str = "auto"
    use_tiling: bool = True
    tile_fraction: float = 0.25
    halo_fraction: float = 1.0


def _options(cls, opts):
    result = cls(**(opts or {})) if isinstance(opts, (dict, type(None))) else opts
    if not isinstance(result, cls):
        raise TypeError(f"opts must be a dict or {cls.__name__}.")
    if isinstance(result, StandaloneOptions):
        if result.mode not in ("auto", "ais", "amg", "apg"):
            raise ValueError("mode must be auto, ais, amg or apg.")
        if not isinstance(result.use_tiling, bool):
            raise ValueError("use_tiling must be a bool.")
        if not np.isfinite(result.tile_fraction) or not 0 < result.tile_fraction <= 1:
            raise ValueError("tile_fraction must be in (0, 1].")
        if not np.isfinite(result.halo_fraction) or result.halo_fraction < 0:
            raise ValueError("halo_fraction must be finite and nonnegative.")
    for name in ("min_mask_area", "min_component_area", "crop_n_layers"):
        if hasattr(result, name):
            value = getattr(result, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
    for name in ("window_size", "points_per_side", "points_per_batch"):
        if hasattr(result, name):
            value = getattr(result, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
    for name in ("max_mask_fraction", "min_component_fraction", "large_box_fraction",
                 "min_prompt_overlap", "pred_iou_thresh", "stability_score_thresh",
                 "box_nms_thresh", "nmm_threshold"):
        if hasattr(result, name) and not 0 <= getattr(result, name) <= 1:
            raise ValueError(f"{name} must be between 0 and 1.")
    for name in ("large_roi_padding", "prompt_padding", "max_mask_to_prompt_ratio"):
        if hasattr(result, name):
            value = getattr(result, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
    if isinstance(result, GuidedOptions) and result.nmm_threshold == 0:
        raise ValueError("nmm_threshold must be greater than zero.")
    if isinstance(result, GuidedOptions):
        limit = result.nmm_max_size_difference_pct
        if limit is not None and (not np.isfinite(limit) or limit < 0):
            raise ValueError("nmm_max_size_difference_pct must be finite and nonnegative, or None.")
    return result


def _images(imgs, pixsizes):
    if not isinstance(imgs, (list, tuple)):
        raise TypeError("imgs must be a list of images, e.g. [image].")
    if pixsizes is not None and not np.isscalar(pixsizes) and len(pixsizes) != len(imgs):
        raise ValueError("pixsizes must have one entry per image.")
    return imgs


def _rgb(image):
    """Preserve uint8 intensities; use the reference percentile scaling otherwise."""
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (3, 4)):
        raise ValueError(f"Expected grayscale, RGB or RGBA image, got {image.shape}.")
    if not image.size:
        raise ValueError("Image must not be empty.")
    if image.dtype != np.uint8:
        values = image.astype(np.float32)
        finite = np.isfinite(values)
        if not finite.any():
            raise ValueError("Image contains no finite values.")
        low, high = np.percentile(values[finite], [1, 99])
        if high <= low:
            low, high = values[finite].min(), values[finite].max()
        if high > low:
            values = 255 * (np.clip(values, low, high) - low) / (high - low)
            values[~finite] = 0
            image = np.rint(values).astype(np.uint8)
        else:
            image = np.zeros(image.shape, dtype=np.uint8)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = image[:, :, :3]
    return np.ascontiguousarray(image)


class MicroSAM:
    """Reusable fine-tuned SAM model, loaded once and shared across calls.
    
    """

    def __init__(self, checkpoint=None, model_type="vit_b", device=None):
        checkpoint = Path(checkpoint) if checkpoint is not None else DEFAULT_CHECKPOINT
        if not checkpoint.is_file():
            raise FileNotFoundError(f"MicroSAM weights not found: {checkpoint}")
        try:
            import torch
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise ImportError(
                "MicroSAM requires torch, torchvision and segment-anything. "
                "See MICROSAM.md in the package root."
            ) from exc
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ValueError("Expected a SAM weights dictionary.")
        state = state.get("model_state", state.get("state_dict", state))
        state = {key.removeprefix("sam."): value for key, value in state.items()}
        if model_type not in sam_model_registry:
            raise ValueError(f"Unknown SAM model type: {model_type}")
        self.model = sam_model_registry[model_type]()
        self.model.load_state_dict(state, strict=True)
        self.model.to(device=self.device)
        self.model.eval()
        self.predictor = SamPredictor(self.model)
        self._crop_key = None

    def begin_image(self):
        # Never reuse an embedding across different images with identical shapes.
        self._crop_key = None
        self.predictor.reset_image()

    def candidates(self, crop_rgb, prompt_box, crop_key):
        import torch
        if self._crop_key != crop_key:
            self.predictor.set_image(crop_rgb)
            self._crop_key = crop_key
        # Keep the reference's float32 torch coordinate transform. The NumPy
        # predict() route can differ at boundary pixels due to float rounding.
        box = torch.tensor(prompt_box[None, :], dtype=torch.float32, device=self.device)
        transformed = self.predictor.transform.apply_boxes_torch(box, crop_rgb.shape[:2])
        with torch.no_grad():
            masks, scores, _ = self.predictor.predict_torch(
                point_coords=None, point_labels=None, boxes=transformed,
                multimask_output=True,
            )
        return masks.detach().cpu().numpy()[0].astype(bool), scores.detach().cpu().numpy()[0].astype(float)

class StandaloneMicroSAM:
    """MicroSAM automatic instance segmentation with the standalone decoder.

    Kept separate from the guided predictor: both checkpoint and inference
    algorithm differ. All standalone model and inference logic lives here.
    """

    def __init__(self, checkpoint=None, model_type="vit_b", device=None, opts=None):
        self.opts = _options(StandaloneOptions, opts)
        checkpoint = Path(checkpoint) if checkpoint is not None else STANDALONE_CHECKPOINT
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Standalone MicroSAM weights not found: {checkpoint}")
        # micro_sam imports Numba's cached functions through torch_em/napari.
        # On restricted Windows installs, its default package cache can stall
        # during import. Keep a writable persistent cache outside site-packages.
        if "NUMBA_CACHE_DIR" not in os.environ:
            cache_dir = Path(tempfile.gettempdir()) / "atems_numba_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)
        try:
            from micro_sam.automatic_segmentation import (
                automatic_instance_segmentation, get_predictor_and_segmenter,
            )
        except ImportError as exc:
            raise ImportError("Standalone segmentation requires micro-sam; see MICROSAM.md.") from exc
        self.predictor, self.segmenter = get_predictor_and_segmenter(
            model_type=model_type, checkpoint=str(checkpoint), device=device,
            segmentation_mode=self.opts.mode, is_tiled=self.opts.use_tiling,
        )
        self._automatic_instance_segmentation = automatic_instance_segmentation

    def automatic(self, image, opts):
        if opts != self.opts:
            raise ValueError("Standalone segmenter options differ; create a new StandaloneMicroSAM.")
        tile_args = {}
        if opts.use_tiling:
            height, width = image.shape[:2]
            tile_shape = tuple(max(1, round(size * opts.tile_fraction)) for size in (height, width))
            halo = tuple(max(1, round(size * opts.halo_fraction)) for size in tile_shape)
            tile_args = {"tile_shape": tile_shape, "halo": halo}
        return np.asarray(self._automatic_instance_segmentation(
            predictor=self.predictor, segmenter=self.segmenter, input_path=image,
            ndim=2, verbose=False, **tile_args,
        ))


def _segment_aggregate(segmenter, image_rgb, box, opts):
    height, width = image_rgb.shape[:2]
    threshold = int(opts.large_box_fraction * opts.window_size)
    if box[2] - box[0] <= threshold and box[3] - box[1] <= threshold:
        crop_box = square_window(box, (height, width), opts.window_size, 0.0)
        mode = "fixed_window"
    else:
        crop_box = pad_box(box, (height, width), opts.large_roi_padding)
        mode = "padded_roi"
    x1, y1, x2, y2 = crop_box
    crop_rgb = image_rgb[y1:y2, x1:x2]
    prompt_full = pad_box(box, (height, width), opts.prompt_padding)
    prompt = local_box(prompt_full, crop_box, crop_rgb.shape[:2])
    candidates, scores = segmenter.candidates(crop_rgb, prompt, tuple(crop_box))
    crop_area = crop_rgb.shape[0] * crop_rgb.shape[1]
    prompt_area = max(1.0, box_area(prompt))
    px1, py1, px2, py2 = np.rint(prompt).astype(int)
    best_mask, best_quality, best_score = None, -np.inf, np.nan
    for mask, score in zip(candidates, scores):
        area = int(mask.sum())
        if not area or not np.isfinite(score):
            continue
        fraction, ratio = area / crop_area, area / prompt_area
        overlap = int(mask[py1:py2, px1:px2].sum()) / area
        if (fraction > opts.max_mask_fraction or ratio > opts.max_mask_to_prompt_ratio
                or overlap < opts.min_prompt_overlap):
            continue
        quality = float(score) - 0.75 * fraction - 0.05 * ratio
        if quality > best_quality:
            best_mask, best_quality, best_score = mask, quality, float(score)
    full = np.zeros((height, width), dtype=bool)
    if best_mask is None:
        return full, best_score, mode, "no_valid_candidate"
    clean = remove_small_components(best_mask, opts.min_component_area, opts.min_component_fraction)
    if not clean.any() or int(clean.sum()) < opts.min_mask_area:
        return full, best_score, mode, "small_mask"
    full[y1:y2, x1:x2] = clean
    return full, best_score, mode, "accepted"


def _detections(detection, shape):
    """Validate and copy det.detect_yolo output; never mutate its arrays."""
    if "image_shape" in detection and tuple(detection["image_shape"]) != tuple(shape):
        raise ValueError("YOLO image_shape differs from the image supplied to MicroSAM.")
    boxes = np.asarray(detection["boxes"], dtype=np.float32).copy()
    scores = np.asarray(detection["confidences"], dtype=np.float32).copy()
    raw_classes = np.asarray(detection["classes"])
    names = list(detection["class_names"])
    if boxes.size == 0:
        boxes = boxes.reshape(0, 4)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("Detection boxes must have shape (N, 4) in original-image XYXY coordinates.")
    n = len(boxes)
    if scores.shape != (n,) or raw_classes.shape != (n,) or len(names) != n:
        raise ValueError("Detection boxes, classes, confidences and class_names must align.")
    if not (np.isfinite(boxes).all() and np.isfinite(scores).all()
            and np.isfinite(raw_classes).all()):
        raise ValueError("Detections must contain finite values.")
    if not np.equal(raw_classes, np.floor(raw_classes)).all() or (raw_classes < 0).any():
        raise ValueError("Class IDs must be nonnegative integers.")
    classes = raw_classes.astype(np.int32)
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("Detection confidences must be between 0 and 1.")
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, shape[1])
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, shape[0])
    if (boxes[:, 2:] <= boxes[:, :2]).any():
        raise ValueError("Detection boxes must have positive area inside the image.")
    name_map = {}
    for class_id, name in zip(classes, names):
        if int(class_id) in name_map and name_map[int(class_id)] != str(name):
            raise ValueError("Each class ID must have a consistent class name.")
        name_map[int(class_id)] = str(name)
    return boxes, scores, classes, name_map


def segment_ygmap(imgs, imgs_detect, pixsizes=None, *, checkpoint=None,
                      model_type="vit_b", device=None, opts=None,
                      segmenter=None, return_instances=False):
    """YGMAP-seg: use existing YOLO detections as box prompts for MicroSAM.

    imgs_detect is the list returned by det.detect_yolo(imgs). Returns a list
    of same-size boolean masks. With return_instances=True, returns
    (imgs_binary, per_image_records). Records include individual full-size masks,
    class/confidence, merged box, merge count, SAM score and acceptance status.
    Rejected detections remain in records with an empty mask and reason.

    NMM uses overlap plus an area-difference guard on the detections still available. 
    Boxes already removed by the YOLO detector cannot be recovered here.
    """
    imgs = _images(imgs, pixsizes)
    if len(imgs_detect) != len(imgs):
        raise ValueError("imgs_detect must have one detection dictionary per image.")
    opts = _options(GuidedOptions, opts)
    binaries, records = [], []
    for image, detection in zip(imgs, imgs_detect):
        rgb = _rgb(image)
        shape = rgb.shape[:2]
        boxes, scores, classes, names = _detections(detection, shape)
        if opts.apply_nmm:
            boxes, scores, classes, sizes = class_aware_nmm(
                boxes, scores, classes, opts.nmm_threshold, opts.preserve_contained_boxes,
                opts.nmm_max_size_difference_pct)
        else:
            sizes = np.ones(len(boxes), dtype=np.int32)
        binary = np.zeros(shape, dtype=bool)
        image_records = []
        if len(boxes):
            if segmenter is None:
                segmenter = MicroSAM(checkpoint, model_type, device)
            segmenter.begin_image()
        for box, score, class_id, size in zip(boxes, scores, classes, sizes):
            mask, sam_score, mode, status = _segment_aggregate(segmenter, rgb, box, opts)
            binary |= mask
            if return_instances:
                image_records.append({
                    "mask": mask, "box": box.copy(), "class_id": int(class_id),
                    "class_name": names[int(class_id)], "yolo_confidence": float(score),
                    "nmm_group_size": int(size), "sam_predicted_iou": sam_score,
                    "sam_window_mode": mode, "status": status,
                })
        binaries.append(binary)
        records.append(image_records)
    return (binaries, records) if return_instances else binaries


def segment_standalone(imgs, pixsizes=None, *, checkpoint=None, model_type="vit_b",
                 device=None, opts=None, segmenter=None, return_instances=False):
    """Standalone tiled MicroSAM, matching the supplied working package.

    Uses its separate checkpoint including decoder_state, mode='auto', tiles
    spanning 25% of each image dimension and halo=100% of each tile dimension.
    Input intensities are preserved for MicroSAM's own preprocessing. 
    """
    imgs = _images(imgs, pixsizes)
    opts = _options(StandaloneOptions, opts)
    binaries, records = [], []
    for image in imgs:
        image = np.asarray(image)
        if image.ndim == 3 and image.shape[2] == 1:
            image = image[:, :, 0]
        if (image.ndim not in (2, 3) or not image.size
                or (image.ndim == 3 and image.shape[2] != 3)):
            raise ValueError("Standalone inputs must be nonempty grayscale or RGB images.")
        if not np.isfinite(image).all():
            raise ValueError("Standalone inputs must contain only finite values.")
        if segmenter is None:
            segmenter = StandaloneMicroSAM(checkpoint, model_type, device, opts)
        if isinstance(segmenter, MicroSAM):
            raise TypeError("Use StandaloneMicroSAM for standalone inference; MicroSAM is box-guided.")
        prediction = np.asarray(segmenter.automatic(image, opts))
        if prediction.shape != image.shape[:2]:
            raise ValueError("MicroSAM prediction does not match the original image shape.")
        binary = prediction > 0
        image_records = []
        if return_instances:
            for label_id in np.unique(prediction[prediction > 0]):
                image_records.append({
                    "mask": prediction == label_id, "instance_id": int(label_id),
                    "status": "accepted",
                })
        binaries.append(binary)
        records.append(image_records)
    return (binaries, records) if return_instances else binaries


# Geometry and component filtering retained from the supplied workflow.
def box_area(box: np.ndarray) -> float:
    x1, y1, x2, y2 = np.asarray(box, dtype=float)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def nmm_overlap(
    box_a: np.ndarray,
    box_b: np.ndarray,
    preserve_contained_boxes: bool,
) -> float:
    """Return the overlap measure used to decide whether boxes are merged."""

    ax1, ay1, ax2, ay2 = np.asarray(box_a, dtype=float)
    bx1, by1, bx2, by2 = np.asarray(box_b, dtype=float)

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = box_area(box_a)
    area_b = box_area(box_b)
    if preserve_contained_boxes:
        reference_area = max(area_a, area_b)
    else:
        reference_area = min(area_a, area_b)
    return intersection / reference_area if reference_area > 0 else 0.0

def class_aware_nmm(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    overlap_threshold: float,
    preserve_contained_boxes: bool,
    max_size_difference_pct: float | None = 100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Merge overlapping same-class boxes with compatible original areas.

    The merged rectangle uses the furthest corners of every box in the group.
    Its confidence is the maximum member confidence. The final return value is
    the number of original boxes represented by each output box.
    """

    if max_size_difference_pct is not None and (
            not np.isfinite(max_size_difference_pct) or max_size_difference_pct < 0):
        raise ValueError("max_size_difference_pct must be finite and nonnegative, or None.")
    areas = np.asarray([box_area(box) for box in boxes])

    def size_compatible(first, second):
        if max_size_difference_pct is None:
            return True
        smaller = min(areas[first], areas[second])
        larger = max(areas[first], areas[second])
        return smaller > 0 and larger <= smaller * (1 + max_size_difference_pct / 100)

    if len(boxes) == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
        )

    output_boxes, output_scores, output_classes, group_sizes = [], [], [], []

    for class_id in np.unique(class_ids):
        class_indices = np.where(class_ids == class_id)[0]
        unvisited = set(class_indices.tolist())

        while unvisited:
            seed = min(unvisited, key=lambda i: (-float(scores[i]), i))
            unvisited.remove(seed)
            group = [seed]
            pending = [seed]

            while pending:
                current = pending.pop()
                for candidate in sorted(unvisited, key=lambda i: (-float(scores[i]), i)):
                    if not all(size_compatible(candidate, member) for member in group):
                        continue
                    if nmm_overlap(
                        boxes[current],
                        boxes[candidate],
                        preserve_contained_boxes,
                    ) >= overlap_threshold:
                        unvisited.remove(candidate)
                        group.append(candidate)
                        pending.append(candidate)

            indices = np.asarray(group, dtype=int)
            member_boxes = boxes[indices]
            output_boxes.append(
                [
                    float(member_boxes[:, 0].min()),
                    float(member_boxes[:, 1].min()),
                    float(member_boxes[:, 2].max()),
                    float(member_boxes[:, 3].max()),
                ]
            )
            output_scores.append(float(scores[indices].max()))
            output_classes.append(int(class_id))
            group_sizes.append(len(group))

    order = np.argsort(np.asarray(output_scores))[::-1]
    return (
        np.asarray(output_boxes, dtype=np.float32)[order],
        np.asarray(output_scores, dtype=np.float32)[order],
        np.asarray(output_classes, dtype=np.int32)[order],
        np.asarray(group_sizes, dtype=np.int32)[order],
    )

def pad_box(box, image_shape, padding_fraction=0.0, padding_pixels=0.0):
    height, width = image_shape
    x1, y1, x2, y2 = np.asarray(box, dtype=float)
    padding_x = max(
        float(padding_pixels),
        float(padding_fraction) * (x2 - x1),
    )
    padding_y = max(
        float(padding_pixels),
        float(padding_fraction) * (y2 - y1),
    )
    return np.asarray(
        [
            max(0, math.floor(x1 - padding_x)),
            max(0, math.floor(y1 - padding_y)),
            min(width, math.ceil(x2 + padding_x)),
            min(height, math.ceil(y2 + padding_y)),
        ],
        dtype=np.int32,
    )

def square_window(box, image_shape, minimum_size, padding_fraction, padding_pixels=0):
    height, width = image_shape
    padded = pad_box(box, image_shape, padding_fraction, padding_pixels)
    x1, y1, x2, y2 = padded
    center_x, center_y = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    size = max(int(minimum_size), int(x2 - x1), int(y2 - y1))
    size = min(size, max(height, width))

    crop_x1 = int(round(center_x - size / 2))
    crop_y1 = int(round(center_y - size / 2))
    crop_x2, crop_y2 = crop_x1 + size, crop_y1 + size

    if crop_x1 < 0:
        crop_x2 -= crop_x1
        crop_x1 = 0
    if crop_y1 < 0:
        crop_y2 -= crop_y1
        crop_y1 = 0
    if crop_x2 > width:
        crop_x1 = max(0, crop_x1 - (crop_x2 - width))
        crop_x2 = width
    if crop_y2 > height:
        crop_y1 = max(0, crop_y1 - (crop_y2 - height))
        crop_y2 = height

    return np.asarray([crop_x1, crop_y1, crop_x2, crop_y2], dtype=np.int32)

def local_box(full_box, crop_box, crop_shape):
    crop_x1, crop_y1, _, _ = crop_box
    crop_height, crop_width = crop_shape
    x1, y1, x2, y2 = np.asarray(full_box, dtype=float)
    return np.asarray(
        [
            np.clip(x1 - crop_x1, 0, crop_width - 1),
            np.clip(y1 - crop_y1, 0, crop_height - 1),
            np.clip(x2 - crop_x1, 1, crop_width),
            np.clip(y2 - crop_y1, 1, crop_height),
        ],
        dtype=np.float32,
    )

def remove_small_components(
    mask: np.ndarray,
    minimum_area: int,
    minimum_fraction_of_largest: float = 0.0,
) -> np.ndarray:
    """Remove disconnected mask fragments below an absolute/relative cutoff.

    The effective cutoff is the larger of ``minimum_area`` and
    ``minimum_fraction_of_largest * largest_component_area``. The largest
    connected component is therefore always retained when it satisfies the
    absolute minimum-area requirement.
    """

    binary = mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    if count <= 1:
        return np.zeros(binary.shape, dtype=bool)

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_area = int(component_areas.max())
    effective_minimum = max(
        int(minimum_area),
        int(math.ceil(largest_area * minimum_fraction_of_largest)),
    )

    clean = np.zeros(binary.shape, dtype=bool)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= effective_minimum:
            clean[labels == label] = True
    return clean

