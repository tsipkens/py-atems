"""
Detection wrappers for py-atems.

AUTHOR: Ethan Xiong
"""

__all__ = ["detect_yolo", "detect_ygmap_seg", "detect_ygmap_pp"]

def _detect(imgs, model_group, confidence, iou_threshold, checkpoint_path):
    from .yolo import Detector

    return Detector(
        confidence=confidence,
        iou_threshold=iou_threshold,
        checkpoint_path=checkpoint_path,
        model_group=model_group,
    ).run(imgs)

def detect_yolo(imgs, *, confidence=0.20, iou_threshold=0.70,
                checkpoint_path=None):
    """General aggregate detector; usable with any aggregate segmentation method.

    Existing calls with only ``imgs`` continue to use the original defaults.
    """
    return _detect(imgs, "aggregate", confidence, iou_threshold, checkpoint_path)


def detect_ygmap_seg(imgs, *, confidence=0.20, iou_threshold=0.70,
                     checkpoint_path=None):
    """Aggregate YOLO detections for YGMAP-seg; same detector as detect_yolo."""
    return detect_yolo(imgs, confidence=confidence, iou_threshold=iou_threshold,
                       checkpoint_path=checkpoint_path)


def detect_ygmap_pp(imgs, *, confidence=0.10, iou_threshold=0.70,
                    checkpoint_path=None):
    """Primary-particle YOLO boxes for YGMAP-pp."""
    return _detect(imgs, "pp", confidence, iou_threshold, checkpoint_path)
