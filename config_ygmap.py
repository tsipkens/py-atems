"""Shared model and display settings for the microSAM/YGMAP entry scripts.

  AUTHOR: Ethan Xiong

  How to use this file:
  1. Set IMAGE_FOLDER in the main_*.py script you plan to run.
  2. Adjust only the settings for that script, then run it with Python:
   - main_ygmap_combined.py: YOLO, YGMAP, PP_YOLO, and YGMAP_PP

  DEVICE controls the PyTorch microSAM models; SHOW_RESULTS controls plotting.
  For a checkpoint setting, None uses the default path shown below. Set a path
  to use another compatible model. The model files must exist at those paths.
  The segmentation modules define fallback defaults and validate these options;
  normal users can make their changes here instead.

"""

DEVICE = None  # None selects CUDA when available; or set "cuda" / "cpu"
SHOW_RESULTS = True

# Used only for YGMAP-seg. This aggregate detector can also be called alone
# as det.detect_yolo(imgs). ONNX input size is fixed at export.
YOLO = dict(
    confidence=0.10,
    iou_threshold=0.70,
    checkpoint_path=None,  # None uses det/Config/detectV5.onnx
)

# Used only for YGMAP-seg. None uses agg/config/MSAM_512_V2.pt.
YGMAP = dict(
    checkpoint=None,
    opts=dict(
        window_size=2048,
        large_box_fraction=0.80,
        large_roi_padding=0.50,
        prompt_padding=0.20,
        min_mask_area=50,
        min_component_area=100,
        min_component_fraction=0.03,
        max_mask_fraction=0.45,
        max_mask_to_prompt_ratio=8.0,
        min_prompt_overlap=0.01,
        apply_nmm=True,
        nmm_threshold=0.90,
        nmm_max_size_difference_pct=100.0,
        preserve_contained_boxes=False,
    ),
)

# Used only by main_microsam.py. None uses the standalone checkpoint.
MICROSAM = dict(
    checkpoint=None,
    opts=dict(
        mode="auto",
        use_tiling=True,
        tile_fraction=0.25,
        halo_fraction=1.0,
    ),
)

# Used only by main_ygmap_pp.py.
PP_YOLO = dict(
    confidence=0.10,
    iou_threshold=0.70,
    checkpoint_path=None,  # None uses det/Config/ppdetectV1.onnx
)

# PP microSAM uses a separate fine-tuned PyTorch checkpoint.
YGMAP_PP = dict(
    checkpoint=None,  # None uses pp/config/MSAM_PP_V1.pt
    opts=dict(
        window_size=128,
        box_padding_fraction=0.25,
        box_padding_pixels=4,
        min_mask_area=10,
        max_mask_to_box_ratio=6.0,
        min_neighbors=1,
        neighbor_padding_fraction=0.35,
        neighbor_padding_pixels=6,
    ),
)


