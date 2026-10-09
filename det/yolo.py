"""
YOLO object detection

AUTHOR: Ethan Xiong, 2026
"""

import ast
from pathlib import Path
import os

import cv2
import numpy as np
import onnxruntime as ort

from tools import tqdm2 as tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = os.getenv("MODEL_DIR", BASE_DIR / "models")

class Detector:
    def __init__(
        self,
        confidence=0.20,
        iou_threshold=0.70,
        checkpoint_path=None,
        model_group="aggregate",
    ):
        # Load the YOLO ONNX model.
        model_files = {
            "aggregate": "YOLO-seg\\ONNX\\detectV7.onnx",
            "pp": "YOLO-pp\\ONNX\\ppdetectV1.onnx",
        }
        if model_group not in model_files:
            raise ValueError("model_group must be 'aggregate' or 'pp'.")
        self.model_group = model_group
        self.checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path is not None
            else MODEL_DIR / model_files[model_group]
        )

        if not 0 <= confidence <= 1 or not 0 <= iou_threshold <= 1:
            raise ValueError("YOLO confidence and IoU threshold must be between 0 and 1.")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"YOLO ONNX model not found: {self.checkpoint_path}")

        self.confidence = confidence
        self.iou_threshold = iou_threshold

        # Use CUDA when available, otherwise use CPU.
        available_providers = ort.get_available_providers()

        providers = []

        if "CUDAExecutionProvider" in available_providers:
            providers.append("CUDAExecutionProvider")

        providers.append("CPUExecutionProvider")

        self.onnx_session = ort.InferenceSession(
            str(self.checkpoint_path),
            providers=providers
        )

        self.input = self.onnx_session.get_inputs()[0]
        self.input_name = self.input.name

        self.output_names = [
            output.name
            for output in self.onnx_session.get_outputs()
        ]

        self.class_names = self._load_class_names()

        self.input_height, self.input_width = (
            self._get_input_size()
        )

        print(f"Loaded YOLO model: {self.checkpoint_path.name}")

    def _load_class_names(self):
        """Read the class names preserved in the model."""

        metadata = self.onnx_session.get_modelmeta().custom_metadata_map
        names_text = metadata.get("names")
        names = ast.literal_eval(names_text)

        return {
            int(class_id): str(class_name)
            for class_id, class_name in names.items()
        }

    def _get_input_size(self):
        """
        Read the model input height and width.

        """

        input_shape = self.input.shape

        height = input_shape[2]
        width = input_shape[3]

        return int(height), int(width)

    @staticmethod
    def prepare_image(image):
        """
        Convert an input image into a uint8 RGB image.

        """

        image = np.asarray(image)

        # Convert non-uint8 images into uint8.
        if image.dtype != np.uint8:
            image_float = image.astype(np.float32)

            finite = np.isfinite(image_float)

            valid_pixels = image_float[finite]

            low = np.percentile(valid_pixels, 1)
            high = np.percentile(valid_pixels, 99)

            if high <= low:
                low = valid_pixels.min()
                high = valid_pixels.max()

            if high > low:
                image_float = np.clip(
                    image_float,
                    low,
                    high
                )

                image_float = (
                    255.0
                    * (image_float - low)
                    / (high - low)
                )

                image = image_float.astype(np.uint8)

            else:
                image = np.zeros(
                    image.shape,
                    dtype=np.uint8
                )

        # Convert image to RGB.
        if image.ndim == 2:
            image_rgb = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2RGB
            )

        elif image.ndim == 3 and image.shape[2] == 1:
            image_rgb = cv2.cvtColor(
                image[:, :, 0],
                cv2.COLOR_GRAY2RGB
            )

        elif image.ndim == 3 and image.shape[2] == 3:
            image_rgb = image.copy()

        elif image.ndim == 3 and image.shape[2] == 4:
            image_rgb = cv2.cvtColor(
                image,
                cv2.COLOR_RGBA2RGB
            )

        return image_rgb

    def letterbox(self, image):
        """
        Resize an image while preserving its aspect ratio.

        """

        original_height, original_width = image.shape[:2]

        scale = min(
            self.input_width / original_width,
            self.input_height / original_height
        )

        resized_width = int(
            round(original_width * scale)
        )

        resized_height = int(
            round(original_height * scale)
        )

        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR
        )

        pad_width = self.input_width - resized_width
        pad_height = self.input_height - resized_height

        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top

        padded = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114)
        )

        return (
            padded,
            scale,
            pad_left,
            pad_top
        )

    def preprocess(self, image):
        """
        Prepare one image for YOLO ONNX inference.
        """

        image_rgb = self.prepare_image(image)

        (
            padded,
            scale,
            pad_left,
            pad_top
        ) = self.letterbox(image_rgb)

        # Convert from uint8 [0, 255] to float32 [0, 1].
        tensor = padded.astype(np.float32) / 255.0

        # HWC -> CHW
        tensor = tensor.transpose(2, 0, 1)

        # Add batch dimension.
        tensor = np.expand_dims(
            tensor,
            axis=0
        )

        tensor = np.ascontiguousarray(
            tensor,
            dtype=np.float32
        )

        return (
            tensor,
            image_rgb,
            scale,
            pad_left,
            pad_top
        )

    @staticmethod
    def xywh_to_xyxy(boxes):
        """
        Convert boxes from:
            center_x, center_y, width, height
        to:
            x1, y1, x2, y2
        """

        converted = np.empty_like(boxes)

        converted[:, 0] = (
            boxes[:, 0] - boxes[:, 2] / 2
        )

        converted[:, 1] = (
            boxes[:, 1] - boxes[:, 3] / 2
        )

        converted[:, 2] = (
            boxes[:, 0] + boxes[:, 2] / 2
        )

        converted[:, 3] = (
            boxes[:, 1] + boxes[:, 3] / 2
        )

        return converted

    def apply_nms(
        self,
        boxes,
        scores,
        class_ids
    ):
        """
        Apply class-aware non-maximum suppression.
        """

        if len(boxes) == 0:
            return np.empty(
                (0,),
                dtype=np.int32
            )

        retained_indices = []

        # Apply NMS separately to each class.
        for class_id in np.unique(class_ids):
            class_indices = np.where(
                class_ids == class_id
            )[0]

            class_boxes = boxes[class_indices]
            class_scores = scores[class_indices]

            boxes_xywh = []

            for box in class_boxes:
                x1, y1, x2, y2 = box

                boxes_xywh.append([
                    float(x1),
                    float(y1),
                    float(x2 - x1),
                    float(y2 - y1),
                ])

            indices = cv2.dnn.NMSBoxes(
                boxes_xywh,
                class_scores.tolist(),
                score_threshold=self.confidence,
                nms_threshold=self.iou_threshold
            )

            if len(indices) > 0:
                indices = np.asarray(
                    indices
                ).reshape(-1)

                retained_indices.extend(
                    class_indices[indices].tolist()
                )

        return np.asarray(
            retained_indices,
            dtype=np.int32
        )

    def postprocess(
        self,
        output,
        original_shape,
        scale,
        pad_left,
        pad_top
    ):
        """
        Convert the raw YOLO output into boxes, classes,
        and confidence scores.

        """

        prediction = np.asarray(output)
        prediction = np.squeeze(prediction)

        if prediction.ndim != 2:
            raise ValueError(
                "Unexpected YOLO output shape: "
                f"{np.asarray(output).shape}"
            )

        if prediction.shape[0] < prediction.shape[1]:
            prediction = prediction.T

        number_of_features = prediction.shape[1]

        # Some end-to-end exports return:
        # x1, y1, x2, y2, confidence, class_id
        if number_of_features == 6:
            boxes = prediction[:, :4]
            scores = prediction[:, 4]
            class_ids = prediction[:, 5].astype(np.int32)

            keep = scores >= self.confidence

            boxes = boxes[keep]
            scores = scores[keep]
            class_ids = class_ids[keep]

        else:
            # Standard raw YOLO export:
            # cx, cy, width, height, class scores...
            boxes_xywh = prediction[:, :4]
            class_scores = prediction[:, 4:]

            class_ids = np.argmax(
                class_scores,
                axis=1
            ).astype(np.int32)

            scores = np.max(
                class_scores,
                axis=1
            )

            keep = scores >= self.confidence

            boxes_xywh = boxes_xywh[keep]
            scores = scores[keep]
            class_ids = class_ids[keep]

            boxes = self.xywh_to_xyxy(
                boxes_xywh
            )

        if len(boxes) == 0:
            return {
                "boxes": np.empty(
                    (0, 4),
                    dtype=np.float32
                ),
                "classes": np.empty(
                    (0,),
                    dtype=np.int32
                ),
                "confidences": np.empty(
                    (0,),
                    dtype=np.float32
                ),
                "class_names": [],
            }

        # Remove letterbox padding.
        boxes[:, [0, 2]] -= pad_left
        boxes[:, [1, 3]] -= pad_top

        # Scale boxes back to the original image size.
        boxes /= scale

        original_height, original_width = original_shape

        boxes[:, [0, 2]] = np.clip(
            boxes[:, [0, 2]],
            0,
            original_width - 1
        )

        boxes[:, [1, 3]] = np.clip(
            boxes[:, [1, 3]],
            0,
            original_height - 1
        )

        retained = self.apply_nms(
            boxes,
            scores,
            class_ids
        )

        boxes = boxes[retained]
        scores = scores[retained]
        class_ids = class_ids[retained]

        names = [
            self.class_names.get(
                int(class_id),
                str(class_id)
            )
            for class_id in class_ids
        ]

        return {
            "boxes": boxes.astype(np.float32),
            "classes": class_ids.astype(np.int32),
            "confidences": scores.astype(np.float32),
            "class_names": names,
        }

    def detect_image(self, image):
        """
        Run YOLO detection on one image.
        """

        (
            input_tensor,
            image_rgb,
            scale,
            pad_left,
            pad_top
        ) = self.preprocess(image)

        outputs = self.onnx_session.run(
            self.output_names,
            {
                self.input_name: input_tensor
            }
        )

        detection = self.postprocess(
            outputs[0],
            image_rgb.shape[:2],
            scale,
            pad_left,
            pad_top
        )

        detection["image_shape"] = (
            image_rgb.shape[:2]
        )

        return detection

    def run(self, imgs):
        """
        Run YOLO detection on a list of images.

        """

        if not imgs:
            return []

        detections = [None] * len(imgs)

        print("Performing YOLO detection:")

        for ii in tqdm(
            range(len(imgs)),
            bar_format="{l_bar}{bar:15}{r_bar}{bar:-15b}"
        ):
            detections[ii] = self.detect_image(
                imgs[ii]
            )

        print("DONE.\n")

        return detections

