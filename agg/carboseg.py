
import functools
from pathlib import Path
import os

import numpy as np

# import albumentations as albu
import onnxruntime as ort

from PIL import Image

from tools import tqdm2 as tqdm

from concurrent.futures import ThreadPoolExecutor


class Classifier:
    """
    A class used to implement the neural network associated with carboseg.
    """
    def __init__(self):
        self.checkpoint_path = Path(__file__).parent / "config\\FPN-resnet50-imagenet.onnx"

        self.onnx_session = ort.InferenceSession(str(self.checkpoint_path))
        self.input_name = self.onnx_session.get_inputs()[0].name

    @staticmethod
    def validate_augmentation(image):
        """ 
        Validates the input image size by padding to 384x480. 
        Pure NumPy replacement for albu.PadIfNeeded.
        """
        target_h, target_w = 384, 480
        h, w = image.shape[:2]

        # Calculate total padding needed
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)

        # Split padding to apply it to both sides (Top/Bottom, Left/Right)
        # This matches Albumentations default "center" padding
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        # Apply padding (constant 0/black is the default)
        # Only pad the first two dimensions (H, W), not the color channels
        if image.ndim == 3:
            return np.pad(image, ((top, bottom), (left, right), (0, 0)), mode="constant")
        else:
            return np.pad(image, ((top, bottom), (left, right)), mode="constant")

    @staticmethod
    def to_tensor_image(x, **kwargs):
        return x.transpose(2, 0, 1).astype("float32")

    @staticmethod
    def to_tensor_mask(x, **kwargs):
        return np.expand_dims(x, axis=0).astype("float32")

    @staticmethod
    def format_preprocess_input(x, mean=None, std=None, input_space="RGB", input_range=None, **kwargs):
        """ From segmentation-models-pytorch package. """

        if input_space == "BGR":
            x = x[..., ::-1].copy()

        if input_range is not None:
            if x.max() > 1 and input_range[1] == 1:
                x = x / 255.0

        if mean is not None:
            mean = np.array(mean)
            x = x - mean

        if std is not None:
            std = np.array(std)
            x = x / std

        return x

    def preprocess(self, image):
        """ 
        Pre-process the image prior to classification using pure NumPy. 
        Replaces Albumentations dependency for this step.
        """
        # 1. Define the ImageNet normalization parameters
        params = {
            "input_space": "RGB",
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }

        # 2. Run the formatting/normalization logic (Method you already have)
        # This handles the RGB/BGR check and the Mean/Std subtraction
        x = self.format_preprocess_input(image, **params)

        # 3. Transpose to Tensor format: (H, W, C) -> (C, H, W)
        # Using your existing static method logic
        x = self.to_tensor_image(x)

        return x

    def classify_image(self, image):
        """
        Run classifier on a single image.
        Takes a single PIL Image as input.
        """
        # Start by prepare image input for classification.
        image = np.asarray(image)
        image = self.validate_augmentation(image)
        image = self.preprocess(image)
        image = np.expand_dims(image, 0)

        # Get raw prediction.
        input_name = self.onnx_session.get_inputs()[0].name
        prediction = self.onnx_session.run(None, {input_name: image.astype(np.float32)})[0]

        # Format and return prediction.
        return prediction.squeeze().round().astype(bool)

    # def run(self, imgs):
    #     """
    #     Upper level wrapper to classify a series of images.
    #     Takes a list of file paths as input.
    #     """
    #     predictions = [None] * len(imgs)  # initialize the predictions list

    #     # Loop through images and generate predictions.
    #     print("Performing carboseg segmentation:")
    #     for ii in tqdm(range(len(imgs)), bar_format="{l_bar}{bar:15}{r_bar}{bar:-15b}"):
    #         img = Image.fromarray(imgs[ii]).convert("RGB")  # read in image
    #         predictions[ii] = self.classify_image(img)  # run classifier on image
    #     print("DONE.\n")

    #     return predictions
    
    def run(self, imgs):
        """
        Processes images in parallel batches of 3.
        Maintains the fixed Batch=1 requirement of the ONNX model.
        """
        if not imgs:
            return []

        print(f"Performing carboseg segmentation (batch=3):")
        
        # Internal helper to handle the per-image logic
        def process_one(img_data):
            img = Image.fromarray(img_data).convert("RGB")
            # Uses the NumPy preprocess/validate methods we built earlier
            x = self.validate_augmentation(np.asarray(img))
            x = self.preprocess(x)
            
            # Add the mandatory batch dimension of 1
            input_tensor = np.expand_dims(x, 0).astype(np.float32)
            
            # Run inference
            pred = self.onnx_session.run(None, {self.input_name: input_tensor})[0]
            return pred.squeeze().round().astype(bool)

        # Execute with 3 threads
        # This allows 3 images to be in different stages (prep/inference) at once
        with ThreadPoolExecutor(max_workers=4) as executor:
            predictions = list(tqdm(
                executor.map(process_one, imgs), 
                total=len(imgs),
                bar_format="{l_bar}{bar:15}{r_bar}{bar:-15b}"
            ))

        print("DONE.\n")
        return predictions

    @staticmethod
    def save_predictions(predictions, image_paths, folder="output"):
        """ Utility to save predictions to the specified folder. """

        # Loop through image paths.
        for ii in range(len(predictions)):
            Image.fromarray(predictions[ii]).save(folder + os.path.sep + os.path.basename(image_paths[ii]))

        print("Images saved.")
