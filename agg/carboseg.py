
import functools
from pathlib import Path
import os

import numpy as np

import albumentations as albu
import onnxruntime as ort

from PIL import Image

from skimage.morphology import disk, closing, opening, remove_small_objects

from tqdm import tqdm


class Classifier:
    """
    A class used to implement the neural network associated with carboseg.
    """

    def __init__(
            self,
    ):

        self.checkpoint_path = Path(__file__).parent / "config\\FPN-resnet50-imagenet.onnx"

        self.onnx_session = ort.InferenceSession(str(self.checkpoint_path))
        self.input_name = self.onnx_session.get_inputs()[0].name

    @staticmethod
    def validate_augmentation(image):
        """ Validates the input image size. """

        # Add paddings to make image shape divisible by 32.
        test_transform = [albu.PadIfNeeded(384, 480)]
        fun = albu.Compose(test_transform)

        # Return validated image.
        return fun(image=image)["image"]

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
        """ Pre-process the image prior to classification. """

        # Get preprocessing function, using default parameters (change if image does not meet these specs).
        # See format_preprocess_input method above, for how these parameters are used.
        params = {
            "input_space": "RGB",
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }
        preprocessing_fun = functools.partial(self.format_preprocess_input, **params)

        # Set up preprocessing.
        _transform = [
            albu.Lambda(image=preprocessing_fun),
            albu.Lambda(image=self.to_tensor_image, mask=self.to_tensor_mask),
        ]
        fun = albu.Compose(_transform)

        # Apply processing and return result.
        return fun(image=image)["image"]

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

    def run(self, imgs):
        """
        Upper level wrapper to classify a series of images.
        Takes a list of file paths as input.
        """

        predictions = [None] * len(imgs)  # initialize the predictions list

        # Loop through images and generate predictions.
        print("Performing carboseg segmentation:")
        for ii in tqdm(range(len(imgs)), bar_format="{l_bar}{bar:15}{r_bar}{bar:-15b}"):
            img = Image.fromarray(imgs[ii]).convert("RGB")  # read in image
            predictions[ii] = self.classify_image(img)  # run classifier on image
        print("DONE.\n")

        return predictions

    @staticmethod
    def save_predictions(predictions, image_paths, folder="output"):
        """ Utility to save predictions to the specified folder. """

        # Loop through image paths.
        for ii in range(len(predictions)):
            Image.fromarray(predictions[ii]).save(folder + os.path.sep + os.path.basename(image_paths[ii]))

        print("Images saved.")
