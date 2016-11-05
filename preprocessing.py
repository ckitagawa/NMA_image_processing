"""
Module for preprocessing images.
"""

import cv2
import numpy as np

def preprocess(img):
    """
    Takes an image and filters it to improve the quality

    Args
        img: an image to preprocess.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = cv2.fastNlMeansDenoising(img)
    img = cv2.bilateralFilter(img, 5, 75, 75)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 3)
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.erode(img, kernel, iterations=0)

    return img

