import cv2
import numpy as np


def erode_mask(mask, diameter):
    """Erodes mask using circular kernel."""
    uint8_mask = mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    uint8_eroded = cv2.erode(uint8_mask, kernel)
    eroded_mask = uint8_eroded > 0
    # We must erode at least one pixel around all walls (in the original mask).
    min_eroded_mask = cv2.erode(uint8_mask, np.ones((3, 3)).astype(np.uint8)) > 0
    eroded_mask &= min_eroded_mask
    return eroded_mask
