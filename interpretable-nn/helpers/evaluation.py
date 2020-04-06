import sys
sys.path.append('../')

import numpy as np
from helpers import utils


def mask_value(heatmap, image,  mask):
    if image.shape != mask.shape:
        raise Exception('Not equal shape of image and segmentation')

    if mask.sum():
        brain_mask = utils.get_mask_of_brain_rgb(image, mask)
        tumor_val = heatmap[mask].sum() / mask.sum()
        brain_val = heatmap[brain_mask].sum() / brain_mask.sum()
        return tumor_val / (tumor_val + brain_val)
    else:
        return 0
