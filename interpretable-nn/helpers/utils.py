import numpy as np


def get_mask_for_gray(segmentation, threshold=0, exact=None):
    if exact:
        return segmentation == exact
    else:
        return segmentation > threshold


def get_mask_of_seg_rgb(segmentation, threshold=0, nchannels=3, exact=None):
    height, width = segmentation.shape[0], segmentation.shape[1]
    new_mask = np.zeros((height, width, nchannels), dtype=bool)
    if exact:
        new_seg = (segmentation == exact)[:, :, 0]
    else:
        new_seg = (segmentation > threshold)[:, :, 0]
    for ch in range(nchannels):
        new_mask[:, :, ch] = new_seg
    return new_mask


def get_mask_of_brain(image, segmentation=None):
    if segmentation is not None and image.shape != segmentation.shape:
        raise Exception('Not equal shape of image and segmentation')
    if len(image.shape) == 3:
        image = image.any(axis=np.argmax(np.asarray(image.shape) == 3))

    return image > 0


def get_mask_of_brain_rgb(image, segmentation=None):
    if segmentation is not None and image.shape != segmentation.shape:
        raise Exception('Not equal shape of image and segmentation')
    if segmentation is not None:
        mask = image > 0
        mask[segmentation] = False
        return mask
    else:
        return image > 0


def sum_image_channels(image):
    return image.sum(axis=np.argmax(np.asarray(image.shape) == 3))


def any_image_channels(image):
    return image.any(axis=np.argmax(np.asarray(image.shape) == 3))