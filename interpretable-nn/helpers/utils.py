import numpy as np


def get_mask_for_gray(segmentation, threshold=0, exact=None):
    """
    Get mask of tumor as gray image

    :param segmentation: segmentation image
    :param threshold: threshold for tumor region
    :param exact: True if mask is count as equation of threshold
    :return: mask of tumor with shape (height, width, 1)
    """
    if exact:
        return segmentation == exact
    else:
        return segmentation > threshold


def get_mask_of_seg_rgb(segmentation, threshold=0, exact=False):
    """
    Get mask of tumor for rgb image

    :param segmentation: segmentation image
    :param threshold: threshold for tumor region
    :param exact: True if mask is count as equation of threshold
    :return: mask of tumor with shape (height, width, 3)
    """
    height, width = segmentation.shape[0], segmentation.shape[1]
    new_mask = np.zeros((height, width, 3), dtype=bool)
    if exact:
        new_seg = (segmentation == threshold)[:, :, 0]
    else:
        new_seg = (segmentation > threshold)[:, :, 0]
    for ch in range(3):
        new_mask[:, :, ch] = new_seg
    return new_mask


def get_mask_of_brain(image, segmentation=None):
    """
    Get mask of brain from original image as gray image
    mask = brain mask - tumor mask

    :param image: original image
    :param segmentation: tumor segmentation image
    :return: mask of brain without tumor region with shape (height, width, 1)
    """
    if segmentation is not None and image.shape != segmentation.shape:
        raise Exception('Not equal shape of image and segmentation')
    if len(image.shape) == 3:
        image = image.any(axis=np.argmax(np.asarray(image.shape) == 3))

    return image > 0


def get_mask_of_brain_rgb(image, segmentation=None):
    """
    Get mask of brain from original image as rgb image
    mask = brain mask - tumor mask

    :param image: original image
    :param segmentation: tumor segmentation image
    :return: mask of brain without tumor region with shape (height, width, 3)
    """
    if segmentation is not None and image.shape != segmentation.shape:
        raise Exception('Not equal shape of image and segmentation')
    if segmentation is not None:
        mask = image > 0
        mask[segmentation] = False
        return mask
    else:
        return image > 0


def sum_image_channels(image):
    """
    Sum values cross channels in image
    :param image: original image
    :return: ndarray with shape (height, weight, 1)
    """
    return image.sum(axis=np.argmax(np.asarray(image.shape) == 3))


def any_image_channels(image):
    """
    Any value cross channels of image
    :param image: original image
    :return: ndarray with shape (height, weight, 1)
    """
    return image.any(axis=np.argmax(np.asarray(image.shape) == 3))


def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]
