import sys

sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
from helpers import utils, plots
from itertools import chain


def mask_value(heatmap, image, mask):
    """
    Function to count metric how analysis hit into mask.
    :param heatmap: saliency or activation map obtained from analyzer
    :param image: original image
    :param mask: mask of region where prediction should hit
    :return: score
    """
    if image.shape != mask.shape:
        raise Exception('Not equal shape of image and segmentation')

    if mask.sum():
        brain_mask = utils.get_mask_of_brain_rgb(image, mask)
        tumor_val = heatmap[mask].sum() / mask.sum()
        brain_val = heatmap[brain_mask].sum() / brain_mask.sum()
        return tumor_val / (tumor_val + brain_val)
    else:
        return 0


def evaluate_analyzer(analyzer, images, segmentation, tumor_region=0):
    """
    Evaluate analyzer on batch of images
    :param analyzer: method of interpretation to be used
    :param images: original images to by processed
    :param segmentation: segmentation of tumors from images
    :param tumor_region: which tumor region should by analyzed (default 0: whole segmentation)
    :return: vector of analyzer scores
    """
    analysis = analyzer.analyze(images)
    pred_analysis = [mask_value(a, i, utils.get_mask_of_seg_rgb(s, tumor_region, exact=tumor_region > 0))for a, i, s in zip(analysis, images, segmentation)]
    return pred_analysis


def evaluate_method(model, analyzer, images, segmentation, y, tumor_region=0):
    """
    Evaluate interpretation on batch of images
    :param model: neural network model
    :param analyzer: interpretation technique model
    :param images: original images
    :param segmentation: tumors segmentation of images
    :param y: list of true values for prediction
    :param tumor_region: number of tumor region (0,1,2,3,4)
    :return: list of tuples (true value, model prediction, interpretation score)
    """
    prediction = model.predict_on_batch(images)
    analysis = evaluate_analyzer(analyzer, images, segmentation, tumor_region)
    pred = [x.argmax() for x in prediction]
    return zip(y.astype('int16'), pred, analysis)


def evaluate_method_generator(model, analyzer, generator, tumor_region=0):
    """
    Evaluate interpretation with generator
    :param model: neural network model
    :param analyzer: interpretation technique model
    :param generator: generator of images and segmentation
    :param tumor_region: number of tumor region (0,1,2,3,4)
    :return: list of tuples (true value, model prediction, interpretation score)
    """
    final_iterator = iter([])
    for (x, y), (x_seg, y_seg) in generator:
        final_iterator = chain(
            final_iterator,
            evaluate_method(model, analyzer, x, x_seg, y, tumor_region)
        )


def process_analysis_image(img):
    img = img.sum(axis=np.argmax(np.asarray(img.shape) == 3))
    img /= np.max(np.abs(img))
    return img


def evaluate_method_visualize(model, analyzer, x, x_seg, y):
    prediction = model.predict_on_batch(x)
    analysis = analyzer.analyze(x)
    prob = [x.max() for x in prediction]
    y_hat = [x.argmax() for x in prediction]
    y_analysis = [mask_value(a, i, utils.get_mask_of_seg_rgb(s))
                  for a, i, s in zip(analysis, x, x_seg)]
    analysis = [process_analysis_image(img) for img in analysis]
    results = zip(prob, y, y_hat, y_analysis)
    titles = [
        ('Prob: {:.2f}'.format(p), 'Label: {}'.format(y),
         'Pred: {}'.format(y_h), 'Eval: {:.2f}'.format(y_a))
        for p, y, y_h, y_a in results]

    plots.plot_rgb_images(x, titles=titles)
    plots.plot_gray_images(x_seg)
    plots.plot_analysis_images(analysis)


def evaluate_masks_visualize(model, analyzer, x, x_seg, y, figsize=(18, 5)):
    image = x[None, :, :, :]
    mask_1 = utils.get_mask_of_seg_rgb(x_seg, exact=1)
    mask_2 = utils.get_mask_of_seg_rgb(x_seg, exact=2)
    mask_3 = utils.get_mask_of_seg_rgb(x_seg, exact=3)
    mask_4 = utils.get_mask_of_seg_rgb(x_seg, exact=4)
    mask_t = utils.get_mask_of_seg_rgb(x_seg)
    pred = model.predict(image)
    prob = pred.max()
    pred = pred.argmax()
    a = analyzer.analyze(image)
    v1 = mask_value(a[0], image[0], mask_1)
    v2 = mask_value(a[0], image[0], mask_2)
    v3 = mask_value(a[0], image[0], mask_3)
    v4 = mask_value(a[0], image[0], mask_4)
    vt = mask_value(a[0], image[0], mask_t)
    data = {"1": v1, "2": v2, "3": v3, "4": v4, "total": vt}
    print(f"Pred: {pred}, prob: {prob:.4f}")
    print(
        f"1: {v1:.4f}, 2: {v2:.4f}, 3: {v3:.4f}, 4: {v4:.4f}, total: {vt:.4f}")
    plot_image_analysis_mask(image[0], x_seg, process_analysis_image(a[0]),
                             data, figsize)


def evaluate_masks(model, analyzer, x, x_seg, y):
    mask_1 = [utils.get_mask_of_seg_rgb(x, threshold=1, exact=True) for x in x_seg]
    mask_2 = [utils.get_mask_of_seg_rgb(x, threshold=2, exact=True) for x in x_seg]
    mask_3 = [utils.get_mask_of_seg_rgb(x, threshold=3, exact=True) for x in x_seg]
    mask_4 = [utils.get_mask_of_seg_rgb(x, threshold=4, exact=True) for x in x_seg]
    mask_t = [utils.get_mask_of_seg_rgb(x) for x in x_seg]
    prediction = model.predict_on_batch(x)
    analysis = analyzer.analyze(x)

    y_hat = [x.argmax() for x in prediction]
    y_t = [mask_value(a, i, m) for a, i, m in zip(analysis, x, mask_t)]
    y_1 = [mask_value(a, i, m) for a, i, m in zip(analysis, x, mask_1)]
    y_2 = [mask_value(a, i, m) for a, i, m in zip(analysis, x, mask_2)]
    y_3 = [mask_value(a, i, m) for a, i, m in zip(analysis, x, mask_3)]
    y_4 = [mask_value(a, i, m) for a, i, m in zip(analysis, x, mask_4)]

    return zip(y, y_hat, y_1, y_2, y_3, y_4, y_t)


def plot_image_analysis_mask(image, mask, analysis, data, figsize=(18, 5)):
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    axes[0].imshow(image)
    axes[0].axis('off')
    mask = mask.reshape((mask.shape[0], mask.shape[1]))
    axes[1].imshow(mask, cmap='gray', vmin=0, vmax=4)
    axes[1].axis('off')

    axes[2].imshow(analysis, cmap='seismic', clim=(-1, 1))
    axes[2].axis('off')

    names = list(data.keys())
    values = list(data.values())
    axes[3].bar(names, values)

    plt.tight_layout()
    plt.show()
