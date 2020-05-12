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
        tumor_val = heatmap[mask].clip(0).sum() / mask.sum()
        brain_val = heatmap[brain_mask].clip(0).sum() / brain_mask.sum()
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
    pred_analysis = [mask_value(a,
                                i,
                                utils.get_mask_of_seg_rgb(s, tumor_region, exact=tumor_region > 0))
                     for a, i, s in zip(analysis, images, segmentation)]
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
    :return: list of tuples
    (true value, model prediction, prediction probability, interpretation score)
    """
    prediction = model.predict_on_batch(images)
    analysis = evaluate_analyzer(analyzer, images, segmentation, tumor_region)
    pred = [x.argmax() for x in prediction]
    prob = [x.max() for x in prediction]
    return zip(y.astype('int16'), pred, prob, analysis)


def evaluate_method_generator(model, analyzer, generator,
                              num_of_data=1, batch_size=1, tumor_region=0):
    """
    Evaluate interpretation with generator
    :param model: neural network model
    :param analyzer: interpretation technique model
    :param generator: generator of images and segmentation
    :param tumor_region: number of tumor region (0,1,2,3,4)
    :param num_of_data: number of images
    :param batch_size: batch size
    :return: list of tuples (true value, model prediction, interpretation score)
    """
    num_of_iteration = num_of_data // batch_size
    final_iterator = iter([])
    for i, ((x, y), (x_seg, y_seg)) in enumerate(generator):
        if i >= num_of_iteration:
            break
        final_iterator = chain(
            final_iterator,
            evaluate_method(model, analyzer, x, x_seg, y, tumor_region)
        )
    return final_iterator


def visualize_method(model, analyzer, x, x_seg, y):
    metrics = evaluate_method(model, analyzer, x, x_seg, y)
    analisis_imgs = analyzer.analyze(x)
    titles = [
        ('Label: {}     '.format(y), 'Pred:  {}     '.format(y_h),
         'Prob:  {:.2f}'.format(p), 'Score: {:.2f}'.format(y_a))
        for y, y_h, p, y_a in metrics]

    for img, seg, anlz, title in zip(x, x_seg, analisis_imgs, titles):
        plot_image_analysis([('rgb', img),
                             ('mask', seg),
                             ('heatmap', anlz)],
                            title)

def visualize_method_by_regions(model, analyzer, x, x_seg, y):
    metrics1 = evaluate_method(model, analyzer, x, x_seg, y, 1)
    metrics2 = evaluate_method(model, analyzer, x, x_seg, y, 2)
    metrics4 = evaluate_method(model, analyzer, x, x_seg, y, 4)
    metricst = evaluate_method(model, analyzer, x, x_seg, y, 0)
    analisis_imgs = analyzer.analyze(x)
    metricst = list(metricst)
    titles = [
        ('Label: {}     '.format(y), 'Pred:  {}     '.format(y_h),
         'Prob:  {:.2f}'.format(p), 'Score: {:.2f}'.format(y_a))
        for y, y_h, p, y_a in metricst]
    histogram_data = [
        {
            'reg_1': m1[3], 'reg_2': m2[3], 'reg_4': m4[3], 'total': mt[3]
        }
        for m1, m2, m4, mt in zip(metrics1, metrics2, metrics4, metricst)]

    print(histogram_data)
    for img, seg, anlz, hist, title in zip(x, x_seg, analisis_imgs, histogram_data, titles):
        plot_image_analysis([('rgb', img),
                             ('mask', seg),
                             ('heatmap', anlz),
                             ('hist', hist)],
                            title)

def plot_image_analysis(graphs, title=None, figsize=(10, 2)):
    """
    Plot multiple plot format in one line and title on left.
    :param graphs: list of tuples, where tuple is (type, data).
    Type is one of [rgb, mask, heatmap, hist],
    Data are ether image or dictionary for 'hist' type.
    :param title: list of string to by printed on left
    :param figsize: size of figure
    """

    fig, axes = plt.subplots(1, len(graphs), figsize=figsize)
    if title is not None:
        txt_left = [l + '\n' for l in title]
        axes[0].set_ylabel(
            ''.join(txt_left),
            rotation=0, verticalalignment='center', horizontalalignment='right'
        )
        axes[0].set_xticks([])
        axes[0].set_yticks([])

    for ax, (type, data) in zip(axes, graphs):

        if type is 'rgb':
            ax.imshow(process_brain_image(data))
        elif type is 'mask':
            mask = data.reshape((data.shape[0], data.shape[1]))
            ax.imshow(mask, cmap='gray', vmin=0, vmax=4)
            ax.axis('off')
        elif type is 'heatmap':
            ax.imshow(process_analysis_image(data), cmap='seismic', clim=(-1, 1))
            ax.axis('off')
        elif type is 'hist':
            names = list(data.keys())
            values = list(data.values())
            ax.bar(names, values)
    plt.tight_layout()
    plt.show()

def process_brain_image(image):
    image = image + np.abs(image.min())
    image = image / image.max()
    return image

def process_analysis_image(img):
    img = img.sum(axis=np.argmax(np.asarray(img.shape) == 3))
    img /= np.max(np.abs(img))
    return img