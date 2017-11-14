import os
import random
import numpy as np
import pandas as pd
import tifffile as tiff
from shapely import wkt
from shapely.geometry import MultiPolygon, Polygon
from cv2 import fillPoly
from datetime import datetime
from scipy.misc import imsave
import matplotlib.pyplot as plt

'''
Global constants
'''
data_directory = '/media/sf_school/project/data/'
num_class = 10
wkt_shapes_fn = os.path.join(data_directory, 'train_wkt_v4.csv')
grid_sizes_fn = os.path.join(data_directory, 'grid_sizes.csv')

'''
Load/Save
'''
def load_image(image_id, image_type=None, bands='all'):
    '''
    Load an image as a numpy ndarray
    image_type:
        -P
        -M
        -A
        -None = load the three-band RGB image
    bands:
        -all = load all bands for that type
        -list of bands to load, zero-indexed (i.e. [1, 3, 4])
    Returns an numpy array with dimensions [height, width, bands]
    '''
    if image_type == None:
        image_folder = 'three_band'
        image_name = '{}.tif'.format(image_id)
    elif image_type in ['P', 'M', 'A']:
        image_folder = 'sixteen_band/'
        image_name = '{}_{}.tif'.format(image_id, image_type)
    else:
        raise Exception('Incorrect image type: {}'.format(image_type))
    filename = os.path.join(data_directory, image_folder, image_name)
    image = tiff.imread(filename)
    # put image in shape (H x W x bands)
    image = np.rollaxis(image, 0, 3)
    if bands != 'all':
        bands = np.array(bands)
        image = image[..., bands]
    return image

def save_mask(mask, image_id, cls, predicted=False):
    '''
    Save a true or predicted mask as an image
    mask: numpy array of type uint8
    image_id: image id the mask is associated with
    cls: the class that the mask represents
    predicted: (bool) was the mask predicted or the true label
    '''
    if predicted:
        folder_name = 'masks_pred'
    else:
        folder_name = 'masks_true'
    time = datetime.now().strftime('%Y%m%d-%H%M')
    mask_name = '{}_mask_{}_{}.png'.format(image_id, cls, time)
    filename = os.path.join(data_directory, folder_name, mask_name)
    imsave(filename, mask)

'''
Preprocessing
'''
def normalize_image(image):
    '''
    Normalize the image
    '''
    # TODO: what type of normalization should we do for the models we plan to build?
    # color? contrast? mean-normalized? values in what range?
    normalized = np.zeros_like(image, dtype=np.float64)
    bands = image.shape[2]
    for i in range(bands):
        a = np.min(image[..., i])
        b = np.max(image[..., i])
        normalized[..., i] = (image[..., i] - a) / (b - a)
    return normalized

def load_grid_sizes(image_id):
    '''
    returns the xmax and ymin to use to scale the wkt shapes
    '''
    grid_sizes_df = pd.read_csv(grid_sizes_fn,
            names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
    grid_sizes = grid_sizes_df[grid_sizes_df.ImageId == image_id]
    xmax, ymin = grid_sizes.values[0][1:]
    return xmax, ymin

def load_wkt_shape(image_id, classes='all'):
    '''
    Load WKT shape as a list of Shapely geometric object
    classes:
        -all = load all classes
        -list of the classes to load, zero-indexed (i.e. [0, 3, 5])
    '''
    shapes = []
    shapes_df = pd.read_csv(wkt_shapes_fn,
            names=['ImageId', 'ClassType', 'MultipolygonWKT'], skiprows=1)
    image_shapes = shapes_df[shapes_df.ImageId == image_id]
    if classes == 'all':
        classes = list(range(num_class))
    for cls in classes:
        if cls not in range(num_class):
            raise Exception('Class {} is not one of {} valid classes.'.format(cls + 1, num_class))
        # wkt_shape can be an empty multipolygon
        wkt_shape = image_shapes[image_shapes.ClassType == cls + 1].MultipolygonWKT.values[0]
        shapes.append(wkt.loads(wkt_shape))
    return shapes

def convert_shape_to_coords(shape, height, width, xmax, ymin):
    '''
    Returns a list of scalled interior and exterior coordinates from a given shape
    '''
    def scale_coords(coords):
        #TODO: Scale image with equations from kaggle description
        scale_y = height / ymin
        scale_x = width / xmax
        scale = np.array([scale_x, scale_y])
        coords *= scale
        coords = np.round(coords).astype(np.int)
        return coords

    exteriors = []
    interiors = []
    for i, polygon in enumerate(shape):
        exterior = np.array(polygon.exterior.coords)
        scaled_ext = scale_coords(exterior)
        exteriors.append(scaled_ext)
        for polygon_interior in polygon.interiors:
            interior = np.array(polygon_interior.coords)
            scaled_int = scale_coords(interior)
            interiors.append(scaled_int)
    return exteriors, interiors


def generate_mask_from_coords(height, width, exteriors, interiors):
    '''
    Creates a numpy array bitmask from the specified coordinates
    '''
    mask = np.zeros((height, width), np.uint8)
    fillPoly(mask, exteriors, 1)
    fillPoly(mask, interiors, 0)
    return mask

def create_mask(image_id, height, width, classes='all'):
    '''
    Load the masks for an image
    height, width: number of pixels
    classes:
        -all = load all classes
        -list of the classes to load, zero-indexed (i.e. [0, 3, 5])
    Returns a numpy array of masks with dimensions [height, width, classes]
    '''
    # Based on functions by author: visoft
    # Link: https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    masks = []
    xmax, ymin = load_grid_sizes(image_id)
    shapes = load_wkt_shape(image_id, classes)
    for shape in shapes:
        coords = convert_shape_to_coords(shape, height, width, xmax, ymin)
        masks.append(generate_mask_from_coords(height, width, coords[0], coords[1]))
    masks = np.rollaxis(np.array(masks), 0, 3)
    # remove extra dimensions if only one class was requested
    masks = np.squeeze(masks)
    return masks

'''
Scoring functions
'''
def my_jaccard(true, pred):
    '''
    Calculate the union over intersection of the true and predicted mask
    true, pred = binary masks of the same size representing the actual and predicted mask
    '''
    true = true.flatten()
    pred = np.rint(pred.flatten())
    intersection = np.sum(true * pred)
    union = true + pred
    union[union > 1] = 1
    union = np.sum(union)
    return intersection/union

'''
Visualizations
'''
def plot_compare_masks(true, pred):
    '''
    Plot the true and predicted masks side by side for comparison
    true, pred: lists of masks that represent the true and predicted labels
    '''
    # TODO: consider using matplotlib.gridspec to fix plot spacing
    assert len(true) == len(pred)
    fig, cls_plts = plt.subplots(len(true), 2)
    if len(true) == 1:
        cls_plts = [cls_plts]
    for i, cls_plt in enumerate(cls_plts):
        cls_plt[0].imshow(true[i])
        #cls_plt[0].set_title('Class {} True Mask'.format(i + 1))
        cls_plt[0].axis('off')
        cls_plt[1].imshow(pred[i])
        #cls_plt[1].set_title('Class {} Prediction'.format(i + 1))
        cls_plt[1].axis('off')
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.show()


def plot_image(image):
    '''
    Plot a grayscale or RGB image
    '''
    pass

