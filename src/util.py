import os
import numpy as np
import tifffile as tiff
import cv2
from datetime import datetime
from scipy.misc import imsave
import matplotlib.pyplot as plt

'''
Global constants
'''
data_directory = '/media/sf_school/project/data/'
model_dir = '/media/sf_school/project/models/'

'''
Load/Save
'''
def load_image(image_id, image_type, bands='all'):
    image_name = image_id + '_{}.tif'
    if bands == 'all':
        bands = ['B' + str(i) for i in list(range(1, 6)) + [7]]
    image_folder = 'landsat/'
    image = []
    for band in bands:
        filename = os.path.join(data_directory, image_folder, image_name.format(band))
        image.append(tiff.imread(filename))
    image = np.array(image)
    image = np.rollaxis(image, 0, 3)
    return image

def load_census_mask(image_id):
    mask_name = image_id + '.tif'
    mask_folder = 'masks_census/'
    filename = os.path.join(data_directory, mask_folder, mask_name)
    mask = tiff.imread(filename)
    return mask

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

def save_data(data, filename):
    '''
    Save the data to the data_dir/prepped/ folder with the given filename
    '''
    filename = os.path.join(data_directory, 'prepped/', filename)
    np.save(filename, data)

def load_data(filename):
    '''
    Load data from the data_dir/prepped/ folder with the given filename
    '''
    filename = os.path.join(data_directory, 'prepped/', filename)
    data = np.load(filename)
    return data

'''
Preprocessing
'''
def normalize_image(image):
    '''
    Normalize the image using mean and standard deviation
    '''
    normalized = np.zeros_like(image, dtype=np.float32)
    bands = image.shape[-1]
    for i in range(bands):
        normalized[..., i] = (image[..., i] - np.mean(image[..., i])) / np.std(image[..., i])
    return normalized

def blur_mask(image, size=5):
    kernel_shape = (size, size)
    print(np.sum(image))
    blurred = cv2.GaussianBlur(image, kernel_shape, 0)
    print(np.sum(blurred))
    return blurred

'''
Visualizations
'''
def plot_compare_census(x, y_true, y_pred):
    '''
    Plot the true and predicted population density side by side for comparison
    true, pred: lists of masks that represent the true and predicted population
    '''
    # TODO: consider using matplotlib.gridspec to fix plot spacing
    assert len(y_true) == len(y_pred) and len(x) == len(y_true)
    fig, plts = plt.subplots(len(x), 3)
    if len(y_true) == 1:
        plts = [plts]
    for i, i_plt in enumerate(plts):
        a = np.min(x[i, ..., :3])
        b = np.max(x[i, ..., :3])
        i_plt[0].imshow((x[i, ..., :3] - a) / (b - a))
        i_plt[0].axis('off')
        i_plt[1].imshow(y_true[i].squeeze())
        i_plt[1].axis('off')
        i_plt[2].imshow(y_pred[i].squeeze())
        i_plt[2].axis('off')
    #plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.show()

