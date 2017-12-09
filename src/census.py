import os
import math
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from util import *
from models import *
from scipy.misc import imsave

'''
Global constants
'''
square_km_per_pixel = 0.9
weights_dir = '/media/sf_school/project/weights/'

'''
Preprocessing
'''
def create_sub_image_grid(image, mask, grid_size=100):
    '''
    Split an image into sub images based on grid_size
    '''
    H, W, bands = image.shape
    gh, gw = grid_size, grid_size
    images = []
    masks = []
    for i in range(0, H - gh + 1, gh):
        for j in range(0, W - gw + 1, gw):
            sub_image = image[i:i + gh, j:j + gw, :].reshape(1, gh, gw, bands)
            sub_mask = mask[i:i + gh, j:j + gw].reshape(1, gh, gw)
            images.append(sub_image)
            masks.append(sub_mask)
    # normalize each sub image or just the entire image?
    images = np.concatenate(images)
    masks = np.concatenate(masks)
    return images, masks

def calculate_census_weights(Y):
    '''
    Return vector of weights same length as number of sub_images, each
    weight corresponds to the percent of population within that sub_image
    '''
    weights = np.sum(Y * square_km_per_pixel, axis=(1, 2))
    weights = weights / np.sum(weights)
    return weights

def prep_data(ids, data_set_size, blur_size):
    images = []
    masks = []
    for i in ids:
        image = load_image(i, 'landsat', 'all')
        images.append(image)
        mask = load_census_mask(i)
        mask = blur_mask(mask, blur_size)
        if mask.shape != image.shape:
            h, w = image.shape[:2]
            mask = mask[:h, :w]
        masks.append(mask)
    print(images[0].shape, np.min(images[0]), np.max(images[0]))
    print(masks[0].shape, np.min(masks[0]), np.max(masks[0]))
    # create X and Y composed of [#sub_images, height, width, bands]
    X, Y = None, None
    for image, mask in zip(images, masks):
        sub_x, sub_y = create_sub_image_grid(image, mask, grid_size=64)
        if X is None or Y is None:
            X = sub_x
            Y = sub_y
        else:
            X = np.concatenate((X, sub_x))
            Y = np.concatenate((Y, sub_y))
    print(X.shape, np.min(X), np.max(X), np.mean(X), X.dtype)
    print(Y.shape, np.min(Y), np.max(Y), np.mean(Y), Y.dtype)
    # calculate weights of sub_images and randomly sample them to build train/cv set
    weights = calculate_census_weights(Y)
    N, *_ = weights.shape
    selection = np.random.choice(N, size=data_set_size, replace=False, p=weights)
    X = X[selection]
    Y = np.expand_dims(Y[selection], axis=-1)
    return X, Y

'''
Scoring
'''
def population_prediction_metrics(y_true, y_pred):
    '''
    Return useful metrics for judging population prediction accuracy
    '''
    assert y_true.shape == y_pred.shape
    return (np.mean(y_true), np.mean(y_pred), np.mean(y_true)-np.mean(y_pred))

def best_predictions(y_true, y_pred):
    '''
    Return indices sorted by best predictions
    '''
    true_pops = np.log(np.sum(y_true * square_km_per_pixel, axis=(1, 2)))
    pred_pops = np.log(np.sum(y_pred * square_km_per_pixel, axis=(1, 2)))
    return np.argsort(np.absolute(true_pops - pred_pops).squeeze())

'''
Experiment
'''
def experiment(experiment_name, model, n_iters=5):
    print(experiment_name)
    # settings
    do_training = False
    do_prediction = True
    ids = ['ohio_01']
    blur_size = 15
    train_set_size = 1000
    test_set_size = 100
    validation_split = 0.1
    # if loading or saving already prepped data
    use_prepped_data = True
    data_set_size = train_set_size + test_set_size
    X_filename = 'X_{}.npy'.format(data_set_size)
    Y_filename = 'Y_{}.npy'.format(data_set_size)
    weights_best_filename = '{}_{}_best.hdf5'.format(experiment_name, data_set_size)
    weights_best_filename = os.path.join(weights_dir, weights_best_filename)
    weights_filename = '{}_{}.hdf5'.format(experiment_name, data_set_size)
    weights_filename = os.path.join(weights_dir, weights_filename)
    # load images and their masks
    if use_prepped_data:
        X = load_data(X_filename)
        Y = load_data(Y_filename)
    else:
        X, Y = prep_data(ids, data_set_size, blur_size)
        save_data(X, X_filename)
        save_data(Y, Y_filename)
    # make train, dev, test and normalize
    X_test = X[-test_set_size:]
    Y_test = Y[-test_set_size:]
    X = X[:train_set_size]
    Y = Y[:train_set_size]
    mean = np.mean(X[:-1 * int(train_set_size * validation_split)], axis=(0, 1, 2))
    std = np.std(X[:-1 * int(train_set_size * validation_split)], axis=(0, 1, 2))
    X = (X - mean) / std
    X_test = (X_test - mean) / std
    print(X.shape, np.min(X), np.max(X), np.mean(X), X.dtype)
    print(Y.shape, np.min(Y), np.max(Y), np.mean(Y), Y.dtype)
    print(X_test.shape, np.min(X_test), np.max(X_test), np.mean(X_test), X_test.dtype)
    print(Y_test.shape, np.min(Y_test), np.max(Y_test), np.mean(Y_test), Y_test.dtype)
    # load the model
    model.summary()
    plot_model(model, to_file='model-{}.png'.format(experiment_name), show_shapes=True)
    if do_training:
        checkpoint = ModelCheckpoint(weights_best_filename, monitor='loss', save_best_only=True)
        model.fit(X, Y, batch_size=10, epochs=n_iters, verbose=2, validation_split=0.2,
                callbacks=[checkpoint])
        model.save_weights(weights_filename)
    else:
        # load weights
        model.load_weights(weights_filename)
    if do_prediction:
        Y_pred = model.predict(X_test)
        metrics = population_prediction_metrics(Y_test, Y_pred)
        print(metrics)
        best_indices = best_predictions(Y_test, Y_pred)
        plot_compare_masks(X_test[best_indices[:10]] ,Y[best_indices[:10]], Y_pred[best_indices[:10]])

def save_examples(y=0, x=0, height=512, width=512):
    '''
    height and width must be evenly divisible by the grid_size the NN was trained on (64)
    '''
    # load image
    image_id = 'ohio_01'
    blur_size = 15
    gs = 64 # grid_size = 64
    weights_filename = os.path.join(weights_dir, 'census_deep_unet_1100_best.hdf5')
    image = load_image(image_id, 'landsat', 'all')
    image = image[y:y+height, x:x+width]
    # scale and save image
    smallest = np.min(image[..., :3])
    display_image = image - smallest
    scale = np.max(display_image) / 255
    display_image = display_image / (scale + 1) / 2
    display_image = display_image.astype(int)
    print(image[..., 0].min(), image[..., 0].max(), image[..., 0].mean())
    print(image[..., 1].min(), image[..., 1].max(), image[..., 1].mean())
    print(image[..., 2].min(), image[..., 2].max(), image[..., 2].mean())
    imsave('image.png', display_image[..., :3])
    image = normalize_image(image)
    mask = load_census_mask(image_id)
    mask = blur_mask(mask, blur_size)
    if mask.shape != image.shape:
        h, w = image.shape[:2]
        mask = mask[y:y+height, x:x+width]
    # save mask
    imsave('mask.png', mask)
    # make the data from the image
    X, Y = create_sub_image_grid(image, mask, grid_size=64)
    # use all the data and do predictions
    model = create_deep_unet()
    model.load_weights(weights_filename)
    Y_pred = model.predict(X)
    Y_pred = Y_pred.squeeze()
    print(Y_pred.shape)
    # score things?
    # stich and save prediction
    prediction = np.zeros((height, width))
    for i, h in enumerate(range(0, height - gs + 1, gs)):
        for j, w in enumerate(range(0, width - gs + 1, gs)):
            prediction[h:h+gs, w:w+gs] = Y_pred[(i * width // gs) + j]
    # save prediction
    imsave('prediction.png', prediction)
    

if __name__=='__main__':
    save_examples(3600, 1000, 1024, 1024)
    '''
    experiment_name = 'census_simple_cnn'
    model = create_simple_cnn()
    experiment(experiment_name, model, 30)
    experiment_name = 'census_simple_unet'
    model = create_simple_unet()
    experiment(experiment_name, model, 30)
    experiment_name = 'census_deep_cnn'
    model = create_deep_cnn()
    experiment(experiment_name, model, 30)
    experiment_name = 'census_deep_unet'
    model = create_deep_unet()
    experiment(experiment_name, model, 30)
    '''
    K.clear_session()

