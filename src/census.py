import os
import matplotlib.pyplot as plt
import numpy as np
#from sklearn.metrics import r2_score
from keras.models import Model
from keras import layers
from keras.optimizers import Adam
from keras import regularizers
from keras.callbacks import ModelCheckpoint
import pydot
from keras.utils import plot_model
from keras import backend as K
from util import load_image, load_census_mask, normalize_image, create_mask, my_jaccard, \
        save_data, load_data, plot_compare_masks, blur_mask

'''
Global constants
'''
square_km_per_pixel = 0.9
data_dir = '/media/sf_school/project/data'

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
    for i in range(0, H - gh, gh):
        for j in range(0, W - gw, gw):
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

'''
Models
'''
def create_simple_cnn():
    '''
    Simple sequential CNN to solve the regression problem of predicting population
    '''
    #K.image_data_format() returns channel order from config file
    inputs = layers.Input(shape=(64, 64, 6))
    conv11 = layers.Conv2D(32, 3, strides=1, activation='relu', padding='same', kernel_regularizer=regularizers.l2(.01))(inputs)
    conv12 = layers.Conv2D(32, 3, strides=1, activation='relu', padding='same', kernel_regularizer=regularizers.l2(.01))(conv11)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv12)

    conv21 = layers.Conv2D(64, 3, strides=1, activation='relu', padding='same', kernel_regularizer=regularizers.l2(.01))(pool1)
    conv22 = layers.Conv2D(64, 3, strides=1, activation='relu', padding='same', kernel_regularizer=regularizers.l2(.01))(conv21)
    up1 = layers.UpSampling2D(size=(2, 2))(conv22)

    conv31 = layers.Conv2D(32, 3, strides=1, activation='relu', padding='same', kernel_regularizer=regularizers.l2(.01))(up1)
    conv32 = layers.Conv2D(32, 3, strides=1, activation='relu', padding='same', kernel_regularizer=regularizers.l2(.01))(conv31)

    final = layers.Conv2D(1, 1, strides=1, activation='linear')(conv32)
    model = Model(inputs=inputs, outputs=final)
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    return model

def create_simple_unet():
    '''
    Create a unet shaped CNN to solve the regression problem of predicting population
    '''
    inputs = layers.Input(shape=(64, 64, 6))
    conv11 = layers.Conv2D(32, 3, strides=1, activation='relu', padding='same', kernel_regularizer=regularizers.l2(.01))(inputs)
    conv12 = layers.Conv2D(32, 3, strides=1, activation='relu', padding='same', kernel_regularizer=regularizers.l2(.01))(conv11)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv12)

    conv21 = layers.Conv2D(64, 3, strides=1, activation='relu', padding='same', kernel_regularizer=regularizers.l2(.01))(pool1)
    conv22 = layers.Conv2D(64, 3, strides=1, activation='relu', padding='same', kernel_regularizer=regularizers.l2(.01))(conv21)
    up2 = layers.UpSampling2D(size=(2, 2))(conv22)

    merge3 = layers.Concatenate(axis=-1)([up2, conv12])
    conv31 = layers.Conv2D(32, 3, strides=1, activation='relu', padding='same', kernel_regularizer=regularizers.l2(.01))(merge3)
    conv32 = layers.Conv2D(32, 3, strides=1, activation='relu', padding='same', kernel_regularizer=regularizers.l2(.01))(conv31)

    final = layers.Conv2D(1, 1, strides=1, activation='linear', kernel_regularizer=regularizers.l2(.01))(conv32)
    model = Model(inputs=inputs, outputs=final)
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    return model


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
Main
'''
def main():
    # settings
    do_training = True
    do_prediction = True
    n_iters = 5
    data_set_size = 200
    ids = ['ohio_01']
    blur_size = 15
    # if loading or saving already prepped data
    experiment_name = 'census_dev01'
    use_prepped_data = True
    save_prepped_data = False
    X_filename = '{}_X_{}.npy'.format(experiment_name, data_set_size)
    Y_filename = '{}_Y_{}.npy'.format(experiment_name, data_set_size)
    # load images and their masks
    if use_prepped_data:
        X = load_data(X_filename)
        Y = load_data(Y_filename)
    else:
        images = []
        masks = []
        for i in ids:
            image = load_image(i, 'landsat', 'all')
            # TODO: is this making the mistake of normalizing the dev and test data too early?
            images.append(normalize_image(image))
            mask = load_census_mask(i)
            # blur mask?
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
        if save_prepped_data:
            save_data(X, X_filename)
            save_data(Y, Y_filename)
    print(X.shape, np.min(X), np.max(X), np.mean(X), X.dtype)
    print(Y.shape, np.min(Y), np.max(Y), np.mean(Y), Y.dtype)
    model = create_simple_cnn()
    if do_training:
        model.summary()
        plot_model(model, to_file='model-sequential.png', show_shapes=True)
        model.fit(X, Y, batch_size=10, epochs=n_iters, verbose=2, validation_split=0.2)
    if do_prediction:
        Y_pred = model.predict(X)
        metrics = population_prediction_metrics(Y, Y_pred)
        print(metrics)
        best_indices = best_predictions(Y, Y_pred)
        plot_compare_masks(X[best_indices[:5]] ,Y[best_indices[:5]], Y_pred[best_indices[:5]])

if __name__=='__main__':
    main()

