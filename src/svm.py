import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import csv
import random
from sklearn import svm, linear_model
from itertools import combinations
from util import *

# TODO:
# - expand to all bands
# - add regularization
# - improve models, move to other file
# - fix the way that images are noremalized
# - fix the way train/cv/test sets are created
# - make pretty image showing output/input

model_dir = '/media/sf_school/project/models/'
results_dir = '/media/sf_school/project/results/'
data_dir = '/media/sf_school/project/data/'

'''
Preprocessing
'''
def create_sub_image_grid(image, mask, grid_size=100):
    '''
    Split an image into sub images based on grid_size
    '''
    H, W, features = image.shape
    gh, gw = grid_size, grid_size
    images = []
    masks = []
    for i in range(0, H - gh + 1, gh):
        for j in range(0, W - gw + 1, gw):
            sub_image = image[i:i + gh, j:j + gw, :].reshape(1, gh, gw, features)
            sub_mask = mask[i:i + gh, j:j + gw].reshape(1, gh, gw)
            images.append(sub_image)
            masks.append(sub_mask)
    images = np.concatenate(images)
    masks = np.concatenate(masks)
    return images, masks

def prep_data(ids, data_set_size, kernel_size):
    images = []
    masks = []
    image_type = 'M'
    classes = [0, 1, 2, 3, 4, 5, 6, 7]
    for i in ids:
        image = load_image(i, image_type)
        # add features (right features and kernel_size?)
        image = np.dstack((image,
                           get_laplacian(image, kernel_size),
                           get_gaussian(image, kernel_size)))
        images.append(image)
        H, W, _ = image.shape
        mask = create_mask(i, H, W, classes)
        masks.append(mask)
    # create X and Y composed of [#sub_images, height, width, features/classes]
    X, Y = None, None
    for image, mask in zip(images, masks):
        sub_x, sub_y = create_sub_image_grid(image, mask, grid_size=64)
        if X is None or Y is None:
            X = sub_x
            Y = sub_y
        else:
            X = np.concatenate((X, sub_x))
            Y = np.concatenate((Y, sub_y))
    # don't normalize yet
    print(X.shape, np.min(X), np.max(X), np.mean(X), np.std(X), X.dtype)
    print(Y.shape, Y.dtype)
    # TODO: weight the sub_images?
    N, *_ = Y.shape
    selection = np.random.choice(N, size=data_set_size, replace=False)
    X = X[selection]
    Y = Y[selection]
    return X, Y

def save_results(results):
    '''
    Writes the results of the experiment to a csv file in the results_dir
    Input:
    results - a dictionary with keys that are tuples of (model_name, features, size)
              and values that are the jaccard score
    '''
    dt = datetime.now().strftime('%Y%m%d-%H%M')
    results_filename = 'ex1_results_{}.csv'.format(dt)
    with open(os.path.join(results_dir, results_filename), 'w', newline='') as results_file:
        results_writer = csv.writer(results_file, delimiter=';', quotechar='|',
                                    quoting=csv.QUOTE_MINIMAL)
        results_writer.writerow(['model', 'features', 'size', 'sample', 'jaccard_score'])
        for key, jaccard_score in results.items():
            model, features, size, sample = key
            results_writer.writerow([model, features, size, sample, jaccard_score])

'''
Features
'''
#cv2.getGaborKernel(ksize, sigma, theta, lambd, gamma[, psi[, ktype]])

def get_laplacian(image, size=3):
    # check if grayscale (only 1 band)
    if len(image.shape) == 2:
        H, W = image.shape
        image = image.reshape(H, W, 1)
        bands = 1
    else:
        bands = image.shape[-1]
    features = np.zeros_like(image)
    for i in range(bands):
        features[..., i] = cv2.Laplacian(image[..., i], ddepth=5, ksize=size) #5?
    return features

def get_gaussian(image, size=3):
    kernel_shape = (size, size)
    # check if grayscale (only 1 band)
    if len(image.shape) == 2:
        H, W = image.shape
        image = image.reshape(H, W, 1)
    features = np.zeros_like(image)
    for i in range(image.shape[2]):
        features[..., i] = cv2.GaussianBlur(image[..., i], kernel_shape, 0)
    return features

'''
Models
'''
def create_svm_model():
    '''
    dual=False if #samples > #features
    C=1.0 is default penalty for errors
    default for multi-class is one vs all
    TODO: attempt class_weight='balanced' at some point
    '''
    return svm.LinearSVC(penalty='l2', dual=False, C=1.0, class_weight='balanced')

def create_logistic_model():
    '''
    dual=False if #samples > #features
    C=1.0 is default penalty for errors
    default for multi-class is one vs all
    TODO: attempt class_weight='balanced' at some point
    '''
    return linear_model.LogisticRegression(penalty='l2', dual=False, C=1.0, class_weight='balanced')

'''
Metrics
'''
def compare_class_scores(true, pred):
    classes = np.max(true)
    for i in range(classes + 1):
        t = np.sum(true == i)
        p = np.sum(pred == i)
        union = np.sum(np.logical_and(true == pred, true == i))
        intersection = np.sum(np.logical_or(true == i, pred == i))
        if t != 0:
            a = union / t
        else:
            a = 0.0
        if i != 0 and intersection != 0:
            j = union / intersection
        else:
            j = 0.0
        print("Class: {},\t True: {},\t Pred: {},\t Acc: {},\t Jac: {}".format(i, t, p, a, j))

'''
Experiments
'''
def create_cv(X, Y, k):
    # attempt at writing a function to generate cv data sets
    # TODO: test this
    N, H, W, D = X.shape
    validation_size = N // k
    train_size = N - validation_size
    X_splits = np.asarray(np.array_split(X, k))
    Y_splits = np.asarray(np.array_split(Y, k))
    X_train = np.zeros((k, train_size, H, W, D))
    Y_train = np.zeros((k, train_size, H, W), dtype=np.int)
    X_validate = np.zeros((k, validation_size, H, W, D))
    Y_validate = np.zeros((k, validation_size, H, W), dtype=np.int)
    for i in range(k):
        X_train[i] = X[np.arange(k) != i].reshape(train_size, H, W, D)
        X_validate[i] = X[i]
        Y_train[i] = Y[np.arange(k) != i].reshape(train_size, H, W)
        Y_validate[i] = Y[i]

def old_experiment(X, Y,
               models=('svm',),
               sizes=[10],
               #sizes=[10, 25, 50, 100, 200, 500, 1000], #have 1600 in total right now
               features=(get_laplacian, get_gaussian),
               k=5):
    '''
    Do experiments for all combinations of settings
    Input:
    X - numpy array of the images of shape (num_subimages, H, W, D)
    '''
    # data is list of subimages, each 100x100
    # labels are corresponding bit masks
    # create tuple of feature combinations to test
    feature_combinations = [None] #[None, (features[0],), (features[1],), features]
    for size in sizes:
        results = dict()
        # randomly order subimages
        order = np.random.permutation(X.shape[0])
        X = X[order]
        Y = Y[order]
        print('##### Size: {} #####'.format(size))
        # select first size subimages
        samples = X[:size]
        labels = Y[:size].reshape(-1)
        N, H, W, D = samples.shape
        M = labels.shape[0]
        # split labels into train and test
        test_labels = labels[:M//k]
        train_labels = labels[M//k:]
        assert M == N * H * W
        for feature_combo in feature_combinations:
            iter_samples = np.copy(samples)
            if feature_combo is not None:
                feature_samples = [iter_samples]
                for feature_function in feature_combo:
                    # add features for each subimage
                    new_samples = np.zeros_like(samples)
                    for i in range(N): # for each image in 1...size
                        new_samples[i] = feature_function(samples[i])
                    feature_samples.append(new_samples)
                iter_samples = np.stack(feature_samples, axis=-1)
            iter_samples = iter_samples.reshape(N, H, W, -1)
            print(feature_combo)
            print(iter_samples.shape)
            # reshape samples to M x D [number of pixels, number of bands]
            D = iter_samples.shape[-1]
            iter_samples = iter_samples.reshape(-1, D)
            # split samples into train and test
            test_samples = iter_samples[:M//k]
            train_samples = iter_samples[M//k:]
            assert train_samples.shape[0] == train_labels.shape[0]
            assert test_samples.shape[0] == test_labels.shape[0]
            for model_name in models:
                model = None
                if model_name == 'svm':
                    model = svm.LinearSVC()
                elif model_name == 'logistic':
                    model = linear_model.LogisticRegression()
                else:
                    break
                # train model
                model.fit(train_samples, train_labels)
                # TODO: cross-validate training
                # predict mask and calculate scores
                train_prediction = model.predict(train_samples)
                train_score = my_jaccard(train_labels, train_prediction)
                test_prediction = model.predict(test_samples)
                # TODO: score breaks when the mask has zero of the class
                test_score = my_jaccard(test_labels, test_prediction)
                print('Model: {}\t Train: {}\t Test: {}'.format(model_name, train_score,
                                                                test_score))
                results[(model_name, feature_combo, size, 'train')] = train_score
                results[(model_name, feature_combo, size, 'test')] = test_score
                #if size > 50 and feature_combo is not None and len(feature_combo) == 2:
                    #save_model(model, model_name, size)
                print(train_samples.shape, train_labels.shape, train_prediction.shape)
                plot_compare_masks(train_samples, train_labels, train_prediction)
        print(results)
        #save_results(results)

def experiment(experiment_name, model_name, size, cv):
    print(experiment_name, model_name)
    # settings
    do_training = True
    do_prediction = True
    ids = ['6100_2_3', '6090_2_0', '6100_2_2', '6120_2_2',
           '6120_2_0', '6150_2_3', '6070_2_3', '6100_1_3',
           '6010_4_2', '6110_4_0', '6140_3_1', '6110_1_2',
           '6140_1_2', '6110_3_1', '6170_2_4', '6060_2_3']
    kernel_size = 15
    train_set_size = size
    test_set_size = 0
    validation_split = cv
    train_stop = int(train_set_size * (1 - validation_split))
    # if loading or saving already prepped data
    use_prepped_data = False
    data_set_size = train_set_size + test_set_size
    X_filename = 'feature_X_{}.npy'.format(data_set_size)
    Y_filename = 'feature_Y_{}.npy'.format(data_set_size)
    model_filename = 'feature_{}_{}.pkl'.format(model_name, data_set_size)
    # load images and their masks
    if use_prepped_data:
        X = load_data(X_filename)
        Y = load_data(Y_filename)
    else:
        X, Y = prep_data(ids, data_set_size, kernel_size)
        save_data(X, X_filename)
        save_data(Y, Y_filename)
    # make train, dev, test and normalize
    #X_test = X[-test_set_size:]
    #Y_test = Y[-test_set_size:]
    #X = X[:train_set_size]
    #Y = Y[:train_set_size]
    mean = np.mean(X[:train_stop], axis=(0, 1, 2))
    std = np.std(X[:train_stop], axis=(0, 1, 2))
    X = (X - mean) / std
    *_, features = X.shape
    #X_test = (X_test - mean) / std
    print(X.shape, np.min(X), np.max(X), np.mean(X), np.std(X), X.dtype)
    print(Y.shape, Y.dtype)
    #print(X_test.shape, np.min(X_test), np.max(X_test), np.mean(X_test), X_test.dtype)
    #print(Y_test.shape, np.min(Y_test), np.max(Y_test), np.mean(Y_test), Y_test.dtype)
    # load the model
    if do_training:
        if model_name == 'svm':
            model = create_svm_model()
        elif model_name == 'logistic':
            model = create_logistic_model()
        print(X[:train_stop].reshape(-1, features).shape)
        print(Y[:train_stop].flatten().shape)
        model.fit(X[:train_stop].reshape(-1, features),
                  Y[:train_stop].flatten())
        save_model(model, model_filename)
    else:
        model = load_model(model_filename)
    if do_prediction:
        X_test = X[train_stop:]
        Y_test = Y[train_stop:]
        Y_pred = model.predict(X_test.reshape(-1, features))
        compare_class_scores(Y_test.flatten(), Y_pred)
        # TODO: fix this function to visualize
        #plot_compare_masks(X[train_stop:], Y[train_stop:], Y_pred)

def baseline():
    '''
    Simple SVM baseline to predict building footprints
    '''
    # import image
    H, W = 200, 200
    add_features = True
    image_id = '6120_2_2' #'6100_2_3'
    image = load_image(image_id, 'M')
    image = normalize_image(image)
    train_image = image[:H, :W, :]
    test_image = image[:H, W:W * 2, :]
    print('Image loaded with dimensions: {}'.format(image.shape))
    print('\tWill train on portion of shape: {}'.format(train_image.shape))
    print('\tWill test on portion of shape: {}'.format(test_image.shape))
    # import mask for buildings
    mask = create_mask(image_id, *image.shape[:2], classes=[0])
    train_mask = mask[:H, :W]
    test_mask = mask[:H, W:W * 2]
    print('Mask loaded with dimensions: {}'.format(mask.shape))
    print('\tWill train on portion of mask of shape: {}'.format(train_mask.shape))
    print('\tWill test on portion of mask of shape: {}'.format(test_mask.shape))
    # add additional features
    if add_features:
        train_features = np.dstack((get_laplacian(train_image, 5), get_gaussian(train_image, 5)))
        #a = [train_image[..., i] for i in range(train_image.shape[2])]
        #b = [features[..., i] for i in range(features.shape[2])]
        #plot_compare_masks(a, b)
        train_image = np.dstack((train_image, train_features))
        test_features = np.dstack((get_laplacian(test_image, 5), get_gaussian(test_image, 5)))
        test_image = np.dstack((test_image, test_features))
    # train
    print('\nBegin training model\n')
    # The default kernel used from the below call to SVC is very slow
    #model = svm.SVC()
    # Use the linear kernel instead:
    model = svm.LinearSVC()
    samples = train_image.reshape(H * W, -1)
    labels = train_mask.reshape(-1)
    model.fit(samples, labels)
    #save_model(model)
    # test
    print('\nBegin testing model\n')
    test_x = [samples, test_image.reshape(H * W, -1)]
    predictions = []
    true = [train_mask, test_mask]
    for x in test_x:
        print('\tTesting on {} samples with {} features each.'.format(x.shape[0], x.shape[1]))
        pred_y = model.predict(x)
        predictions.append(pred_y.reshape(H, W))
    # calculate score
    for p, t in zip(predictions, true):
        score = my_jaccard(t, p)
        print('Score: {}'.format(score))
    # plot predictions
    print('Ploting predictions')
    plot_compare_masks(true, predictions)

def make_proposal_graphic():
    image_id = '6100_2_3'
    image_type = 'M'
    image = load_image(image_id, 'M')
    image = normalize_image(image)
    H, W, _ = image.shape
    mask = create_mask(image_id, H, W, classes=[0])
    features = np.dstack((get_laplacian(image, 3), get_gaussian(image, 3)))
    image = np.dstack((image, features)).reshape(H, W, -1)
    D = image.shape[-1]
    samples = image.reshape(-1, D)
    labels = mask.reshape(-1)
    model = load_model('svm_100_20171115-1239.pkl')
    prediction = model.predict(samples)
    true = [labels.reshape(H, W)]
    predictions = [prediction.reshape(H, W)]
    plot_compare_masks(np.expand_dims(image, axis=0), [mask], predictions)

def main():
    '''
    image_ids = []
    for x in range(6010, 6181, 10):
        for y in range(0, 5):
            for z in range(0, 5):
                i = '{}_{}_{}'.format(x, y, z)
                image_ids.insert(random.randint(0, len(image_ids)), i)
    image_ids = ['6100_2_3', '6120_2_2', '6120_2_0', '6100_1_3',
                 '6110_4_0', '6140_3_1', '6110_1_2', '6140_1_2',
                 '6110_3_1', '6060_2_3', '6070_2_3', '6100_2_2']
    '''
    image_ids = ['6100_2_3']
    image_type = 'M'
    models = ['svm']
    sizes = [100]
    features = None #(get_laplacian, get_gaussian)
    prepared_data = False
    # import data
    if not prepared_data:
        print('Preparing Data')
        data, labels = prep_data(image_ids, max(sizes), image_type)
        #save_data(data)
        #save_data(labels, True)
    else:
        data = load_data()
        labels = load_data(labels=True)
    experiment(data, labels, models, sizes, features)
    #results = experiment(data, labels, models, sizes, features)
    #save_results(results)

if __name__ == '__main__':
    cv = .2
    experiment_name = 'small_01'
    model_name = 'logistic'
    size = 500
    experiment(experiment_name, model_name, size, cv)
    experiment_name = 'small_01'
    model_name = 'svm'
    size = 500
    experiment(experiment_name, model_name, size, cv)
    experiment_name = 'medium_01'
    model_name = 'logistic'
    size = 1000
    experiment(experiment_name, model_name, size, cv)
    experiment_name = 'medium_01'
    model_name = 'svm'
    size = 1000
    experiment(experiment_name, model_name, size, cv)
    experiment_name = 'large_01'
    model_name = 'logistic'
    size = 2000
    experiment(experiment_name, model_name, size, cv)
    experiment_name = 'large_01'
    model_name = 'svm'
    size = 2000

