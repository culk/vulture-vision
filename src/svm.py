import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import csv
import random
from sklearn import svm, linear_model
from sklearn.externals import joblib
from itertools import combinations
import util

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
File I/O
'''
def save_model(model, name, size):
    dt = datetime.now().strftime('%Y%m%d-%H%M')
    model_filename = '{}_{}_{}.pkl'.format(name, size, dt)
    print('Saving model to disk: {}'.format(os.path.join(model_dir, model_filename)))
    joblib.dump(model, os.path.join(model_dir, model_filename))

def load_model(filename=None):
    # load most recent model
    if filename is None:
        filename = os.listdir(model_dir)[-1]
    model = joblib.load(os.path.join(model_dir, filename))
    return model

def save_data(data, labels=False):
    # save prepped data
    # TODO: compressed using np.savez?
    if labels:
        desc = 'labels'
    else:
        desc = 'samples'
    dt = datetime.now().strftime('%Y%m%d-%H%M')
    data_filename = '{}_{}_{}.npy'.format(desc, data.shape[0], dt)
    np.save(os.path.join(data_dir, 'prepped', data_filename), data)

def prep_data(ids, size, image_type, height=100, width=100, classes=[0]):
    samples = []
    labels = []
    n = 0
    threshold = 5
    while len(samples) < size:
        # import next image
        if n >= len(ids):
            print(len(samples))
            raise Exception('Not enough images to generate size of data')
        try:
            image = util.load_image(ids[n], image_type)
            image = util.normalize_image(image)
            H, W, _ = image.shape
            mask = util.create_mask(ids[n], H, W, classes)
        except:
            n += 1
            continue
        # add all height x width subregions of the image and mask
        for i in range(H//height):
            for j in range(W//width):
                i_start = i * height
                i_stop = i_start + height
                j_start = j * width
                j_stop = j_start + width
                if np.sum(mask[i_start:i_stop, j_start:j_stop]) < threshold:
                    continue
                samples.append(image[i_start:i_stop, j_start:j_stop])
                labels.append(mask[i_start:i_stop, j_start:j_stop])
        n += 1
    # convert into list of 100x100 subimages
    samples = np.asarray(samples[:size])
    labels = np.asarray(labels[:size])
    return samples, labels

def load_data(labels=False):
    # load most recently prepped data
    if labels:
        desc = 'labels'
    else:
        desc = 'samples'
    filenames = os.listdir(os.path.join(data_dir, 'prepped'))
    filename = list(filter(lambda s: desc in s, filenames))[-1]
    data = np.load(os.path.join(data_dir, 'prepped', filename))
    return data

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
    if len(image.shape) == 2:
        H, W = image.shape
        image = image.reshape(H, W, 1)
    features = np.zeros_like(image)
    for i in range(image.shape[2]):
        features[..., i] = cv2.Laplacian(image[..., i], ddepth=5, ksize=size)
    return features

def get_gaussian(image, size=3):
    kernel_shape = (size, size)
    if len(image.shape) == 2:
        H, W = image.shape
        image = image.reshape(H, W, 1)
    features = np.zeros_like(image)
    for i in range(image.shape[2]):
        features[..., i] = cv2.GaussianBlur(image[..., i], kernel_shape, 0)
    return features

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

def experiment(X, Y,
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
        # TODO: rotate/flip some samples
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
                train_score = util.my_jaccard(train_labels, train_prediction)
                test_prediction = model.predict(test_samples)
                # TODO: score breaks when the mask has zero of the class
                test_score = util.my_jaccard(test_labels, test_prediction)
                print('Model: {}\t Train: {}\t Test: {}'.format(model_name, train_score,
                                                                test_score))
                results[(model_name, feature_combo, size, 'train')] = train_score
                results[(model_name, feature_combo, size, 'test')] = test_score
                #if size > 50 and feature_combo is not None and len(feature_combo) == 2:
                    #save_model(model, model_name, size)
                print(train_samples.shape, train_labels.shape, train_prediction.shape)
                util.plot_compare_masks(train_samples, train_labels, train_prediction)
        print(results)
        #save_results(results)

def baseline():
    '''
    Simple SVM baseline to predict building footprints
    '''
    # import image
    H, W = 200, 200
    add_features = True
    image_id = '6100_2_3'
    image = util.load_image(image_id, 'M')
    image = util.normalize_image(image)
    train_image = image[:H, :W, :]
    test_image = image[:H, W:W * 2, :]
    print('Image loaded with dimensions: {}'.format(image.shape))
    print('\tWill train on portion of shape: {}'.format(train_image.shape))
    print('\tWill test on portion of shape: {}'.format(test_image.shape))
    # import mask for buildings
    mask = util.create_mask(image_id, *image.shape[:2], classes=[0])
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
        #util.plot_compare_masks(a, b)
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
        score = util.my_jaccard(t, p)
        print('Score: {}'.format(score))
    # plot predictions
    print('Ploting predictions')
    util.plot_compare_masks(true, predictions)

def make_proposal_graphic():
    image_id = '6100_2_3'
    image_type = 'M'
    image = util.load_image(image_id, 'M')
    image = util.normalize_image(image)
    H, W, _ = image.shape
    mask = util.create_mask(image_id, H, W, classes=[0])
    features = np.dstack((get_laplacian(image, 3), get_gaussian(image, 3)))
    image = np.dstack((image, features)).reshape(H, W, -1)
    D = image.shape[-1]
    samples = image.reshape(-1, D)
    labels = mask.reshape(-1)
    model = load_model('svm_100_20171115-1239.pkl')
    prediction = model.predict(samples)
    true = [labels.reshape(H, W)]
    predictions = [prediction.reshape(H, W)]
    util.plot_compare_masks(np.expand_dims(image, axis=0), [mask], predictions)

def main():
    '''
    image_ids = []
    for x in range(6010, 6181, 10):
        for y in range(0, 5):
            for z in range(0, 5):
                i = '{}_{}_{}'.format(x, y, z)
                image_ids.insert(random.randint(0, len(image_ids)), i)
    '''
    image_ids = ['6100_2_3', '6120_2_2', '6120_2_0', '6100_1_3',
                 '6110_4_0', '6140_3_1', '6110_1_2', '6140_1_2',
                 '6110_3_1', '6060_2_3', '6070_2_3', '6100_2_2']
    #image_ids = ['6100_2_3']
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
    #main()
    make_proposal_graphic()

