import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import csv
from sklearn import svm
from sklearn.externals import joblib
from itertools import combinations
import util

# TODO: which filters to use?
#cv2.getGaborKernel(ksize, sigma, theta, lambd, gamma[, psi[, ktype]])
model_dir = '/media/sf_school/project/models/'
results_dir = '/media/sf_school/project/results/'

'''
File I/O
'''
def save_model(model):
    dt = datetime.now().strftime('%Y%m%d-%H%M')
    model_filename = 'svm_{}.pkl'.format(dt)
    print('Saving model to disk: {}'.format(os.path.join(model_dir, model_filename)))
    joblib.dump(model, os.path.join(model_dir, model_filename))

def load_model():
    # load most recent model
    filename = os.listdir(model_dir)[-1]
    model = joblib.load(os.path.join(model_dir, filename))
    return model

def save_data():
    # save prepped data
    pass

def prep_data():
    # import images
    # convert into list of 100x100 subimages
    # save subimages and masks to disk
    pass

def load_data():
    # load already prepped data
    pass

def save_results(results):
    '''
    Writes the results of the experiment to a csv file in the results_dir
    Input:
    results - a dictionary with keys that are tuples of (model_name, features, size)
              and values that are the jaccard score
    '''
    dt = datetime.now().strftime('%Y%m%d-%H%M')
    results_filename = 'svm_results_{}.csv'.format(dt)
    with open(os.path.join(results_dir, results_filename), 'w', newline='') as results_file:
        results_writer = csv.writer(results_file, delimiter=',', quotechar='|',
                                    quoting=csv.QUOTE_MINIMAL)
        results_writer.writerow(['model', 'features', 'size', 'jaccard_score'])
        for (model, features, size), jaccard_score in results.items():
            results_writer.writerow([model, features, size, jaccard_score])

'''
Features
'''
def get_laplacian(image, size=3):
    features = np.zeros_like(image)
    for i in range(image.shape[2]):
        features[..., i] = cv2.Laplacian(image[..., i], ddepth=1, ksize=size)
    return features

def get_gaussian(image, size=3):
    kernel_shape = (size, size)
    features = np.zeros_like(image)
    for i in range(image.shape[2]):
        features[..., i] = cv2.GaussianBlur(image[..., i], kernel_shape, 0)
    return features

'''
Experiments
'''
def experiment(X, Y,
               models=(svm.LinearSVC()),
               sizes=[10],
               #sizes=[10, 25, 50, 100, 200, 500, 1000], #have 1600 in total right now
               features=(get_laplacian, get_gaussian)):
    '''
    Do experiments for all combinations of settings
    Input:
    X - numpy array of the images of shape (num_subimages, H, W, D)
    '''
    # data is list of subimages, each 100x100
    # labels are corresponding bit masks
    # randomly sample max(size) subimages
    # create tuple of feature combinations to test
    feature_combinations = [tuple(combinations(features, r))
                            for r in range(1, len(features) + 1)]
    feature_combinations.append(None)
    print(feature_combinations)
    for size in sizes:
        # TODO: rotate/flip some samples
        # select first size subimages
        samples = X[:size]
        labels = Y[:size].reshape(-1, 1)
        N, H, W, _ = samples.shape
        M = labels.shape[0]
        assert M == N * H * W
        for feature_combo in feature_combinations:
            for feature in feature_combo:
                # add features for each subimage
                new_samples = np.zeros_like(samples)
                for i in range(samples.shape[0]):
                    new_samples[i] = feature_function(samples[i])
                samples = np.dstack(samples, new_samples)
            # reshape input and labels
            D = samples.shape[-1]
            samples = samples.reshape(-1, D) # M x D [number of pixels, number of bands]
            assert samples.shape[0] == M
            for model_name in models: # logistic regression / svm
                if model_name == 'svm':
                    model = svm.LinearSVC()
                elif model_name == 'lr':
                    # TODO: implement logistic regression
                    break
                # train model
                model.fit(samples, labels)
                # crossvalidate model?
                # record model results, as csv?
                #   jaccard score, 
                # save copy of model?

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

def main():
    models = ('svm')
    sizes = [10]
    # import data
    data = None
    labels = None
    experiment(data, labels, models, sizes, features=(get_laplacian, get_gaussian))

if __name__ == '__main__':
    baseline()

