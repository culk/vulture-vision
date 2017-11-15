import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
from sklearn import svm
from sklearn import linear_model
from sklearn.externals import joblib
import util

# TODO: which filters to use?
#cv2.getGaborKernel(ksize, sigma, theta, lambd, gamma[, psi[, ktype]])
model_dir = '~/Downloads'

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


def main():
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
    model = linear_model.LogisticRegression()
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

if __name__ == '__main__':
	main()
