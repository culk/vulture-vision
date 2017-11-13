import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
from sklearn import svm
from sklearn.externals import joblib
import util

# TODO: which filters to use?
#cv2.getGaborKernel(ksize, sigma, theta, lambd, gamma[, psi[, ktype]])
model_dir = '/media/sf_school/project/models/'

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

def main():
    '''
    Simple SVM baseline to predict building footprints
    '''
    # import image
    image_id = '6100_2_3'
    image = util.load_image(image_id, 'M')
    image = util.normalize_image(image)
    H, W = 400, 400
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
    # train
    print('\nBegin training model\n')
    model = svm.SVC()
    samples = train_image.reshape(H * W, -1)
    labels = train_mask.reshape(-1)
    model.fit(samples, labels)
    save_model(model)
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

