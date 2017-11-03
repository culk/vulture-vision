import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.wkt import loads as wkt_loads
import shapely.wkt
import shapely.affinity
from shapely.geometry import MultiPolygon, Polygon
import tifffile as tiff
from keras.models import Model
from keras import layers
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from sklearn.metrics import jaccard_similarity_score
from util import load_image, normalize_image, create_mask, my_jaccard


# globals
num_class = 10
data_dir = '/media/sf_school/project/data'
train_labels = pd.read_csv(os.path.join(data_dir, 'train_wkt_v4.csv'))
grid_sizes = pd.read_csv(os.path.join(data_dir, 'grid_sizes.csv'), names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)

# stitch images together and save images and masks to disk
def image_stitch(ids, img_type='M'):
    print("\nloading {} images with ids={}".format(len(ids), ids))
    if img_type == 'M':
        H, W = 800, 800
        X = np.zeros((2 * H, 2 * W, 8))
        Y = np.zeros((2 * H, 2 * W, num_class))
    else:
        print('unsupported image type')
        return 0

    for i, image_id in enumerate(ids):
        image = load_image(image_id, 'M')
        image = normalize_image(image)
        print(image_id, image.shape, np.amin(image), np.amax(image))
        X[(i//2) * H:(i//2) * H + H, (i%2) * W:(i%2) * W + W] = image[:H, :W]
        for cls in range(num_class):
            Y[(i//2) * H:(i//2) * H + H, (i%2) * W:(i%2) * W + W, cls] = create_mask(image.shape[:2], image_id, cls + 1)[:H, :W]
    print(np.amax(Y), np.amin(Y))

    np.save(os.path.join(data_dir, 'input_{}x{}_{}.npy'.format(H*2, W*2, img_type)), X)
    np.save(os.path.join(data_dir, 'mask_{}x{}_{}.npy'.format(H*2, W*2, num_class)), Y)

# this is actually training and dev sets...
def gen_test_dev_sets(img_type='M'):
    sample_size = 100
    dev_size = int(sample_size * .2)
    
    if img_type == 'M':
        H, W = 800, 800
    else:
        print('unsupported image type')
        return 0

    image = np.load(os.path.join(data_dir, 'input_{}x{}_{}.npy'.format(H*2, W*2, img_type)))
    mask = np.load(os.path.join(data_dir, 'mask_{}x{}_{}.npy'.format(H*2, W*2, num_class)))

    print("\nbegin dividing samples")
    sub_h, sub_w = 160, 160
    indices = np.random.permutation(sample_size)
    train_indices = indices[dev_size:]
    dev_indices = indices[:dev_size]
    sub_images = []
    sub_masks = []
    for i in range(10):
        for j in range(10):
            sub_images.append(image[i*sub_h:i*sub_h + sub_h, j*sub_w:j*sub_w + sub_w])
            sub_masks.append(mask[i*sub_h:i*sub_h + sub_h, j*sub_w:j*sub_w + sub_w])
    sub_images = np.transpose(sub_images, (0, 3, 1, 2))
    sub_masks = np.transpose(sub_masks, (0, 3, 1, 2))
    train_X = sub_images[train_indices]
    train_Y = sub_masks[train_indices]
    dev_X = sub_images[dev_indices]
    dev_Y = sub_masks[dev_indices]
    print("train dims = \t{} {}".format(train_X.shape, train_Y.shape))
    print("dev dims = \t{} {}".format(dev_X.shape, dev_Y.shape))

    np.save(os.path.join(data_dir, 'train_input_{}x{}_{}'.format(H*2, W*2, img_type)), train_X)
    np.save(os.path.join(data_dir, 'train_mask_{}x{}_{}'.format(H*2, W*2, num_class)), train_Y)
    np.save(os.path.join(data_dir, 'dev_input_{}x{}_{}'.format(H*2, W*2, img_type)), dev_X)
    np.save(os.path.join(data_dir, 'dev_mask_{}x{}_{}'.format(H*2, W*2, num_class)), dev_Y)

def create_model():
    #K.image_data_format() returns channel order from config file
    inputs = layers.Input(shape=(8, 160, 160))
    conv1 = layers.Conv2D(32, 3, strides=1, activation='relu', padding='same', input_shape=(8, 160, 160))(inputs)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(64, 3, strides=1, activation='relu', padding='same')(pool1)

    #up3 = layers.merge([layers.UpSampling2D(size=(2, 2))(conv2), conv1], mode='concat', concat_axis=1)
    up3 = layers.UpSampling2D(size=(2, 2))(conv2)
    merge3 = layers.Concatenate(axis=1)([up3, conv1])
    conv3 = layers.Conv2D(32, 3, strides=1, activation='relu', padding='same')(merge3)

    final = layers.Conv2D(num_class, 1, strides=1, activation='sigmoid')(conv3)
    model = Model(inputs=inputs, outputs=final)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train(model):
    n_iters = 2
    weights_filename_cp = '/media/sf_school/project/weights/baseline_{epoch:02d}-{val_acc:.2f}.hdf5'
    weights_filename = '/media/sf_school/project/weights/baseline_finish.hdf5'
    H, W = 800, 800
    img_type = 'M'
    train_X = np.load(os.path.join(data_dir, 'train_input_{}x{}_{}.npy'.format(H*2, W*2, img_type)))
    train_Y = np.load(os.path.join(data_dir, 'train_mask_{}x{}_{}.npy'.format(H*2, W*2, num_class)))
    dev_X = np.load(os.path.join(data_dir, 'dev_input_{}x{}_{}.npy'.format(H*2, W*2, img_type)))
    dev_Y = np.load(os.path.join(data_dir, 'dev_mask_{}x{}_{}.npy'.format(H*2, W*2, num_class)))
    
    print("begin training on {} samples".format(train_X.shape[0]))
    checkpoint = ModelCheckpoint(weights_filename_cp, monitor='loss', save_best_only=True)
    # for once I know the scores are calculated correctly
    model.load_weights(weights_filename)
    model.fit(train_X, train_Y, batch_size=10, epochs=n_iters, verbose=2,
              callbacks=[checkpoint], validation_data=(dev_X, dev_Y))
    model.save_weights(weights_filename)

def predict(model, image_id):
    scores = []
    H, W = 800, 800
    # load test image and mask
    print("\npredicting mask for new image and calculating score")
    image = load_image(image_id, 'M')[:H, :W]
    image = normalize_image(image)
    mask = np.zeros((H, W, num_class))
    print(image_id, image.shape, np.amin(image), np.amax(image))
    for cls in range(num_class):
        mask[..., cls] = create_mask(image.shape[:2], image_id, cls + 1)[:H, :W]
    print(np.amax(mask), np.amin(mask))

    # use the model to predict the mask
    images, masks = [], []
    sub_h, sub_w = 160, 160
    print("check sub images are correctly arrayed")
    for i in range(5):
        for j in range(5):
            ys = i*sub_h
            ye = ys + sub_h
            xs = j*sub_w
            xe = xs + sub_w
            images.append(image[ys:ye, xs:xe])
            masks.append(mask[ys:ye, xs:xe])
    #images = np.rollaxis(np.array(images), 3, 1)
    #masks = np.rollaxis(np.array(masks), 3, 1)
    images = np.transpose(images, (0, 3, 1, 2))
    masks = np.transpose(masks, (0, 3, 1, 2))
    #mask = np.transpose(mask, (0, 1, 2))
    predictions = model.predict(images, batch_size=4, verbose=1)
    prediction = np.zeros_like(mask)
    for i in range(5):
        for j in range(5):
            ys = i*sub_h
            ye = ys + sub_h
            xs = j*sub_w
            xe = xs + sub_w
            for cls in range(num_class):
                prediction[ys:ye, xs:xe, cls] = predictions[i * 5 + j, cls]
    print(predictions.shape)
    print(masks.shape)

    plt.figure()
    # it would be cool if it printed 4 pictures
    # true          predicted
    # intersection  union
    plt1 = plt.subplot(121)
    plt1.set_title(image_id + ' building mask')
    plt1.imshow(mask[..., 0], cmap=plt.get_cmap('gray'))
    plt2 = plt.subplot(122)
    plt2.set_title('building prediction')
    plt2.imshow(np.rint(prediction[..., 0]), cmap=plt.get_cmap('gray'))
    plt.show()

    # calc the jaccard score
    for cls in range(num_class):
        scores.append(my_jaccard(mask[..., cls], prediction[..., cls]))
#        scores.append(jaccard_similarity_score(mask[..., cls].flatten(), 
#                                                   np.rint(prediction[..., cls].flatten())))

#    scores, thresholds = [], []
#    for cls in range(num_class):
#        c_masks = masks[:, cls, ...]
#        c_preds = predictions[:, cls, ...]
#        c_masks = c_masks.reshape(-1, masks.shape[3])
#        c_preds = c_preds.reshape(-1, masks.shape[3])
#        score, best_threshold = 0, 0
#        for i in range(1, 10):
#            threshold = i / 10.0
#            preds_b = c_preds > threshold
#            j_index = jaccard_similarity_score(c_masks, preds_b)
#            if j_index > score:
#                score = j_index
#                best_threshold = threshold
#        print(cls, score, best_threshold)
#        scores.append(score)
#        thresholds.append(best_threshold)

    return scores

def main():
    # images represent a diversity of features with a focus on man-made features
    do_training = True
    ids = ['6100_2_3', '6110_4_0', '6110_1_2', '6140_3_1']
    test_id = '6100_2_2'
    #image_stitch(ids)
    #gen_test_dev_sets()
    model = create_model()
    if do_training:
        model.summary()
        train(model)
    scores = predict(model, test_id)
    print("Jaccard similarity scores:")
    for cls in range(num_class):
        print("class: {}\t score: {}".format(cls + 1, scores[cls]))
    print("average: {}".format(sum(scores)/len(scores)))

if __name__=='__main__':
    main()

