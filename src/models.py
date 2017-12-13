from keras.models import Model
from keras import layers
from keras.optimizers import Adam
from keras import regularizers
from keras import backend as K

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

def create_deep_cnn():
    #K.image_data_format() returns channel order from config file
    inputs = layers.Input(shape=(64, 64, 6))
    conv11 = layers.Conv2D(64, 3, strides=1, activation='relu', padding='same', kernel_regularizer=regularizers.l2(.01))(inputs)
    conv12 = layers.Conv2D(64, 3, strides=1, activation='relu', padding='same', kernel_regularizer=regularizers.l2(.01))(conv11)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv12)

    conv21 = layers.Conv2D(128, 3, strides=1, activation='relu', padding='same', kernel_regularizer=regularizers.l2(.01))(pool1)
    conv22 = layers.Conv2D(128, 3, strides=1, activation='relu', padding='same', kernel_regularizer=regularizers.l2(.01))(conv21)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv22)
    
    conv31 = layers.Conv2D(256, 3, strides=1, activation='relu', padding='same', kernel_regularizer=regularizers.l2(.01))(pool2)
    conv32 = layers.Conv2D(256, 3, strides=1, activation='relu', padding='same', kernel_regularizer=regularizers.l2(.01))(conv31)
    up3 = layers.UpSampling2D(size=(2, 2))(conv32)

    conv41 = layers.Conv2D(128, 3, strides=1, activation='relu', padding='same', kernel_regularizer=regularizers.l2(.01))(up3)
    conv42 = layers.Conv2D(128, 3, strides=1, activation='relu', padding='same', kernel_regularizer=regularizers.l2(.01))(conv41)
    up4 = layers.UpSampling2D(size=(2, 2))(conv42)

    conv51 = layers.Conv2D(64, 3, strides=1, activation='relu', padding='same', kernel_regularizer=regularizers.l2(.01))(up4)
    conv52 = layers.Conv2D(64, 3, strides=1, activation='relu', padding='same', kernel_regularizer=regularizers.l2(.01))(conv51)

    final = layers.Conv2D(1, 1, strides=1, activation='linear')(conv52)
    model = Model(inputs=inputs, outputs=final)
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    return model

def create_deep_unet():
    #K.image_data_format() returns channel order from config file
    inputs = layers.Input(shape=(64, 64, 6))
    conv11 = layers.Conv2D(64, 3, strides=1, activation='relu', padding='same', kernel_regularizer=regularizers.l2(.01))(inputs)
    conv12 = layers.Conv2D(64, 3, strides=1, activation='relu', padding='same', kernel_regularizer=regularizers.l2(.01))(conv11)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv12)

    conv21 = layers.Conv2D(128, 3, strides=1, activation='relu', padding='same', kernel_regularizer=regularizers.l2(.01))(pool1)
    conv22 = layers.Conv2D(128, 3, strides=1, activation='relu', padding='same', kernel_regularizer=regularizers.l2(.01))(conv21)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv22)
    
    conv31 = layers.Conv2D(256, 3, strides=1, activation='relu', padding='same', kernel_regularizer=regularizers.l2(.01))(pool2)
    conv32 = layers.Conv2D(256, 3, strides=1, activation='relu', padding='same', kernel_regularizer=regularizers.l2(.01))(conv31)
    up3 = layers.UpSampling2D(size=(2, 2))(conv32)

    merge4 = layers.Concatenate(axis=-1)([up3, conv22])
    conv41 = layers.Conv2D(128, 3, strides=1, activation='relu', padding='same', kernel_regularizer=regularizers.l2(.01))(merge4)
    conv42 = layers.Conv2D(128, 3, strides=1, activation='relu', padding='same', kernel_regularizer=regularizers.l2(.01))(conv41)
    up4 = layers.UpSampling2D(size=(2, 2))(conv42)

    merge5 = layers.Concatenate(axis=-1)([up4, conv12])
    conv51 = layers.Conv2D(64, 3, strides=1, activation='relu', padding='same', kernel_regularizer=regularizers.l2(.01))(merge5)
    conv52 = layers.Conv2D(64, 3, strides=1, activation='relu', padding='same', kernel_regularizer=regularizers.l2(.01))(conv51)

    final = layers.Conv2D(1, 1, strides=1, activation='linear')(conv52)
    model = Model(inputs=inputs, outputs=final)
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    return model

