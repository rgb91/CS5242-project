import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Activation, MaxPool2D, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D


def network(input_shape=(49, 49, 49, 1), class_num=2):
    """Example CNN
    
    Keyword Arguments:
        input_shape {tuple} -- shape of input images. Should be (28,28,1) for MNIST and (32,32,3) for CIFAR (default: {(28,28,1)})
        class_num {int} -- number of classes. Shoule be 10 for both MNIST and CIFAR10 (default: {10})
    
    Returns:
        model -- keras.models.Model() object
    """

    im_input = Input (shape=input_shape)

    t = Convolution3D (32, (24, 24, 24), padding='same') (im_input)  # (24,24,24)
    t = MaxPooling3D (pool_size=(2, 2, 2)) (t)
    t = Convolution3D (64, (12, 12, 12), padding='same') (t)
    t = MaxPooling3D (pool_size=(2, 2, 2)) (t)
    t = Convolution3D (128, (6, 6, 6), padding='same') (t)
    t = Flatten () (t)
    t = Dense (256) (t)
    t = Activation ('relu') (t)
    t = Dense (class_num) (t)

    output = Activation ('softmax') (t)
    model = Model (input=im_input, output=output)
    return model
