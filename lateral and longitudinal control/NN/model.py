# necessary imports
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, Conv1D, MaxPooling2D
from tensorflow.keras.utils import plot_model
import tensorflow as tf
import sys

# generate a model for given input size
#
# required inputs :
# tuple with 3 elements (height, width, number_of_channels)
#
# optional inputs :
# mention how we want to flatten the NN
# two options available -> flatten / conv1d
# flatten is the default option
#
# example:      directionalModel((144, 144, 3), 'conv1d')

def directionModel(shape=(144, 144, 1), option="flatten"):
    visible = Input(shape=shape)

    conv1 = Conv2D(16, kernel_size=5, activation='relu', padding='same')(visible)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(32, kernel_size=3, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(128, kernel_size=3, activation='relu', padding='same')(pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # way 1 to flatten - many params
    if option == 'flatten':
        flat = Flatten()(pool4)

    # way 2 to flatten - less params
    elif option == 'conv1d':
        conv5 = Conv1D(1024, kernel_size=3, activation='relu', padding='same')(pool4)
        flat  = MaxPooling2D(pool_size=(conv5.shape[1], conv5.shape[2]))(conv5)
        flat = tf.squeeze(flat, axis=[1, 2])

    # mentioned method doesn't exist
    else:
        print("\nInvalid flatten option")
        sys.exit(1)

    hidden1 = Dense(512, activation='relu')(flat)
    hidden2 = Dense(128, activation='relu')(hidden1)

    # direction
    output1 = Dense(3, activation='softmax')(hidden2)

    model = Model(inputs=visible, outputs=output1)
    return model

# writing documentation is FUN ???!!! :(
def signalsModel():
    input = Input(shape=(2,))

    hidden1 = Dense(16, activation='relu')(input)
    hidden2 = Dense(32, activation='relu')(hidden1)
    hidden3 = Dense(64, activation='relu')(hidden2)
    hidden4 = Dense(32, activation='relu')(hidden3)

    output = Dense(3, activation='sigmoid')(hidden4)

    model = Model(inputs=input, outputs=output)
    return model