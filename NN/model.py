# necessary imports
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, Conv1D, MaxPooling2D
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
# example:      generate((144, 144, 3), 'conv1d')

def generate(shape, option="flatten"):
    visible = Input(shape=shape)

    conv1 = Conv2D(64, kernel_size=2, activation='relu')(visible)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(32, kernel_size=2, activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(16, kernel_size=2, activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # way 1 to flatten - many params
    if option == 'flatten':
        flat = Flatten()(pool3)

    # way 2 to flatten - less params
    elif option == 'conv1d':
        conv4 = Conv1D(128, kernel_size=2, activation='relu')(pool3)
        flat  = MaxPooling2D(pool_size=(conv4.shape[1], conv4.shape[2]))(conv4)

    # mentioned method doesn't exist
    else:
        print("\nInvalid flatten option")
        sys.exit(1)

    hidden1 = Dense(32, activation='relu')(flat)

    output1 = Dense(1, activation='linear')(hidden1)
    output2 = Dense(1, activation='linear')(hidden1)

    model = Model(inputs=visible, outputs=[output1, output2])
    return model