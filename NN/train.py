import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
from helper import load_images, load_signals, norm_img
from nn import generate
from sklearn.model_selection import train_test_split
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config import EPOCHS, BATCH_SIZE

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", help="dataset path", default="dataset")
args = parser.parse_args()

DATASET_BASE_PATH = args.path
DATASET_IMAGES = DATASET_BASE_PATH + "\images\\"
DATASET_SIGNALS = DATASET_BASE_PATH + "\signals.csv"

images = load_images(DATASET_IMAGES, DATASET_SIGNALS)
images = norm_img(images)
signals = load_signals(DATASET_SIGNALS)

images_train, images_test, signals_train, signals_test = train_test_split(
            images, signals, test_size=0.33)

model = generate(shape=images[0].shape, flatten="conv1d")
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
history = model.fit(images_train, signals_train, epochs=EPOCHS, batch_size=BATCH_SIZE
            validation_data=(images_test, signals_test))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(images_test,  signals_test, verbose=2)