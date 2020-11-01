import numpy
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
from helper import load_input, load_output, norm_img
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
DATASET_IMAGES = DATASET_BASE_PATH + "/"
DATASET_SIGNALS = DATASET_BASE_PATH + "/dataset.csv"

images, speed = load_input(DATASET_IMAGES, DATASET_SIGNALS)
output = load_output(DATASET_SIGNALS)

print(len(images[0]))
images = numpy.asarray(images)
print(images.shape)
speed = numpy.asarray(images)
speed = speed/(speed.max()- speed.min()+1)
print('normalized by ', speed.max()- speed.min()+1)
output = numpy.asarray(output)

model = generate(shape=(144, 144, 1))
model.compile(optimizer='adam', loss=['mse', 'mse'], metrics=['accuracy'], loss_weights=[1.0, 1.0])

checkpoint_path = "NN/checkpoint/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True,
                                                 verbose=1, save_freq = BATCH_SIZE)

history = model.fit(x=[images, speed], y=output, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                validation_split=0.33, callbacks=[cp_callback])
model.save("model")

print(model.metrics_names)
print(history.history.keys())
plt.figure(figsize=(15, 4))
acc = history.history['dense_1_accuracy']
val_acc = history.history['val_dense_1_accuracy']

loss=history.history['dense_1_loss']
val_loss=history.history['val_dense_1_loss']

epochs_range = range(EPOCHS)
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
plt.savefig('plot.png')