from re import VERBOSE
import numpy
from numpy.lib.shape_base import expand_dims
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
from helper import load_input, load_output
from nn import generate
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
print('done')
output = load_output(DATASET_SIGNALS)

print(len(images[0]))
images = numpy.asarray(images)
print(images.shape)
speed = numpy.asarray(speed)
print('normalizing by ', speed.max()- speed.min()+1)
speed = speed/(speed.max()- speed.min()+1)
print(speed.shape)
output = expand_dims(numpy.asarray(output),axis=1)
print(output.shape)
print(type(output[0]), type(output), type(output))

print(images[0])
print(speed[0:10])
print(output[0:10])

model = generate(shape=(256, 256, 1))
model.summary()
model.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])

checkpoint_path = "NN/checkpoint/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True,
                                                 verbose=1, save_freq = 50)

history = model.fit(x=[images, speed], y=output, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=True, 
                validation_split=0.33, callbacks=[cp_callback])
model.save("model")

print(model.metrics_names)
print(history.history.keys())
'''
plt.figure(figsize=(15, 4))
acc = history.history['dense_2_accuracy']
val_acc = history.history['val_dense_2_accuracy']

loss=history.history['dense_2_loss']
val_loss=history.history['val_dense_2_loss']

epochs_range = range(10)
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
'''

data = [0, 429, 545, 811, 915, 921, 969, 1305, 1445]

for d in data:
    print('-------------------------------------------------')
    image = expand_dims(images[d], axis=0)
    spee = expand_dims(speed[d], axis=0)
    print(model.predict([image, spee]))
    print(output[d])