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

images_train, images_test, signals_train, signals_test = train_test_split(images, signals, test_size=0.33)

train_dataset = tf.data.Dataset.from_tensor_slices((images_train, signals_train))
train_dataset = train_dataset.batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((images_test, signals_test))
test_dataset = test_dataset.batch(BATCH_SIZE)

model = generate(shape=images[0].shape, flatten="conv1d")
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

checkpoint_path = "NN/checkpoint/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True,
                                                 verbose=1, save_freq = 2)

history = model.fit(train_dataset, epochs=EPOCHS,
            validation_data=test_dataset, callbacks=[cp_callback])

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

model.save("model")