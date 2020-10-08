import csv
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config import BASE_PATH
import numpy as np
from PIL import Image

def load_signals(filename):
    signals = []
    with open(BASE_PATH + filename) as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            signals.append(row[1:4])
    signals = np.array(signals, dtype=np.float32)
    return signals

def load_images(path, filename):
    images = []
    with open(BASE_PATH + filename) as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            img = Image.open(BASE_PATH + path + row[0])
            img = np.array(img)
            images.append(img)
    images = np.array(images)
    return images

def norm_img(img):
    img = img / 255.0
    return img

