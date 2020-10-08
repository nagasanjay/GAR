import csv
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config import BASE_PATH
import numpy as np
from PIL import Image

def load_signals(filename):
    signals = []
    
    if os.path.exists(BASE_PATH + filename + ".npz"):
        signals = np.load(BASE_PATH + filename + ".npz", allow_pickle=True)
        return np.array(signals['signals'])

    with open(BASE_PATH + filename) as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            signals.append([row[1:4]])
    print(signals)
    signals = np.array(signals, dtype=np.float32)
    print(signals)
    np.savez_compressed(BASE_PATH + filename, signals=signals)
    return signals

def open_and_reshape(fname):
    img = Image.open(fname).resize((144, 144))
    return img
    

def load_images(path, filename):
    images = []
    files = []

    if os.path.exists(BASE_PATH + path + "img.npz"):
        images = np.load(BASE_PATH + path + "img.npz", allow_pickle=True)
        return np.array(images['images'])

    with open(BASE_PATH + filename) as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            files.append(row[0])
        images = np.array([np.array(open_and_reshape(BASE_PATH + path + fname)) for fname in files])
    np.savez_compressed(BASE_PATH  + path + "img", images=images)
    return images

def norm_img(img):
    img = img / 255.0
    return img