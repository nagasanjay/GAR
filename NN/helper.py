import csv
from math import exp
import os
import sys
from numpy.core.defchararray import asarray

from numpy.lib.function_base import append
from numpy.lib.shape_base import expand_dims
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config import BASE_PATH
import numpy as np
from PIL import Image
import cv2

def load_output(filename):
    signals = []

    with open(BASE_PATH + filename) as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            signals.append(list(map(lambda x : float(x), row[1:4])))
    signals = np.array(signals)
    signals[:, 1] = (signals[:, 1]+1)/2
    return signals


def open_and_reshape (fname):
    #img = Image.open(fname).resize((144, 144))

    #print(fname)
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    edges = cv2.Canny(img,70,100)

    height, width = edges.shape[:2]

    start_row, start_col = int(height * .5), int(0)
    end_row, end_col = int(height), int(width)
    cropped = edges[start_row:end_row , start_col:end_col]

    width = 144
    height = 144 
    dim = (width, height)
    resized = cv2.resize(cropped, dim, interpolation = cv2.INTER_AREA) 
    resized = expand_dims(resized, axis=-1)
    return resized
    

def load_input (path, filename):
    images = []
    speed = []

    with open(BASE_PATH + filename) as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            speed.append(float(row[1]))
            images.append(norm_img(open_and_reshape(BASE_PATH + path + row[0])))

    return (images, speed)


def norm_img(img):
    #print(img.shape)
    img = img / 255.0
    img = img.tolist()
    return img
