#import tensorflow as tf
#import matplotlib.pyplot as plt
import argparse
from helper import load_images, load_signals, norm_img
from nn import generate

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", help="dataset path", required=True)
args = parser.parse_args()

DATASET_BASE_PATH = args.path
DATASET_IMAGES = DATASET_BASE_PATH + "\images\\"
DATASET_SIGNALS = DATASET_BASE_PATH + "\signals.csv"

images = load_images(DATASET_IMAGES, DATASET_SIGNALS)
images = norm_img(images)
signals = load_signals(DATASET_SIGNALS)

model = generate(shape=images[0].shape, flatten="conv1d")