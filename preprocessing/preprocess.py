import csv
import cv2
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config import BASE_PATH


file = BASE_PATH + "/dataset/dataset.csv"
files = []

with open(file) as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        files.append(BASE_PATH + "/dataset/" + row[0])

for fname in files:    
    img = cv2.imread(fname)
    img = cv2.Canny(img,70,100)
    height, width = img.shape[:2]

    start_row, start_col = int(height * .5), int(0)
    end_row, end_col = int(height), int(width)
    img = img[start_row:end_row , start_col:end_col]

    width = 256
    height = 256 
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    os.remove(fname)
    cv2.imwrite(fname, img)