import csv
import cv2
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config import BASE_PATH


file = BASE_PATH + "/dataset/dataset copy.csv"

def process_images(file, BASE_PATH, dim):
    files = []
    names = []
    with open(file) as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            files.append(BASE_PATH + "/dataset/" + row[0])
            if row[-1] == '0' :
                names.append(BASE_PATH+"/dataset/0/"+row[0])
            elif row[-1] == '1' :
                names.append(BASE_PATH+"/dataset/1/"+row[0])
            else:
                names.append(BASE_PATH+"/dataset/2/"+row[0])

    i = 0
    for fname in files:    
        img = cv2.imread(fname)
        img = cv2.Canny(img,70,100)
        height, width = img.shape[:2]

        start_row, start_col = int(height * .5), int(0)
        end_row, end_col = int(height), int(width)
        img = img[start_row:end_row , start_col:end_col]

        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        print(names[i])
        cv2.imwrite(names[i], img)
        i += 1

def choose_dir(s):
    x = s[-2]
    if x < 0.45:
        return 0
    elif x > 0.55:
        return 2
    return 1

def convert(r):
    for i in range(1, len(r)):
        r[i] = round(float(r[i]), 2)
    r[-2] = (r[-2] + 1) / 2.0
    return r

def process_signals(file):
    signals = []
    with open(file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            signals.append(row)

    print(signals[0])
    signals = list(map(convert, signals))
    print(signals[0])
    dir = list(map(choose_dir, signals))
    print(signals[0])

    for i in range(len(dir)):
        signals[i].append(dir[i])

    print(signals[0])
    with open(file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(signals)

process_signals(file)
process_images(file, BASE_PATH, (144, 144))