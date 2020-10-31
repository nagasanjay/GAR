import cv2
import numpy as np
import os

images = []
folder = '/home/zeus/Desktop/dataset'
for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder,filename))
    if img is not None:
        images.append(img)

for img in images:
	#img = cv2.imread(i,0)
	edges = cv2.Canny(img,70,100)

	height, width = edges.shape[:2]

	start_row, start_col = int(height * .5), int(0)
	end_row, end_col = int(height), int(width)
	cropped = edges[start_row:end_row , start_col:end_col]


	#cv2.imwrite("cropped.png", cropped_bot) 
	#cv2.imshow("cropped image", cropped)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()


	width = 144
	height = 144 
	dim = (width, height)
	resized = cv2.resize(cropped, dim, interpolation = cv2.INTER_AREA) 
	cv2.imshow("Resized image", resized)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	#cv2.imwrite("1.png" , resized)
 







