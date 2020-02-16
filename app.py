import cv2
import numpy as np
import matplotlib.pyplot as plt

from functools import reduce
import operator
import math

from flask import Flask, render_template, request, redirect
import jinja2

app = Flask(__name__)

MIN_POLYGON_AREA = 50000
LOW_THRESH = 80
HIGH_THRESH = 140

shape_dict = {3 : 'triangle', 4 : 'rectangle', 
	5 : 'pentagon', 6 : 'hexagon', 
	11 : 'circle', 12 : 'circle',
	13 : 'circle', 14 : 'circle', 
	15 : 'circle', 16 : 'circle'}

electronics_dict = {'rectangle' : 'voltage source', 
'triangle' : 'resistor', 
'circle' : 'LED', 
'pentagon' : 'potentiometer'}


def show_image(img_name, title='Fig', grayscale=False):
    if not grayscale:
        plt.imshow(img_name)
        plt.title(title)
        plt.show()


# we will get a POST request containing an image.
# step 1 : remove the center logo that was used to calibrate ARKit
# step 2 : remove lines (via thresholding, filters)
# step 3 : identify shapes representing electronic components
# step 4 : create and return a clockwise list of electronic components back to iOS app 

@app.route('/process', methods=['POST'])
def process_image():
	#img = flask.request.files.get('imagefile', '')
	img = request.files['file']

	# img = cv2.imread('5_shapes.jpg', 0) # will eventually replace this line
	# img_copy = img

	### Step 1 : Remove the center logo that was used to calibrate ARKit ###
	img[1300:2200, 900:2100] = 255

	# show_image(img)

	### Step 2 : Remove lines (via thresholding, filters) ###

	for i in range(15):
		img = cv2.medianBlur(img, 5)

	#RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	#(b, g, r) = cv2.split(RGB_img)
	#b_bad, g_bad, r_bad = b < 130, g < 35, r < 220
	#b[b_bad], g[g_bad], r[r_bad] = 255, 255, 255
	
	# ret2, thresh2 = cv2.threshold(b, 130, 140, cv2.THRESH_BINARY)
	# ret3, thresh3 = cv2.threshold(g, 35, 75, cv2.THRESH_BINARY)
	# ret4, thresh4 = cv2.threshold(r, 200, 240, cv2.THRESH_BINARY)
	# bgr_thresh = cv2.merge((thresh2, thresh3, thresh4))
	# bgr_thresh = cv2.merge((b, g, r))

	#ret, thresh1 = cv2.threshold(RGB_img,30,255,cv2.THRESH_BINARY)
	pix_bad = ((img < LOW_THRESH) | (img > HIGH_THRESH)) # arbitrary color
	img[pix_bad] = 255
	img = 255 - img

	# dilation
	kernel = np.ones((5, 5), np.uint8)
	img = cv2.dilate(img, kernel, iterations = 1)
	
	# erosion
	kernel = np.ones((5, 5), np.uint8)
	img = cv2.erode(img, kernel, iterations = 1)

	# median blur
	for i in range(15):
		img = cv2.medianBlur(img, 5)

	### Step 3 : Identify shapes representing electronic components ###

	contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	coords = []
	coords_dict = {}

	for cnt in contours:
		if cv2.contourArea(cnt) < MIN_POLYGON_AREA:
			continue

		else:
			# print(cv2.contourArea(cnt))
			approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
			# print(len(approx)) # number of sides in the polygon
			
			# create bounding rectangle pixels
			x,y,w,h = cv2.boundingRect(cnt)
			coords.append((x, y))
			coords_dict[(x, y)] = shape_dict[len(approx)]
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)

	### Step 4 : Create and return a clockwise list of electronic components back to iOS app ###
	center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
	points_sorted = sorted(coords, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)[::-1]
	# print(points_sorted)
	#for p in points_sorted:
	#	print(coords_dict[p])

	electronics_list = [electronics_dict[coords_dict[p]] for p in points_sorted]
	print(electronics_list)

	return electronics_list

	show_image(img)

if __name__ == '__main__':
	app.run()


# process_image()
