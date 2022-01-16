# python detect_barcode_opencv.py --image images/barcode_01.jpg

# import the necessary packages
import numpy as np
import argparse
import cv2
import glob
from random import randint
from dynamosoft import *
#from scandit import *


def img_enhancement(img):
    filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img=cv2.filter2D(img,-1,filter)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    thresh1 = convert_3_channel(thresh1)
    return thresh1

def detect_bar_code(img,decoder):
	scale = 4
	image = cv2.resize(img.copy(),None,fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
	image_org = image.copy()
	# print(image_org.shape)
	#convert to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	#calculate x & y gradient
	gradX = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
	gradY = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)

	# subtract the y-gradient from the x-gradient
	gradient = cv2.subtract(gradX, gradY)
	gradient = cv2.convertScaleAbs(gradient)
	# print('gradient---',gradient.shape)
	# cv2.imshow("gradient-sub",cv2.resize(gradient,None, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC))
	blurred = cv2.blur(gradient, (3, 3))

	# threshold the image
	(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
	# print("threshed",thresh.shape)
	# cv2.imshow("threshed",cv2.resize(thresh,None, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC))
	# construct a closing kernel and apply it to the thresholded image
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
	thresh_1 = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
	# print("morphology",thresh_1.shape)
	# cv2.imshow("morphology",cv2.resize(closed,None, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC))
	# perform a series of erosions and dilations
	closed = cv2.erode(thresh_1, None, iterations = 4)
	closed = cv2.dilate(closed, None, iterations = 4)
	# print("erode/dilate",closed.shape)
	# cv2.imshow("erode/dilate",cv2.resize(closed,None, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC))
	inter_mediate = np.vstack([np.hstack([gradient,thresh]), np.hstack([thresh_1,closed])])
	inter_mediate = cv2.resize(inter_mediate, None, fx=1/scale, fy=1/scale, interpolation = cv2.INTER_CUBIC)
	# find the contours in the thresholded image, then sort the contours
	# by their area, keeping only the largest one
	# closed = cv2.resize(closed, None, fx=1/scale, fy=1/scale, interpolation = cv2.INTER_CUBIC)
	cnts,hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]

	# c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
	# c1 = sorted(cnts, key = cv2.contourArea, reverse = True)[1]
    

	sort_cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
	buf = 40
	# sort_cnts_ext = []
	results = {}
	for j,cnt in enumerate(sort_cnts):
		if cv2.contourArea(cnt) > 20000:
			print('------ Barcode %d'%j,'---------')
			x,y,w,h = cv2.boundingRect(cnt)
			# x-buf ,y-buf x+w+buf,y-buf, x+w+buf,y+h+buf, x-buf,y+h+buf
			crop = image_org[y-buf:y+h+buf,x-buf:x+w+buf,:].copy()
			# thresh = img_enhancement(crop.copy())
			crop_area = crop.shape[0]*crop.shape[1]
			# print('Aaaaaaaaaaaaaaa',crop_area)
			if crop_area > 5:
				cv2.imwrite('temp/temp.png',crop)


				# cv2.imwrite('temp/'+name.split('.')[0]+'_%d.png'%j,crop)

				############################# Pzybar ##############################
				if decoder == "Pyzbar":
					barcodes = read_barcodes(crop)
					if len(barcodes)!=0:
						for barcode in barcodes:
							barcode_info = barcode.data.decode('utf-8')
							print('pyzbar decodeing-----',barcode_info)
							##################################################################################
							box_id = text+'-'+str(j)
							coord =[x-buf ,y-buf, x+w+buf,y-buf, x+w+buf,y+h+buf, x-buf,y+h+buf]
							box_coord = [int(i/4) for i in coord]
							# box_coord  = np.divide(coord, 4)	
							results.update({box_id:box_coord})
							###################################################################################
							cv2.putText(image, barcode_info, (x, y+150), cv2.FONT_HERSHEY_COMPLEX, 2.5, (0, 255, 0), 5)
							# break;
					else:
						print("Pyzbar not able to decode")
				###################################################################

				############################ Dynamosoft ###########################
				if decoder == "Dynamosoft":
					text_results = dynamosoft()
					# print(text_results)
					if text_results != None:
						for text_result in text_results:
							print("Dynamosoft decoding-----Barcode Text: " + text_result.barcode_text)
							text = text_result.barcode_text
							cv2.putText(image, text, (x, y + 50), cv2.FONT_HERSHEY_COMPLEX, 2.5, (0, 0, 255), 5)
							##################################################################################
							box_id = text+'-'+str(j)
							coord =[x-buf ,y-buf, x+w+buf,y-buf, x+w+buf,y+h+buf, x-buf,y+h+buf]
							box_coord = [int(i/4) for i in coord]
							# box_coord  = np.divide(coord, 4)	
							results.update({box_id:box_coord})
							###################################################################################
					else:
						print("Dynamosoft not able to decode")
				############################ scandit ###########################
				"""
				if decoder == "Scandit":
					codes = scandit(crop)
					if len(codes):
						text = codes[0].data
						cv2.putText(image, text, (x, y + 250), cv2.FONT_HERSHEY_COMPLEX, 2.5, (255, 0, 0), 5)
						##################################################################################
						box_id = text+'-'+str(j)
						coord =[x-buf ,y-buf, x+w+buf,y-buf, x+w+buf,y+h+buf, x-buf,y+h+buf]
						box_coord = [int(i/4) for i in coord]
						# box_coord  = np.divide(coord, 4)	
						results.update({box_id:box_coord})
						###################################################################################
					else:
						print("Scandit not able to decode")
				"""
			#####################################################################################
			cv2.rectangle(image,(x-buf,y-buf),(x+w+buf,y+h+buf),(255,0,0),5)
			# cv2.rectangle(img,(int((x-buf)/4),int((y-buf)/4)),(int((x+w+buf)/4),int((y+h+buf)/4)),(255,0,0),5)
			#####################################################################################
			# sort_cnts_ext.append(cnt)
	image = cv2.resize(image, None, fx=1/scale, fy=1/scale, interpolation = cv2.INTER_CUBIC)
	# print("Final Results------------>",results)
	return image,results

if __name__ == '__main__':
	img_files = glob.glob('img/*')
	for img_file in img_files:
		name = img_file.split('/')[-1]
		print('#################',img_file.split('/')[-1],'##################')
		image = cv2.imread(img_file)
		image, inter_mediate,results = detect_bar_code(image,"Scandit")
		print("Final Results------------>",results)
		# print(image.shape)
		cv2.imwrite('nishant_temp_result/'+name,image)
		cv2.imwrite('inter_mediate/'+name,inter_mediate)


