# python detect_barcode_opencv.py --image images/barcode_01.jpg

# import the necessary packages
import numpy as np
import argparse
import cv2
import glob
from random import randint
from dynamosoft import *
from scandit import *
import pandas as pd
from pixel_2_world import *


def correction(tvecsP):
    xplot = []
    yplot = []
    X = []

    for ii in tvecsP:
        # print('inside correction',ii)
        for j in ii:
            # print('inside correction',ii)
            xplot.append(j[1])
    for ii in tvecsP:
        for j in ii:
            yplot.append(j[2])

    for y in range(len(xplot)):
        X.append([0, xplot[y], yplot[y]])

    X = np.matrix(X)
    th = -10
    T = np.matrix([[1, 0, 0],
                [0, np.cos(th*np.pi/180),
                -np.sin(th*np.pi/180)],
                [0, np.sin(th*np.pi/180),
                np.cos(th*np.pi/180)]])
    Y = X*T

    for i in range(len(tvecsP)):
        y = tvecsP[i][0][1]
        z = tvecsP[i][0][2]
        tvecsP[i][0][1] = Y[i, 1]
        tvecsP[i][0][2] = Y[i, 2]

    return tvecsP

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
    corners = []
    results = {}

    for j,cnt in enumerate(sort_cnts):
        if cv2.contourArea(cnt) > 2000:
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
                            corners = box_coord
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
                            corners = box_coord
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
                        corners = box_coord
                        ###################################################################################
                    else:
                        print("Scandit not able to decode")
                        text = "Null"
                        cv2.putText(image, text, (x, y + 250), cv2.FONT_HERSHEY_COMPLEX, 2.5, (255, 0, 0), 5)
                        ##################################################################################
                        box_id = text+'-'+str(j)
                        coord =[x-buf ,y-buf, x+w+buf,y-buf, x+w+buf,y+h+buf, x-buf,y+h+buf]
                        box_coord = [int(i/4) for i in coord]
                        # box_coord  = np.divide(coord, 4)  
                        results.update({box_id:box_coord})
                        corners = box_coord
                """
            #####################################################################################
            cv2.rectangle(image,(x-buf,y-buf),(x+w+buf,y+h+buf),(255,0,0),5)
            # cv2.rectangle(img,(int((x-buf)/4),int((y-buf)/4)),(int((x+w+buf)/4),int((y+h+buf)/4)),(255,0,0),5)
            #####################################################################################
            # sort_cnts_ext.append(cnt)
    image = cv2.resize(image, None, fx=1/scale, fy=1/scale, interpolation = cv2.INTER_CUBIC)
    # print("Final Results------------>",results)
    # print(corners)
    center = [int((corners[0]+corners[2]+corners[4]+corners[6])/4),int((corners[1]+corners[3]+corners[5]+corners[7])/4)]
    return image, inter_mediate,results, center

def pose_estimation(corners):
    # print(corners)
    cameraMatrix = [[878.33202015, 0, 485.74167328],
                        [0, 878.44704215, 323.28120842],
                        [0, 0, 1]]
    distCoeffs = [[0.13555811, -0.54607789, 0.00108346,
                    -0.00431513, 0.52654226]]
    cameraMatrix = np.array(cameraMatrix)
    distCoeffs = np.array(distCoeffs)
    img_points = np.array(corners,dtype=np.float64)

    img_points = img_points.reshape((4,2))
    # print(img_points)


    #########################################################################################
    # img_points = np.array([[607, 653],[653, 654],[653, 701],[606, 700]],dtype=np.float64)
    # img_points = img_points.reshape((4,2))
    #########################################################################################
    # print(img_points.shape)
    model_points =  np.array([[0.0,0.0,0.0],[10.0,0.0,0.0],[10.0,-4.0,0.0],[0.0,-4.0,0.0]],dtype=np.float64)
    # model_points =  np.array([[0.0,0.0,0.0],[7.0,0.0,0.0],[7.0,-7,0.0],[0.0,-7,0.0]],dtype=np.float64)
    model_points = model_points.reshape((4,3))
    # print(model_points.shape)
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, img_points, cameraMatrix, distCoeffs)
    tvecsP = np.array(translation_vector).reshape((1,1,3))
    ##############################################################
    xplot = []
    yplot = []
    X = []

    for ii in tvecsP:
        for j in ii:
            xplot.append(j[1])
    for ii in tvecsP:
        for j in ii:
            yplot.append(j[2])

    for y in range(len(xplot)):
        X.append([0, xplot[y], yplot[y]])

    X = np.matrix(X)
    th = 4
    T = np.matrix([[1, 0, 0],
                [0, np.cos(th*np.pi/180),
                -np.sin(th*np.pi/180)],
                [0, np.sin(th*np.pi/180),
                np.cos(th*np.pi/180)]])
    Y = X*T

    for i in range(len(tvecsP)):
        y = tvecsP[i][0][1]
        z = tvecsP[i][0][2]
        tvecsP[i][0][1] = Y[i, 1]
        tvecsP[i][0][2] = Y[i, 2]
    #############################################################
    # print(rotation_vector, translation_vector)
    ##########################################################################
    (first_end_point2D, jacobian) = cv2.projectPoints(np.array([(10.0, 0.0, 0.0)]), rotation_vector, translation_vector, cameraMatrix, distCoeffs)

    return rotation_vector, translation_vector, first_end_point2D,img_points,tvecsP


if __name__ == '__main__':
    img_files = glob.glob('datapose/datapose1/*.jpg')
    for img_file in img_files:
        name = img_file.split('/')[-1]
        depth_name = name.split('.')[0]+'.depth.png'
        depth_img = cv2.imread('datapose_1/datapose1/'+depth_name,-1)


        print('Depth image shape----',depth_img.shape)

        print('#################',img_file.split('/')[-1],'##################')

        image = cv2.imread(img_file)
        print('image shape---',image.shape)
        image_1, inter_mediate,results, center = detect_bar_code(image.copy(),"Scandit")
        print(center)

        depth_value_1 = get_box_depth(depth_img, center[0], center[1])

        # depth_value_2,dimg = get_box_depth(depth_img[:,:,2], center[1], center[0])

        # cv2.circle(image,(center[0], center[1]), 63, (0,0,255), -1)
        # cv2.circle(depth_img,(center[1], center[0]), 63, (0,0,255), -1)

        # cv2.imwrite("temp/depth.png",depth_img)
        # cv2.imwrite("temp/rgb.png",image)
        # cv2.imwrite("temp/dimg.png",dimg)



        xyz = calculate_XYZ(center[0], center[1],depth_value_1)
        print(xyz.reshape((1,1,3)).shape)
        xyz = correction(xyz.reshape((1,1,3)))
        print('world corrdinate of center--',(xyz/10).astype(np.float16))
        # print("Final Results------------>",results)
        # print(image.shape)
        # cv2.imwrite('Results/'+name,image)
        cv2.imwrite('inter_mediate/'+name,inter_mediate)





        ###############################################################################################################
        # rotation_vector, translation_vector,first_end_point2D,image_points,tvecsP = pose_estimation(corners)
        # print(translation_vector,'##############',tvecsP)
        # #################################################################
        # for p in image_points:
        #   cv2.circle(image, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
        # for i,p in enumerate(image_points):
        #   cv2.putText(image, str(i), (int(p[0]), int(p[1])), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        # # print(image_points)
        # p1 = ( int(image_points[1][0]), int(image_points[1][1]))
        # p2 = ( int(first_end_point2D[0][0][0]), int(first_end_point2D[0][0][1]))
        # cv2.line(image, p1, p2, (255,0,0), 2)
        ##########################################################################################################
        cv2.imwrite('Results/'+name,image)


        
        