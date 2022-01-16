import numpy as np
import cv2
import cv2.aruco as aruco

   
'''
    drawMarker(...)
        drawMarker(dictionary, id, sidePixels[, img[, borderBits]]) -> img
'''
 
aruco_dict = aruco.Dictionary_get(aruco.DICT_7X7_250) # boxes
#aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL) # navigation
#aruco_dict = aruco.Dictionary_get(aruco.DICT_7X7_1000) # navigation
#aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250) # pallets

# #For nav markers
# ids = []
# for i in range(1,9):
#     for j in range(1,7):
#         ids.append((100*i)+j)


print(aruco_dict)
# second parameter is id number
# last parameter is total image size
## for nav marker
# for i in ids:
for i in range(0,200):
    img = aruco.drawMarker(aruco_dict, i, 220)
    #img = aruco.drawMarker(aruco_dict, i, 700)

    #Put border around image
    h,w=img.shape[0:2]
    #base_size=h+44,w+44
    base_size=h+88,w+88
    base=np.zeros(base_size,dtype=np.uint8)
    #cv2.imshow('base',base)
    #cv2.waitKey(1)
   
    base[np.where((base == 0))] = 255
    #cv2.imshow('base',base)
    #cv2.waitKey(1)

    #cv2.imshow('img',img)
    #cv2.waitKey(0)
   
    #base[22:h+22,22:w+22]=img
    base[44:h+44,44:w+44]=img

    #Save image
    strr = "sku_"+str(i)+"_DICT_7X7_250.jpg"
    cv2.imwrite(strr, base)
 
#cv2.imshow('frame',base)
#cv2.waitKey(0)
#cv2.destroyAllWindows()