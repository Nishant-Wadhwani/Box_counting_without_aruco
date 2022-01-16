import os
import sys
from dbr import *
import cv2
import numpy as np
from pyzbar import pyzbar



def dynamosoft():
    reader = BarcodeReader()
    reader.init_license("t0069fQAAAGmLNgeYFjz3mbu95Ea52IhY4ISkZHT7DYS23O06xAtUa5oQE/PRvfBM2fTPr4bUucQJMLuZtRDXvh7gonXy857T")    
    settings = reader.get_runtime_settings()
    settings.barcode_format_ids = EnumBarcodeFormat.BF_ALL
    settings.barcode_format_ids_2 = EnumBarcodeFormat_2.BF2_POSTALCODE | EnumBarcodeFormat_2.BF2_DOTCODE
    settings.excepted_barcodes_count = 32
    reader.update_runtime_settings(settings)
    try:
        image = r"./temp/temp.png"
        text_results = reader.decode_file(image)
    except BarcodeReaderError as bre:
        print(bre)
    return text_results



def convert_3_channel(thresh):
    img = np.zeros([thresh.shape[0],thresh.shape[1],3],dtype=np.uint8)
    img[:,:,0] = thresh
    img[:,:,1] = thresh
    img[:,:,2] = thresh
    return img

def read_barcodes(img):
    filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img=cv2.filter2D(img,-1,filter)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    thresh1 = convert_3_channel(thresh1)
    barcodes = pyzbar.decode(thresh1)
    

    # # print(barcodes)
    # for barcode in barcodes:
    #     x, y , w, h = barcode.rect

    #     barcode_info = barcode.data.decode('utf-8')
    #     # cv2.rectangle(frame, (x, y),(x+w, y+h), (0, 255, 0), 2)
        
    #     font = cv2.FONT_HERSHEY_DUPLEX
    #     print(barcode_info)
    #     cv2.putText(frame, barcode_info, (x + 6, y - 6), font, 2.0, (0, 0, 255), 2)
    #     with open("barcode_result.txt", mode ='w') as file:
    #         file.write("Recognized Barcode:" + barcode_info)
    return barcodes