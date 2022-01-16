#!/usr/bin/env python

"""
OpenCV barcode scanner sample.

This example configures the SDK for a single image use case without any
resource restrictions.

PREREQUISITES:
    1. Copy ../public_api/python/scanditsdk.py locally
    2. Update LD_LIBRARY_PATH variable with a path to libscanditsdk.so library, 
       e.g. 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/$USER/scanditsdk/lib'
    3. Update SCANDIT_LICENSE_KEY variable with a license key

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import print_function
import sys
import tempfile
import scanditsdk as sc
import cv2

SCANDIT_LICENSE_KEY = "ARvAwisqCjK6Htq9kQ/asSQ0VGsHRkfIKWuedEBcb7FITrPtSyvb/Vdrf4nyBWUUPxEvZ8xaBom1Tjxm/iJ5zed1fUx6VMtVgHvme2gJwePrEzsGjj7+U7YBFmXLmJZ9uoTyEkYEJoAaemkf1SLN1x44jXjQBxeKW6shzpjwpTzxoG/7ZA/hC+IkNAzUrkoraiKF4ln510cCL81bl85msL3CzJGTLgSZ1pmIZ1zZAyGymJGaWrmTLAaXXoVGtoUV0dJPb7D8VYNTcA5swTSjvR18iqvv3a8kU1MX+GCjZnICuybLdhLrLOoD5HbcqYNf97P5EEne9YQ/ns0Vk+YV/IzuZoSEfUYb9G60Y65KE+ZON/2VgRp6yAZWCcX6K4SjXYoQQ75HfdF59Lw9BNF7sgNFTCo3yqvTqIoDMWgRal7afg/ZL98KGmhBVJ2T4u7ymsCH+zBKXDK6SeDSVBPqq1+uGI0IEqhlJEZ6o0n1tlbWa2uaZlnea3dzUatpnXH+lGLtp8W3Dh2W9yfpStL0PS51UESa9i1Pp8OzFk+K1QkcBWu3wX8AeNK8QuXFTjHVkVYemxTUaiHvmLns/IRRGW2yJCOY6aK0grqgbNM1t/P8pSMVHxQjq1YJuvdWNsd+bUehXSZSMrD6kHXarVAkyQvfYxoBgUeNEOVv6mNEWm/4klCeYC2ww/MTlyzZc3a7mqUVG3y33I8P3LTqVUar5+HZLdNJCjifg1ksK58ilkcWtRBHWGLF/1jrg0xo+4bAmwO1LfyU2/kf0+N5F3gq66u1M9JOHpwsehHUtbc="


def getOpenCvImageDataFromFile(image_path):
    # We convert the image to RGB (default for imread) for consistency.
    # Gray or any YUV formats are recommended if speed is critical.
    return cv2.imread(image_path)


def getScannerImageDescriptionFromOpenCvImage(opencv_image):
    # Describes dimensions as well as internal memory layout of an image buffer.
    # Depending on the image layout different properties must be set.
    # All image layouts require at least width, height and memory size.
    height = opencv_image.shape[0]
    width = opencv_image.shape[1]
    channel = opencv_image.shape[2]
    descr = sc.ImageDescription()
    descr.width = width
    descr.height = height
    descr.first_plane_row_bytes = channel * width
    descr.layout = sc.IMAGE_LAYOUT_RGB_8U
    descr.memory_size = channel * width * height
    return descr


def createRecognitionContext():
    # The recognition context requires a directory for caching and book-keeping reasons.
    # Here we use a temporary directory, but for production code, a directory that
    # persists reboots should be used as it improves efficiency of the application.
    data_directory = tempfile.gettempdir()
    context = sc.RecognitionContext(SCANDIT_LICENSE_KEY, data_directory)
    return context


def createScannerSettings():
    # Use the single frame preset, that switches the scanner from real-time video
    # stream processing to a single frame mode.
    # This preset should only be used on fast devices. Execution time per frame can be
    # up to a 100 times slower.
    settings = sc.BarcodeScannerSettings(preset=sc.PRESET_ENABLE_SINGLE_FRAME_MODE)

    # We assume the worst camera system (the camera cannot change its focus).
    settings.focus_mode = sc.SC_CAMERA_FOCUS_MODE_FIXED

    # We want to scan at most one code per frame.
    # Should be set to a higher number if false positives are likely.
    settings.max_number_of_codes_per_frame = 1

    # By default, 1d and 2d codes are searched in the full image.
    # The area can be defined but will not be used by the barcode scanner.
    settings.code_location_constraint_1d = sc.SC_CODE_LOCATION_IGNORE
    settings.code_location_constraint_2d = sc.SC_CODE_LOCATION_IGNORE

    # No assumptions about the code direction are made.
    settings.code_direction_hint = sc.SC_CODE_DIRECTION_NONE

    # Enable needed symbologies.
    settings.enable_symbology(sc.SYMBOLOGY_EAN13, True)
    settings.enable_symbology(sc.SYMBOLOGY_UPCA, True)
    settings.enable_symbology(sc.SYMBOLOGY_QR, True)
    settings.enable_symbology(sc.SYMBOLOGY_CODE128, True)

    # Change the EAN13 symbology settings to enable scanning of color-inverted barcodes.
    settings.symbologies[sc.SYMBOLOGY_EAN13].color_inverted_enabled = True
    return settings


def processOpenCvImage(frame_seq, cv_image_data):
    # Get an image description for barcode scanner.
    # print('66666666666666666666666666666666666666666666666666666666666')
    image_description = getScannerImageDescriptionFromOpenCvImage(cv_image_data)


    # Get an access to the image data through numpy-specific API.
    image_data = cv_image_data.__array_interface__["data"][0]


    # Process the barcode scanning.
    z = frame_seq.process_frame(image_description, image_data)
    # print('66666666666666666666666666666666666666666666666666666666666')
    return z


# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python3 OpenCvPySample.py path_to_an_image_file")
#         exit(1)

#     print("OpenCV barcode scanner sample for a single image use case")
#     print("Version 1.0")

#     # Get an OpenCV image from the file.
#     image_path = sys.argv[1]
#     opencv_image_data = getOpenCvImageDataFromFile(image_path)

#     # Create a recognition context.
#     recognition_context = createRecognitionContext()

#     # Create a scanner settings.
#     scanner_settings = createScannerSettings()

#     # Create a scanner and start the scanning.
#     scanner = sc.BarcodeScanner(recognition_context, scanner_settings)
#     scanner.wait_for_setup_completed()

#     frame_seq = recognition_context.start_new_frame_sequence()
#     scanning_status = processOpenCvImage(frame_seq, opencv_image_data)
#     frame_seq.end()
#     # print('%%%%%%%%%%%%%%%%%%',scanning_status.status, sc.RECOGNITION_CONTEXT_STATUS_SUCCESS)
#     if scanning_status.status != sc.RECOGNITION_CONTEXT_STATUS_SUCCESS:
#         print(
#             "Processing frame failed with code {}: {}".format(
#                 scanning_status.status, scanning_status.get_status_flag_message()
#             )
#         )
#         exit(2)

#     codes = scanner.session.newly_recognized_codes
#     # print(codes.location,/n,codes.data,/n,codes.symbology_string)
#     if len(codes):
#         for code in codes:
#             print(code.location,'\n',code.data,'\n',code.symbology_string)
#             print(
#                 "Barcode found at location ",
#                 code.location,
#                 ": ",
#                 code.data,
#                 " (",
#                 code.symbology_string,
#                 ")",
#                 sep="",
#             )
#     else:
#         print("No barcode found.")





def scandit(opencv_image_data):
	# recognition_context = createRecognitionContext()
	recognition_context = createRecognitionContext()
	# Create a scanner settings.
	scanner_settings = createScannerSettings()

	# Create a scanner and start the scanning.
	scanner = sc.BarcodeScanner(recognition_context, scanner_settings)
	scanner.wait_for_setup_completed()

	frame_seq = recognition_context.start_new_frame_sequence()
	# print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&7')
	scanning_status = processOpenCvImage(frame_seq, opencv_image_data)
	# print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&7')

	frame_seq.end()
	# print('%%%%%%%%%%%%%%%%%%',scanning_status.status, sc.RECOGNITION_CONTEXT_STATUS_SUCCESS)
	if scanning_status.status != sc.RECOGNITION_CONTEXT_STATUS_SUCCESS:
	    print(
	        "Processing frame failed with code {}: {}".format(
	            scanning_status.status, scanning_status.get_status_flag_message()
	        )
	    )
	    exit(2)

	codes = scanner.session.newly_recognized_codes

	return codes

if __name__ == '__main__':
	img = cv2.imread('img_5.png')
	codes = scandit(img)
	print('########',codes[0].location,'\n',codes[0].data,'\n',codes[0].symbology_string)



