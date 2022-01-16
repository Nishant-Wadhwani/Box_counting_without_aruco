import os
import cv2
import math
import json
import time
import pickle
import imutils
import subprocess
import collections
import numpy as np
import cv2.aruco as aruco
import matplotlib.pyplot as plt
from datetime import timedelta
from scipy.spatial import distance as dist
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
import glob
import detect_barcode_opencv_exp
import csv
import pandas as pd


# Get pallet info
PALLET_WIDTH = 121.92  # In cm
PALLET_BREADTH = 121.92  # In cm

# Get aruco info
PALLET_MARKERLENGTH = 0.063  # 7 cm
pallet_info = [PALLET_WIDTH, PALLET_BREADTH]
# Get aruco info
SKU_MARKERLENGTH = 0.07  # 7 cm
PALLET_MARKERLENGTH = 0.063  # 7 cm
aruco_info = [SKU_MARKERLENGTH, PALLET_MARKERLENGTH]


class Camera_Info:

    """

        Camera Specific Information based on the configuration

    """

    def __init__(self):

        self.cameraMatrix = [[1367.27 , 0, 950.995],
                             [0, 1365.57, 564.756],
                             [0, 0, 1]]
        self.distCoeffs = [[0.13555811, -0.54607789, 0.00108346,
                            -0.00431513, 0.52654226]]
        self.cameraMatrix = np.array(self.cameraMatrix)
        self.distCoeffs = np.array(self.distCoeffs)
        self.camera_matrix = [self.cameraMatrix, self.distCoeffs]


class Dictionary_Marker:

    """

        Dictionary Marker Specific Information based on the id

    """

    def __init__(self):

        self.aruco_dict_sku = aruco.Dictionary_get(
                               aruco.DICT_5X5_1000)  # SKU dictionary
        self.aruco_dict_pallet = aruco.Dictionary_get(
                               aruco.DICT_6X6_1000)  # Pallet dictionary
        # self.aruco_dict_rack = aruco.Dictionary_get(
        # aruco.DICT_7X7_1000)
        self.parameters = aruco.DetectorParameters_create()
        self.parameters = aruco.DetectorParameters_create()
        self.parameters.perspectiveRemoveIgnoredMarginPerCell = 0.4
        self.parameters.maxErroneousBitsInBorderRate = 0.5
        self.parameters.errorCorrectionRate = 0.75
        self.parameters.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
        self.parameters.cornerRefinementMaxIterations = 30
        self.parameters.cornerRefinementMinAccuracy = 0.1


class Sku_Stat:

    """

        SKU Specific Information that will be
        retrieved from database containing
        sku id

    """

    def __init__(self):

        self.box_width = []
        self.box_length = []
        self.box_height = []
        self.arranging_pattern1 = []
        self.arranging_pattern1_originx = []
        self.arranging_pattern1_originy = []
        self.length_wise_max_level1 = []
        self.width_wise_max_level1 = []
        self.arranging_pattern2 = []
        self.arranging_pattern2_originx = []
        self.arranging_pattern2_originy = []
        self.length_wise_max_level2 = []
        self.width_wise_max_level2 = []
        self.max_height = []


class Pallet_Marker:

    """

        Pallet Marker Specific Information that will be
        retrieved from image itself

    """

    def __init__(self):

        self.idsP = []
        self.cornersP = []
        self.pallet_aruco_depth_tvec = []
        self.pallet_aruco_height_tvec = []
        self.pallet_aruco_distance_tvec = []


class Box_Marker:

    """

        Box Reference Point Specific Information that will be
        retrieved from the image itself using Box_Detection
        Model

    """

    def __init__(self):

        self.cornersS = []
        self.L_box_ref_moved = []
        # self.rvecsS = []
        self.tvecsS = []
        self.unique_ids = []


class Box_Pos_Info:

    """

        Box 3-D  Specific Information wrt Pallet Marker that will be
        retrieved from the image itself using box_tvecsS module

    """

    def __init__(self):

        self.boxes_details = []
        self.im_clust_depth = []
        self.im_clust_height = []
        self.im_clust_distance = []
        self.im_point_depth = []
        self.im_point_height = []
        self.im_point_distance = []
        self.im_pointR = []


def get_box_depth(dimg, x, y):

    """

        Description:-
        Calculate the depth for a particular bounding box's centre patch

        Arguments:-
        dimg:- depth image
        x:- x_center
        y:- y_center

        Returns depth in cm's in float

    """

    # take a small patch of 12*12 around the centre pixel of the box
    patch = dimg[y-6:y+6, x-6:x+6]
    sum_patch = np.sum(patch)
    elements_patch = np.count_nonzero(patch)
    avg_depth_patch = sum_patch/elements_patch
    z_cm = np.around((avg_depth_patch/10), decimals=2)
    return(z_cm)  # returns in cm upto 2 decimal places


def get_center(corner):

    """

        Description:-
        Trying to average all the four coordinates
        in order to get centre of a region

        Arguments:-
        corner:- List containing corner points

        Returns center of provided rectangle

    """

    x1 = corner[0][0][0]
    y1 = corner[0][0][1]

    x2 = corner[0][1][0]
    y2 = corner[0][1][1]

    x3 = corner[0][2][0]
    y3 = corner[0][2][1]

    x4 = corner[0][3][0]
    y4 = corner[0][3][1]

    x = int((x1+x2+x3+x4)/4)
    y = int((y1+y2+y3+y4)/4)
    # This is in pixels
    return (x, y)


def distance_based_clustering(im_point, im_clust, thres):

    """

        Description:-
        This function clusters data based on euclidean distance

        Arguments:-
        im_point - set of points
        im_clust - point initial flag values
        thres - distance threshold

        Returns the cluster number to which particular image point lies.

    """

    # Process im_point
    im_point = im_point[0]

    # Convert points to numpy array
    clust = np.array(im_clust)
    lenn = len(clust)

    # Actual clustering get done here
    for i in range(lenn):

        if clust[i] == -1:
            # If this particular point is
            # not assigned to any cluster,
            # create new cluster point
            clust[i] = i

        for j in range(lenn):
            # Compare the euclidean distance
            # between ith and jth point

            # Get distance between two points
            ecu_dist = np.sqrt((im_point[i][0] - im_point[j][0])**2 +
                               (((im_point[i][1] - im_point[j][1]))**2))

            if (ecu_dist) < thres:
                # Merge points, if distance is less than threshold value

                if clust[j] == -1:
                    # If jth point is not part of any other cluster,
                    # just merge it into the ith cluster
                    clust[j] = clust[i]

                else:
                    # Otherwise merge entire cluster to which jth point
                    # belong to ith cluster
                    ind = np.where((clust == clust[j]))
                    clust[ind] = clust[i]

    return clust


def get_unique_pixels(clusters, tvecs, index):

    """

        Description:-
        Getting unique set of pixels through unique
        cluster numbers

        Args:
        clusters - Clustering function output
        tvecs - Boxes position information
        index - Index to get desired value among (x, y, z)

        Returns unique set of pixels

    """

    # Convert output to numpy array
    clusters = np.array(clusters)

    # Get unique cluster numbers
    unique_pixels = np.vstack({tuple(r) for r in clusters.reshape(-1, 1)})

    list_sum = []
    for val in unique_pixels:

        # Get all points belong to this cluster
        pts = np.where((clusters == val))

        summ = 0
        count = 0

        # Mark boxes
        for pt in pts[0]:

            summ = summ + tvecs[pt][0][index]*100
            count = count+1

        summ = summ/count

        list_sum.append(summ)

    unique_pixels = list(unique_pixels)

    zipped_pairs = zip(list_sum, unique_pixels)
    unique_pixels = [x for _, x in sorted(zipped_pairs)]

    return unique_pixels


def generate_unique_color_for_each_box(length):

    """

        Description:-
        Generate unique color for each box,
        useful for generating level wise image output

        Arguments:-
        length - Number of color required

        Returns color list

    """

    colorListForBoxes = []

    for val in range(0, length):

        # Generate random color
        color = [0, 0, 0]

        while(color == [0, 0, 0] or color == [255, 255, 255]):
            color = list(np.random.choice(range(256), size=3))

        colorListForBoxes.append(color)

    return colorListForBoxes


def drawBoxes(img,
              box_length,
              box_width,
              arranging_pattern,
              arranging_pattern_originx,
              arranging_pattern_originy,
              colorListForBoxes):

    """

        Description:-
        Draw boxes stacking pattern

        Arguments:-
        img:- blank input image
        box_length:- length of the box in centi-meter
        box_width:- width of the box in centi-meter
        arranging_pattern:- tells how boxes are arranged on the pallet
        arranging_pattern_originx:- list of "x" co-ordinate of
                                    (left - down) corner of each box
        arranging_pattern_originy:- list of "y" co-ordinate of
                                    (left - down) corner of each box
        colorListForBoxes:- list of color, each color repersent one box
                            on floor image

        Returns the image with stacking pattern of boxes

    """

    i = 0

    for letter in arranging_pattern:

        xPos = arranging_pattern_originy[i]
        yPos = arranging_pattern_originx[i]

        # print "letter = ", letter

        if letter == "L":

            x1 = int(xPos)
            y1 = int(yPos)
            x2 = int(xPos) + int(round(box_width))
            y2 = int(yPos) + int(round(box_length))

            cv2.rectangle(img, (x1, y1), (x2, y2), colorListForBoxes[i], -1)
            # cv2.putText(img,str(i),(x1 +10, y1+10), font, 0.5, (0, 0, 0), 1)

        if letter == "B":

            x1 = int(xPos)
            y1 = int(yPos)
            x2 = int(xPos) + int(round(box_length))
            y2 = int(yPos) + int(round(box_width))

            cv2.rectangle(img, (x1, y1), (x2, y2), colorListForBoxes[i], -1)
            # cv2.putText(img,str(i),(x1+10, y1+10), font, 0.5, (0, 0, 0), 1)

        i = i + 1

    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return img


def draw_boxes_m_fix(img,
                     box_length,
                     box_width,
                     arranging_pattern,
                     arranging_pattern_originx,
                     arranging_pattern_originy,
                     colorListForBoxes,
                     m_fact):

    """

        Description:-
        Draw boxes stacking pattern by taking magnifying
        factor into account.

        Arguments:
        img - blank input image
        box_length - length of the box in centi-meter
        box_width - width of the box in centi-meter
        arranging_pattern - List that tells how boxes
                            are arranged on the pallet
        arranging_pattern_originx - list of "x" co-ordinate of
                                    (left - down) corner of each box
        arranging_pattern_originy - list of "y" co-ordinate of
                                    (left - down) corner of each box
        colorListForBoxes - list of color, each color repersent one box
                            on floor image
        m_fact - Magnifying factor

        Returns boxes stacking pattern image

    """

    # Variabel to keep track of number of boxes
    i = 0

    # Draw each boxes one by one on blank image
    for letter in arranging_pattern:

        # Get the position information of each box
        xPos = arranging_pattern_originy[i] * m_fact
        yPos = arranging_pattern_originx[i] * m_fact

        # print "letter = ", letter

        # Get color
        color = colorListForBoxes[i]
        color = np.int32(color)
        color = color.tolist()
        # print ("color = ", color)

        if letter == "L":
            # Get "top - left" and "right - bottom"
            # info of box kept length wise

            x1 = int(xPos)
            y1 = int(yPos)
            x2 = int(xPos) + int(round(box_width * m_fact))
            y2 = int(yPos) + int(round(box_length * m_fact))

            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            cv2.putText(img, str(i), (x1 + 10, y1 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 1)

        if letter == "B":
            # Get "top - left" and "right - bottom"
            # info of box kept breadth wise

            x1 = int(xPos)
            y1 = int(yPos)
            x2 = int(xPos) + int(round(box_length * m_fact))
            y2 = int(yPos) + int(round(box_width * m_fact))

            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            cv2.putText(img, str(i), (x1 + 10, y1 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 1)

        i = i + 1

    # Rotate the image
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return img


def get_box_count(tempList,
                  sku,
                  value,
                  level,
                  flag,
                  score_matrix,
                  level_details,
                  box_list_all):

    """

        Description:-
        Getting Count of boxes category wise i.e
        visible boxes, hidden boxes and absent boxes
        by taking sku specific information like id for
        consideration.

        Arguments:-
        tempList:- List specifying category of boxes
        sku:- Object of Sku_Stat which contains sku specific information
        value:- List specifying pre-defined category of boxes
        level:- Level of Boxes laying on the pallet
        flag:- Flag which is specific to stacking pattern
        score_matrix:- Numpy Array which stores confidence score pertaining to
        each sku
        level_details:- Dictionary containing level details
        box_list_all:- List conatining Dictionaries which stores all the
        relevant information related to each sku like category,
        location_id, depth_error, height_error, distance_error

        Returns count of specified category of boxes

    """

    # Variables to store results
    zerosListX = []
    zerosListY = []
    box_details_all = {}

    # Get list of pecified category of boxes
    indexes = np.where((tempList == value))

    # Get score matrix
    score_matrix = np.array(score_matrix)

    # Visible boxes
    if value == 1:

        for i in range(0, len(indexes[0])):

            box_details = {}

            iddd = indexes[0][i]

            row = score_matrix[np.where(score_matrix[:, 1] == float(iddd))]
            row = row[np.where(row[:, 0] == float(level))]

            strr = "Level_" + str(level) + "_Box_" + str(iddd)

            val = [float(row[0][0]),
                   float(row[0][1]),
                   "V",
                   float(round(row[0][2], 2)),
                   float(round(row[0][3], 2)),
                   float(round(row[0][4], 2)),
                   float(round(row[0][5], 2))]

            box_details['Box_Location_ID'] = int(row[0][1])
            box_details['Box_Status'] = "V"
            box_details['Overall_Error_Score'] = float(round(row[0][2], 2))
            box_details['Height_Error'] = float(round(row[0][3], 2))
            box_details['Distance_Error'] = float(round(row[0][4], 2))
            box_details['Depth_Error'] = float(round(row[0][5], 2))

            strr = "Box_" + str(iddd)
            box_details_all[strr] = box_details

            box_list_all.append(box_details)

    # Hidden boxes
    if value == 2:

        for i in range(0, len(indexes[0])):

            box_details = {}

            iddd = indexes[0][i]

            strr = "Level_" + str(level) + "_Box_" + str(iddd)
            val = [float(level), float(iddd), "H"]

            box_details['Box_Location_ID'] = int(iddd)
            box_details['Box_Status'] = "H"

            strr = "Box_" + str(iddd)
            box_details_all[strr] = box_details

            box_list_all.append(box_details)

    # Absent boxes
    if value == 0:

        for i in range(0, len(indexes[0])):

            box_details = {}

            iddd = indexes[0][i]

            strr = "Level_" + str(level) + "_Box_" + str(iddd)
            val = [float(level), float(iddd), "A"]

            box_details['Box_Location_ID'] = int(iddd)
            box_details['Box_Status'] = "A"

            strr = "Box_" + str(iddd)
            box_details_all[strr] = box_details

            box_list_all.append(box_details)

    for i in range(0, len(indexes[0])):

        iddd = indexes[0][i]

        if flag == "odd":

            if level % 2 == 0:

                if sku.arranging_pattern2[iddd] == "L":
                    x = sku.arranging_pattern2_originx[iddd]+(sku.box_length/2)
                    y = sku.arranging_pattern2_originy[iddd]+(sku.box_width/2)
                if sku.arranging_pattern2[iddd] == "B":
                    x = sku.arranging_pattern2_originx[iddd]+(sku.box_width/2)
                    y = sku.arranging_pattern2_originy[iddd]+(sku.box_length/2)

            else:

                if sku.arranging_pattern1[iddd] == "L":
                    x = sku.arranging_pattern1_originx[iddd]+(sku.box_length/2)
                    y = sku.arranging_pattern1_originy[iddd]+(sku.box_width/2)
                if sku.arranging_pattern1[iddd] == "B":
                    x = sku.arranging_pattern1_originx[iddd]+(sku.box_width/2)
                    y = sku.arranging_pattern1_originy[iddd]+(sku.box_length/2)

        if flag == "even":

            if level % 2 == 0:

                if sku.arranging_pattern1[iddd] == "L":
                    x = sku.arranging_pattern1_originx[iddd]+(sku.box_length/2)
                    y = sku.arranging_pattern1_originy[iddd]+(sku.box_width/2)
                if sku.arranging_pattern1[iddd] == "B":
                    x = sku.arranging_pattern1_originx[iddd]+(sku.box_width/2)
                    y = sku.arranging_pattern1_originy[iddd]+(sku.box_length/2)

            else:

                if sku.arranging_pattern2[iddd] == "L":
                    x = sku.arranging_pattern2_originx[iddd]+(sku.box_length/2)
                    y = sku.arranging_pattern2_originy[iddd]+(sku.box_width/2)
                if sku.arranging_pattern2[iddd] == "B":
                    x = sku.arranging_pattern2_originx[iddd]+(sku.box_width/2)
                    y = sku.arranging_pattern2_originy[iddd]+(sku.box_length/2)

        zerosListX.append(x)
        zerosListY.append(y)

    return(zerosListX, zerosListY, len(indexes[0]), box_details_all)


def draw_plot(plt,
              arranging_pattern,
              arranging_pattern_originx,
              arranging_pattern_originy,
              box_length,
              box_width):

    """

        Description:-
        Plots stacking patterns

        Arguments:
        plt - matplotlib.pyplot module
        box_length - length of the box in centi-meter
        box_width - width of the box in centi-meter
        arranging_pattern - List that tells how boxes
                            are arranged on the pallet
        arranging_pattern_originx - list of "x" co-ordinate of
                                    (left - down) corner of each box
        arranging_pattern_originy - list of "y" co-ordinate of
                                    (left - down) corner of each box


    """

    i = 0
    xPos = 0
    yPos = 0

    for letter in arranging_pattern:

        xPos = arranging_pattern_originx[i]
        yPos = arranging_pattern_originy[i]

        if letter == "L":

            plt.gca().add_patch(Rectangle(
                (xPos, yPos),
                box_length,
                box_width,
                linewidth=1,
                edgecolor='r',
                facecolor='none'))

        if letter == "B":

            plt.gca().add_patch(Rectangle(
                (xPos, yPos),
                box_width,
                box_length,
                linewidth=1,
                edgecolor='r',
                facecolor='none'))

        i = i + 1


def get_complete_length(arranging_pattern,
                        arranging_pattern_originx,
                        arranging_pattern_originy,
                        length_wise_max,
                        width_wise_max,
                        box_length,
                        box_width):

    """

        Description:-
        Gives dimension for top view image for calculation
        of errors which will be useful in deciding flag value
        for box counting.

        Arguments:-
        arranging_pattern - Boxes arranging pattern
        arranging_pattern_originx - "x" co-ordinate of boxes
        arranging_pattern_originy - "y" co-ordinate of boxes
        length_wise_max - Number of boxes visible from front
        width_wise_max - Number of boxes visible from side
        box_length - Length of the box
        box_width - Width of the box

        Returns dimension for top view image

    """

    dim_length = 0
    dim_width = 0

    count = 1

    dim_length = arranging_pattern_originx[length_wise_max +
                                           width_wise_max - 2]
    dim_width = arranging_pattern_originy[length_wise_max +
                                          width_wise_max - 2]

    if arranging_pattern[length_wise_max + width_wise_max - 2] == "L":

        dim_length = dim_length + box_length
        dim_width = dim_width + box_width
    else:

        dim_length = dim_length + box_width
        dim_width = dim_width + box_length

    return (dim_length, dim_width)


def read_dict(box_id):

    """"

        Description:-
        Read from dictionary - loading data from database

        Arguments:
        box_id - SKU id

        Return Box information pertaining to
        specific sku id

    """

    # list to store all sku info from database
    # in the order of identified sku ids
    sku_info = []

    # Open pickel file
    with open("box_count/tools/Dictionary/sku_database_1.pkl", "rb") as pk:
        sku_db = pickle.loads(pk.read())

    # Get the info only for identified SKUs from database
    for i in range(0, 1):
        sku_id = 'sku id ' + str(box_id)
        sku_id = sku_db[sku_id]
        sku_info.append(sku_id)

    # if more than one sku id detected
    for i in range(0, 1):
        box_width = sku_info[i]['box_width']
        box_length = sku_info[i]['box_length']
        box_height = sku_info[i]['box_height']
        W = sku_info[i]['W']
        L = sku_info[i]['L']
        arranging_pattern1 = \
            sku_info[i]['arranging_pattern1']
        arranging_pattern1_originx = \
            sku_info[i]['arranging_pattern1_originx']
        arranging_pattern1_originy = \
            sku_info[i]['arranging_pattern1_originy']
        length_wise_max_level1 = \
            sku_info[i]['length_wise_max_level1']
        width_wise_max_level1 = \
            sku_info[i]['width_wise_max_level1']
        arranging_pattern2 = \
            sku_info[i]['arranging_pattern2']
        arranging_pattern2_originx = \
            sku_info[i]['arranging_pattern2_originx']
        arranging_pattern2_originy = \
            sku_info[i]['arranging_pattern2_originy']
        length_wise_max_level2 = \
            sku_info[i]['length_wise_max_level2']
        width_wise_max_level2 = \
            sku_info[i]['width_wise_max_level2']
        max_height = sku_info[i]['max_height']

        # For only the arranging patterns
        arr = arranging_pattern1_originx
        arr_new = []

        for i in range(0, len(arr)):
            foo = eval(arr[i])
            arr_new.append(foo)
        arranging_pattern1_originx = arr_new

        arr = arranging_pattern1_originy
        arr_new = []

        for i in range(0, len(arr)):
            foo = eval(arr[i])
            arr_new.append(foo)
        arranging_pattern1_originy = arr_new

        arr = arranging_pattern2_originx
        arr_new = []

        for i in range(0, len(arr)):
            foo = eval(arr[i])
            arr_new.append(foo)
        arranging_pattern2_originx = arr_new

        arr = arranging_pattern2_originy
        arr_new = []

        for i in range(0, len(arr)):
            foo = eval(arr[i])
            arr_new.append(foo)
        arranging_pattern2_originy = arr_new

    box_info = [box_width,
                box_length,
                box_height,
                arranging_pattern1,
                arranging_pattern1_originx,
                arranging_pattern1_originy,
                length_wise_max_level1,
                width_wise_max_level1,
                arranging_pattern2,
                arranging_pattern2_originx,
                arranging_pattern2_originy,
                length_wise_max_level2,
                width_wise_max_level2,
                max_height]

    return box_info


def get_x_and_y_method1(x_val,
                        y_val,
                        dim_width,
                        m_fact):

    """

        Description:-
        Calculate pallet adjusted co-ordinate values

        Arguments:-
        x_val:- x value
        y_val:- y value
        dim_width:- Width of top-view image
        m_fact:- Magnifying Factor

        Returns pallet adjusted co-ordinate values

    """

    # Calculate co-ordinates values
    y = int((dim_width * m_fact)/2 + (x_val * m_fact))
    x = int(round(dim_width)) - int(math.ceil(abs(y_val)))

    # Adjust values
    y = y - 1
    x = abs(x) * m_fact - 1

    return (y, x)


def process_pallet_markers(gray,
                           Dictionary_Marker,
                           Camera_Info,
                           file_handler,
                           details_dict):

    """

        Description:-
        Calculate pallet position information both in pixels(Image wise)
        and in cms(wrt camera) by using cv2.arucomarker
        library and correction matrix into account

        Arguments:
        gray - gray scale input image
        Dictionary_Marker - Pallet marker dictionary
        Camera_Info - Camera matrix
        file_handler - File handler to store error messages
        details_dict - Json file

        Returns pallet position information

    """
    # Variable to store pallet specific information
    pallet = Pallet_Marker()

    # Get the lists of pallet ids and the corners
    cornersP, idsP, rejectedImgPointsP = aruco.detectMarkers(
         gray,
         Dictionary_Marker.aruco_dict_pallet,
         parameters=Dictionary_Marker.parameters)

    # print(cornersP)

    # Draw detected markers
    gray = aruco.drawDetectedMarkers(gray, cornersP, idsP)

    # Display image
    """plt.imshow(gray)
    plt.show()"""

    # Get number of unique pallet ids detected
    length_pallet = len(cornersP)

    if length_pallet == 0:
        # Condition to check if any pallet is
        # present in image or not

        # print("Error 1 - No pallet detected")

        # Write error message to the output file
        strr = "Error 1 - No pallet detected\n"
        file_handler.write(strr)

        return (None)

    if length_pallet > 1:
        # Condition to check if multiple pallets
        # are present in image or not
        # print("Error 2 - Multiple pallets detected")

        # Write error message to the output file
        strr = "Error 2 - Multiple pallets detected\n"
        file_handler.write(strr)

        return (None)

    # Get the id of detected pallet
    details_dict['pallet_id'] = int(idsP[0][0])

    # Get rvecsP + tvecsP - Get pallet pos info
    (rvecsP, tvecsP, _) = aruco.estimatePoseSingleMarkers(
         cornersP,
         PALLET_MARKERLENGTH,
         Camera_Info.cameraMatrix,
         Camera_Info.distCoeffs)
    print("THis is function inside:- ",tvecsP)

    # Correct points
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

    # Extract pallet marker pos (depth, height and distance) info
    # We have calculated the tvecsP
    tvecsP = tvecsP*100
    pallet.pallet_aruco_depth_tvec = tvecsP[0][0][2]
    pallet.pallet_aruco_height_tvec = tvecsP[0][0][1]
    pallet.pallet_aruco_distance_tvec = tvecsP[0][0][0]
    pallet.cornersP = cornersP
    pallet.idsP = idsP

    # Visualize detected pallet on an image
    for corner in cornersP:
        # Get center of each box and plot on image
        (x, y) = get_center(corner)
        gray[y-20:y+20, x-20:x+20] = 255

    return (pallet)


def process_box_markers(i,
                        file_handler,
                        details_dict,
                        sku_ref_pts_list, cornersS_final_list,
                        L_box_ref_moved_list, barcode_id, SKU_name):

    """

        Description:-
        Gives boxes position information wrt camera as well as image wise
        by using box_tvecsS
        which is inculcating pallet marker position information both
        in pixels and in cms as well as camera matrix configuration

        Arguments:
        i:- Counter  that will keep track of images send to a particular module
        file_handler:- File handler to store error messages
        details_dict:- Json file
        out_dir:- Output Directory for saving the output
                  result of box_detection module
        cornersS_final_list:- List which will
                           store the box corner coordinates of true boxes
        sku_ref_pts_list:- List which will contain
                           tvecsS of all the boxes
        L_box_ref_moved_list:- List which will
                               store the box ref moved up coordinates
                               and a depth value

        Returns boxes position information

    """

    # Variable to store boxes specific information
    cornersS = []
    L_box_ref_moved = []
    idsS = []
    # rejectedImgPointsS = []
    box = Box_Marker()

    # Get Lists of ids and the corners
    idsS = [barcode_id]
    details_dict['sku_code'] = barcode_id
    details_dict['sku_name'] = SKU_name


    (tvecsS, cornersS, L_box_ref_moved) =  \
        (sku_ref_pts_list[i], cornersS_final_list[i], L_box_ref_moved_list[i])
    tvecsS = tvecsS/100
    # Draw detected markers
    # gray = aruco.drawDetectedMarkers(gray, cornersS, idsS)
    # cornersS = corners_modified
    # gray = aruco.drawDetectedMarkers(gray, rejectedImgPointsS)

    """
    cv2.namedWindow("Pallet Aruco", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Pallet Aruco", 600,600)
    cv2.imshow("Pallet Aruco", gray)
    cv2.waitKey(0)
    """

    # Check if any box is detected
    if len(cornersS) == 0:

        strr = str(0)
        file_handler.write(strr)

        return (None)

    # Update "json" file
    details_dict['No_of_visible_boxes'] = int(len(cornersS))

    # Get list of unique box id numbers
    id_sku_list = list(idsS)
    id_sku_list = np.array(id_sku_list)
    unique_ids = np.vstack({tuple(r) for r in id_sku_list.reshape(-1, 1)})

    # If none of the box ids are detected
    if len(unique_ids) == 0:

        strr = "Total Count = 0\n"
        file_handler.write(strr)

        return (None)

    # If more than on type of boxes are detected
    if len(unique_ids) > 1:

        strr = "Error 3 - Multiple ids detected for boxes\n"
        file_handler.write(strr)

        return (None)

    # Assigne all these information to the box object

    box.cornersS = cornersS
    # box.rvecsS = rvecsS
    box.tvecsS = tvecsS
    box.unique_ids = unique_ids
    box.L_box_ref_moved = L_box_ref_moved

    return (box)


def read_sku_info_from_dictionary(unique_ids, file_handler):
    """

        Description:-
        Gives sku specific information by reading id from
        the dictionary database of sku's.

        Arguments:
        Box_Marker.unique_ids - List of unique box ids
        file_handler - File handler to store error messages

        Returns sku specific information

    """

    # Variable to store sku specific information
    sku = Sku_Stat()

    # Read box specific information from dictionary
    try:
        box_info = read_dict(unique_ids[0][0])

    except:
        strr = ("Dictionary not found for id " +
                str(unique_ids[0][0]))
        file_handler.write(strr)

        print(strr)

        return (None)

    # Get box specific info
    sku.box_width = box_info[0]
    sku.box_length = box_info[1]
    sku.box_height = box_info[2]
    sku.arranging_pattern1 = box_info[3]
    sku.arranging_pattern1_originx = box_info[4]
    sku.arranging_pattern1_originy = box_info[5]
    sku.length_wise_max_level1 = box_info[6]
    sku.width_wise_max_level1 = box_info[7]
    sku.arranging_pattern2 = box_info[8]
    sku.arranging_pattern2_originx = box_info[9]
    sku.arranging_pattern2_originy = box_info[10]
    sku.length_wise_max_level2 = box_info[11]
    sku.width_wise_max_level2 = box_info[12]
    sku.max_height = box_info[13]

    return (sku)


def get_box_pos_information(frame,
                            Box_Marker,
                            Pallet_Marker,
                            Sku_Stat):

    """

        Description:-
        Calculates 3D position information of each box
        wrt pallet marker

        Arguments:
        frame - Input image
        Box_Marker - Boxes 3D position information
        Pallet_Marker - Pallet 3D position information
        Sku_Stat - SKU details

        Returns 3D position information of each box

    """

    # Create object of "Box_Pos_Info" class
    pos = Box_Pos_Info()

    # Declare variabel to store all statistics related to visible boxes
    pos.boxes_details = np.zeros((len(Box_Marker.cornersS), 7),
                                 dtype="float32")

    # Start with first box marker
    j = 0
    m = 0

    for corner in Box_Marker.cornersS:

        # This is needed at the time of clustering
        pos.im_clust_depth.append(-1)  # For each point set flag -1
        pos.im_clust_height.append(-1)  # For each point set flag -1
        pos.im_clust_distance.append(-1)  # For each point set flag -1
        # Get center of each box
        x = Box_Marker.L_box_ref_moved[m][0]
        y = Box_Marker.L_box_ref_moved[m][1]
        # Get box height (y)
        box_height_detected = round(
            abs(
                Box_Marker.tvecsS[j][0][1]*100 -
                Pallet_Marker.pallet_aruco_height_tvec
                ), 2)
        pos.boxes_details[j][1] = box_height_detected

        # Get box depth (z)
        box_depth_detected = (Box_Marker.tvecsS[j][0][2] * 100 -
                              Pallet_Marker.pallet_aruco_depth_tvec) + \
                              round(Sku_Stat.box_width/4, 2)

        pos.boxes_details[j][0] = box_depth_detected

        # Get box distance (x)
        box_distance_detected = round(
            Box_Marker.tvecsS[j][0][0]*100 -
            Pallet_Marker.pallet_aruco_distance_tvec, 0)
        pos.boxes_details[j][2] = box_distance_detected

        # Write all these values on an input image
        cv2.putText(frame, str(j+1), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 2)
        cv2.putText(frame, str(round(box_depth_detected, 2)),
                    (x, y+20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 2)
        cv2.putText(frame, str(round(box_height_detected, 2)),
                    (x, y+40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 2)
        cv2.putText(frame, str(round(box_distance_detected, 2)),
                    (x, y+60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 2)

        # This is needed at the time of clustering
        pos.im_point_depth.append((box_depth_detected, 0))
        pos.im_point_height.append((box_height_detected, 0))
        pos.im_point_distance.append((box_distance_detected, 0))

        # This is needed to get each SKU's aruco info
        pos.im_pointR.append((x, y))

        j = j+1
        m = m+1

    """
    # Display the image
    plt.imshow(frame)
    plt.show()
    """

    return (pos)


def cluster_level_wise(frame,
                       box,
                       pos,
                       thres_height,
                       details_dict):

    """

        Description:-
        Gives list of boxes grouped level wise by taking
        distance(height) based clustering approach

        Arguments:
        frame - Input image
        box - Boxes 3D position information
        pos - Boxes 3D position information
                with respect to pallet marker
                position along with other statistics
        thres_height - Height threshold value
        details_dict - Json file

        Returns list of boxes grouped level wise

    """
    # Convert to np array
    pos.im_point_height = np.array([pos.im_point_height])

    # Apply clustering
    clusters_level_wise = distance_based_clustering(
        pos.im_point_height,
        pos.im_clust_height,
        thres_height)

    # Get unique pixels
    unique_pixels_level_wise = get_unique_pixels(
        clusters_level_wise,
        box.tvecsS,
        1)

    # Update "json" file
    details_dict['No_Of_Levels_Per_Pallet'] = int(
        len(unique_pixels_level_wise))

    # Visualize clustering output - Level
    j = 1
    for val in reversed(unique_pixels_level_wise):

        # Get all points belong to this cluster
        pts = np.where((clusters_level_wise == val))

        # Mark boxes
        sum_level = 0
        for pt in pts[0]:

            sum_level = sum_level+pos.boxes_details[pt][1]
            strr = "[L"+str(j)+","
            cv2.putText(frame, strr,
                        (pos.im_pointR[pt][0]-90, pos.im_pointR[pt][1]+100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)
            pos.boxes_details[pt][3] = j

        """
        # Get average height value
        sum_level_average = sum_level/len(pts[0])
        sum_level_average = round(sum_level_average, 2)

        for pt in pts[0]:
            pos.boxes_details[pt][1] = sum_level_average"""

        j = j+1

    return pos.boxes_details


def paint_hidden_boxes(x_list,
                       index,
                       y_list,
                       dim_length,
                       dim_width,
                       m_fact,
                       color_list_for_boxes,
                       level_boxes_im_copy1,
                       level_boxes_im_copy2,
                       boxes_stat,
                       count,
                       box_status_flag):

    """

        Description:-
        Returns level wise painted floor plan

        Arguments:-
        x_list:- List containing the x co-ordinate of box
        index:- Gives an index of a particular box placed on the pallet
        y_list:- List containing the y co-ordinate of box
        dim_length:- Length of top-view image
        dim_width:- Width of top-view image
        m_fact:- Magnifying Factor
        color_list_for_boxes:- List that will tell which index
                               belong to which color
        level_boxes_im_copy1:- Check if point belongs to painted reason or not
        level_boxes_im_copy2:- Just to visualize which box is being processed
        boxes_stat:- For getting SKU id and assocaited stacking pattern details
        count:- Count of boxes
        box_status_flag:- List that will store the box status flag

        Returns level wise painted floor plan

    """

    # Get the (y, x) co-ordinates of box
    if box_status_flag[index] == 5.0:

        y = int(x_list[index]) * m_fact
        x = int(round(dim_width)) - int(y_list[index])
        x = x * m_fact

    else:

        (y, x) = get_x_and_y_method1(
            x_list[index],
            y_list[index],
            dim_width,
            m_fact)

    # Check if point lies inside the pallet
    if (y >= (dim_length * m_fact) or
            y < 0 or
            x >= (dim_width * m_fact) or
            x < 0):

        return

    # Get the index that belong to this color
    idd = np.where((
        color_list_for_boxes == level_boxes_im_copy1[x, y, :]).all(axis=1))

    # Check if point belongs to painted reason or not
    if ((level_boxes_im_copy1[x, y, 0] == 255) and
            (level_boxes_im_copy1[x, y, 1] == 255) and
            (level_boxes_im_copy1[x, y, 2] == 255)):

        return

    else:

        # Set flag for this box to visible
        boxes_stat[idd] = 1

        # Get all the indices belong to this particular box
        indexx_list = np.where(
            (level_boxes_im_copy1 ==
                level_boxes_im_copy1[x, y, :]).all(axis=2))

        # Just to visualize which box is being processed
        level_boxes_im_copy2[x-10:x+10, y-10:y+10, :] = [255, 255, 255]
        # plt.imshow(level_boxes_im_copy2)
        # plt.show()

        # Turn this particlar box to white
        level_boxes_im_copy1[indexx_list] = [255, 255, 255]

        # Get the min and max of x-coordinates
        indexx_list_x = indexx_list[0]
        min_x = min(indexx_list_x)
        max_x = max(indexx_list_x)

        # Get the min and max of y-coordinates
        indexx_list_y = indexx_list[1]
        min_y = min(indexx_list_y)
        max_y = max(indexx_list_y)

        # Paint all boxes hidden by this box to white
        level_boxes_im_copy1[0:max_x, min_y:max_y, :] = [255, 255, 255]

        """# Display the image
        plt.imshow(level_boxes_im_copy1)
        plt.show()
        plt.show(block=False)
        plt.pause(3)
        plt.close()"""

    # plt.imshow(level_boxes_im_copy2)


def generate_shadowed_image(Box_Pos_Info,
                            sku,
                            color_list_for_boxes_odd_a,
                            color_list_for_boxes_even_a,
                            sub_folder_name,
                            img_name,
                            m_fact,
                            flag, debug):

    """

        Description:-
        Returns white painted visible and hidden
        boxes location on floor plan image

        Arguments:-
        Box_Pos_Info:- Box Position Information wrt Pallet Marker
        sku:- SKU Specific Information
        color_list_for_boxes_odd_a:- List containing color information
        of boxes for flag odd
        color_list_for_boxes_even_a:- List containing color information
        of boxes for flag even
        sub_folder_name:-  Count_Result directory path for saving the results
        img_name:- Image name in string format
        m_fact:- magnifying factor
        flag:- Which gives information related to stacking pattern

        Returns white painted visible and hidden
        boxes location on floor plan image

    """

    # Get start time
    start_time = time.time()

    # Get level details
    level_details = Box_Pos_Info.boxes_details[:, 3]

    # Get maximum level detected
    max_level = max(level_details)

    # Variabel
    box_on_off_list = []
    Level_wise_imge_stat = []

    # Get image dimension for odd level
    (dim_length1, dim_width1) = get_complete_length(
        sku.arranging_pattern1,
        sku.arranging_pattern1_originx,
        sku.arranging_pattern1_originy,
        sku.length_wise_max_level1,
        sku.width_wise_max_level1,
        sku.box_length,
        sku.box_width)

    dim_length1 = int(math.ceil(dim_length1))
    dim_width1 = int(math.ceil(dim_width1))

    # Get image dimension for even level
    (dim_length2, dim_width2) = get_complete_length(
        sku.arranging_pattern2,
        sku.arranging_pattern2_originx,
        sku.arranging_pattern2_originy,
        sku.length_wise_max_level2,
        sku.width_wise_max_level2,
        sku.box_length,
        sku.box_width)

    dim_length2 = int(math.ceil(dim_length2))
    dim_width2 = int(math.ceil(dim_width2))

    # Get one painted image for each level
    count = 1
    for level in range(1, int(max_level+1)):

        # Get all points belong to this cluster
        pts = np.where((level_details == level))

        # Variabels to store info related to all
        # boxes for this particular level
        x_list = []
        y_list = []
        flag_list = []

        # Get all the boxes belonges to this level
        for pt in pts[0]:
            x_list.append(Box_Pos_Info.boxes_details[pt, 2])
            y_list.append(Box_Pos_Info.boxes_details[pt, 0])
            flag_list.append(Box_Pos_Info.boxes_details[pt, 6])

        # Create blank images
        odd_level_boxes_im = np.zeros(
            (dim_length1 * m_fact, dim_width1 * m_fact, 3),
            np.uint8)  # for odd level
        even_level_boxes_im = np.zeros(
            (dim_length2 * m_fact, dim_width2 * m_fact, 3),
            np.uint8)  # for even level

        # Draw images
        odd_level_boxes_im = draw_boxes_m_fix(
            odd_level_boxes_im,
            sku.box_length,
            sku.box_width,
            sku.arranging_pattern1,
            sku.arranging_pattern1_originx,
            sku.arranging_pattern1_originy,
            color_list_for_boxes_odd_a,
            m_fact)
        even_level_boxes_im = draw_boxes_m_fix(
            even_level_boxes_im,
            sku.box_length,
            sku.box_width,
            sku.arranging_pattern2,
            sku.arranging_pattern2_originx,
            sku.arranging_pattern2_originy,
            color_list_for_boxes_even_a,
            m_fact)

        # Define boxes stat
        boxes_stat = np.zeros(
            (len(color_list_for_boxes_odd_a)),
            dtype="int")

        # Make copy
        even_level_boxes_im_copy1 = even_level_boxes_im.copy()
        odd_level_boxes_im_copy1 = odd_level_boxes_im.copy()
        even_level_boxes_im_copy2 = even_level_boxes_im.copy()
        odd_level_boxes_im_copy2 = odd_level_boxes_im.copy()

        # Repeat it for all levels
        for index in range(0, len(x_list)):

            if flag == "odd":

                if level % 2 == 0:

                    # Paint the image
                    paint_hidden_boxes(
                        x_list, index,
                        y_list,
                        dim_length2,
                        dim_width2,
                        m_fact,
                        color_list_for_boxes_even_a,
                        even_level_boxes_im_copy1,
                        even_level_boxes_im_copy2,
                        boxes_stat,
                        count,
                        flag_list)

                    strr_plot = (sub_folder_name +
                                 "/" + img_name +
                                 "_P_" + str(level) +
                                 ".PNG")
                    if debug == "debug":
                        cv2.imwrite(strr_plot, even_level_boxes_im_copy2)

                else:

                    # Paint the image
                    paint_hidden_boxes(
                        x_list,
                        index,
                        y_list,
                        dim_length1,
                        dim_width1,
                        m_fact,
                        color_list_for_boxes_odd_a,
                        odd_level_boxes_im_copy1,
                        odd_level_boxes_im_copy2,
                        boxes_stat, count,
                        flag_list)

                    strr_plot = (sub_folder_name +
                                 "/" +
                                 img_name +
                                 "_P_" +
                                 str(level) +
                                 ".PNG")
                    if debug == "debug":
                        cv2.imwrite(strr_plot, odd_level_boxes_im_copy2)

            if flag == "even":

                if level % 2 == 0:

                    # Paint the image
                    paint_hidden_boxes(
                        x_list,
                        index,
                        y_list,
                        dim_length1,
                        dim_width1,
                        m_fact,
                        color_list_for_boxes_odd_a,
                        odd_level_boxes_im_copy1,
                        odd_level_boxes_im_copy2,
                        boxes_stat,
                        count,
                        flag_list)

                    strr_plot = (sub_folder_name +
                                 "/" +
                                 img_name +
                                 "_P_" +
                                 str(level) +
                                 ".PNG")
                    if debug == "debug":
                        cv2.imwrite(strr_plot, odd_level_boxes_im_copy2)

                else:

                    # Paint the image
                    paint_hidden_boxes(
                        x_list,
                        index,
                        y_list,
                        dim_length2,
                        dim_width2,
                        m_fact,
                        color_list_for_boxes_even_a,
                        even_level_boxes_im_copy1,
                        even_level_boxes_im_copy2,
                        boxes_stat,
                        count,
                        flag_list)

                    strr_plot = (sub_folder_name +
                                 "/" +
                                 img_name +
                                 "_P_" +
                                 str(level) +
                                 ".PNG")
                    if debug == "debug":
                        cv2.imwrite(strr_plot, even_level_boxes_im_copy2)

            count = count + 1

        # Update box on/off list
        box_on_off_list.append(boxes_stat)

        # Update level wise shawoded image stat
        if flag == "odd":

            if level % 2 == 0:

                Level_wise_imge_stat.append(even_level_boxes_im_copy1)

            else:

                Level_wise_imge_stat.append(odd_level_boxes_im_copy1)

        if flag == "even":

            if level % 2 == 0:

                Level_wise_imge_stat.append(odd_level_boxes_im_copy1)

            else:

                Level_wise_imge_stat.append(even_level_boxes_im_copy1)

    elapsed_time = time.time() - start_time
    elapsed = str(timedelta(seconds=elapsed_time))
    # print ("Generate Shadowed Image - Time Taken = ", elapsed)

    return (box_on_off_list, Level_wise_imge_stat)


def border_closeness(level_boxes_im_copy2,
                     min_x,
                     max_x,
                     min_y,
                     max_y,
                     x,
                     y):

    """

        Description:-
        Returns nearest border distance from
        the estimated point

        Arguments:-
        level_boxes_im_copy2:- Image just to visualize
                               which box is being processed
        min_x, max_x, min_y, max_y, x, y:-  Corner Co-ordinates

        Returns nearest border distance from
        the estimated point

    """
    # Get co-ordinates of all the corners
    p0 = np.array([x, y])
    p1 = np.array([min_x, min_y])
    p2 = np.array([max_x, min_y])
    p3 = np.array([max_x, max_y])
    p4 = np.array([min_x, max_y])

    # Get distance from all of the borders
    cv2.line(level_boxes_im_copy2,
             (min_y, min_x),
             (min_y, max_x),
             [255, 255, 255],
             10)
    c1 = np.cross(p2 - p1, p0 - p1) / np.linalg.norm(p2 - p1)

    cv2.line(level_boxes_im_copy2,
             (max_y, max_x),
             (max_y, min_x),
             [255, 255, 255],
             10)
    c3 = np.cross(p4 - p3, p0 - p3) / np.linalg.norm(p4 - p3)

    cv2.line(level_boxes_im_copy2,
             (max_y, min_x),
             (min_y, min_x),
             [255, 255, 255],
             10)
    c4 = np.cross(p1 - p4, p0 - p4) / np.linalg.norm(p1 - p4)

    # Get the smallest distance
    min_c = min(c1, c3, c4)

    return min_c


def get_error(sku,
              arranging_pattern,
              x_list,
              y_list,
              index,
              dim_width,
              dim_length,
              m_fact,
              color_level,
              level_boxes_im_copy1,
              level_boxes_im_copy2,
              e_box_check):
    """

        Description:-
        Return error values for the box

        Arguments:-
        sku:- SKU Specific Information
        Arranging Pattern:- List containing Stacking Pattern of Boxes
        x_list:- List containing the x co-ordinate of box
        index:- Gives an index of a particular box placed on the pallet
        y_list:- List containing the y co-ordinate of box
        dim_length:- Length of top-view image
        dim_width:- Width of top-view image
        m_fact:- Magnifying Factor
        color_level:- List of Colored Level wise boxes
        level_boxes_im_copy1:- Check if point belongs to painted reason or not
        level_boxes_im_copy2:- Just to visualize which box is being processed
        e_box_check:- Error matrix for boxes in numpy array format

        Return error values for the box

    """
    # Get the (x, y) co-ordinates of the box
    (y, x) = get_x_and_y_method1(x_list[index],
                                 y_list[index],
                                 dim_width,
                                 m_fact)

    # Condition to check point lies on the pallet or not
    if (y >= (dim_length * m_fact) or
            y < 0 or x >= (dim_width * m_fact) or
            x < 0):

        return (0, 0)

    # Check if point lies in hollow space or not
    if np.array_equal(level_boxes_im_copy1[x, y, :], [0, 0, 0]):

        return (0, 0)

    # Get the index that belong to this color
    idd = np.where((color_level == level_boxes_im_copy1[x, y, :]).all(axis=1))
    idd = idd[0][0]

    # Check if box is already marked
    e_box_check[idd] = e_box_check[idd] + 1

    # Get the color belonges to this box
    color = color_level[idd]

    # Get all the indices that belong to this particular box
    indexx_list = np.where((level_boxes_im_copy1 == color).all(axis=2))

    # Get the x and y component of this list
    indexx_list_x = indexx_list[0]
    indexx_list_y = indexx_list[1]

    # Get the min and max of x-coordinates
    indexx_list_x = indexx_list[0]
    min_x = min(indexx_list_x)
    max_x = max(indexx_list_x)

    # Get the min and max of y-coordinates
    indexx_list_y = indexx_list[1]
    min_y = min(indexx_list_y)
    max_y = max(indexx_list_y)

    # Refrence point is 1/4th distance away from the base of the box
    if arranging_pattern[idd] == "L":

        x_centre = (min_x +
                    (int(sku.box_width) - int(sku.box_width/4)) * m_fact)
        y_centre = min_y + int(sku.box_length/2) * m_fact
    if arranging_pattern[idd] == "B":

        x_centre = (min_x +
                    (int(sku.box_length) - int(sku.box_length/4)) * m_fact)
        y_centre = min_y + int(sku.box_width/2) * m_fact

    # Just to visualize which box is being processed
    level_boxes_im_copy2[x_centre-10:x_centre+10,
                         y_centre-10:y_centre+10,
                         :] = [0, 0, 0]
    level_boxes_im_copy2[x-10:x+10, y-10:y+10, :] = [255, 255, 255]

    # Calculate distance error
    error1 = np.sqrt((x-x_centre)**2 + (y-y_centre)**2)

    # Add penalty
    if e_box_check[idd] > 1:
        error1 = error1 + 100000

    # Calculate border closeness error
    error2 = border_closeness(
        level_boxes_im_copy2,
        min_x,
        max_x,
        min_y,
        max_y,
        x,
        y)

    # Return error
    return (error1, error2)


def decide_level_pattern(boxes_3d_info,
                         sku,
                         color_odd,
                         color_even,
                         m_fact):
    """

        Returns info that decides stacking pattern
        for each level

        Args:
            boxes_3d_info - Boxes 3D refrenced
                            position values
            sku - SKU specific information
            color_odd - Pre generated List of
                        color for even level
            color_even - Pre generated List of
                         color for odd level
            m_fact - Zoom in factor

        Retuens error_odd_start, error_even_start, closeness_odd_start,
        closeness_even_start which will help in deciding stacking
        pattern for each level

    """

    # Get start time
    start_time = time.time()

    # Get level details
    level_details = boxes_3d_info.boxes_details[:, 3]

    # Get maximum level detected
    max_level = max(level_details)

    # Get image dimension for odd level
    (dim_length1, dim_width1) = get_complete_length(
        sku.arranging_pattern1,
        sku.arranging_pattern1_originx,
        sku.arranging_pattern1_originy,
        sku.length_wise_max_level1,
        sku.width_wise_max_level1,
        sku.box_length,
        sku.box_width)
    dim_length1 = int(math.ceil(dim_length1))
    dim_width1 = int(math.ceil(dim_width1))

    # Get image dimension for even level
    (dim_length2, dim_width2) = get_complete_length(
        sku.arranging_pattern2,
        sku.arranging_pattern2_originx,
        sku.arranging_pattern2_originy,
        sku.length_wise_max_level2,
        sku.width_wise_max_level2,
        sku.box_length,
        sku.box_width)
    dim_length2 = int(math.ceil(dim_length2))
    dim_width2 = int(math.ceil(dim_width2))

    error_odd_start = 0
    error_even_start = 0

    closeness_odd_start = 0
    closeness_even_start = 0

    flag = 0

    for level in range(1, int(max_level+1)):

        # Get all points belong to this cluster
        pts = np.where((level_details == level))

        # Variabels to store info related to all
        # boxes for the level
        x_list = []
        y_list = []

        # Get all the boxes belonges to this level
        for pt in pts[0]:
            x_list.append(boxes_3d_info.boxes_details[pt, 2])
            y_list.append(boxes_3d_info.boxes_details[pt, 0])

        # Create blank images
        odd_level_boxes_im = np.zeros(
            (dim_length1 * m_fact, dim_width1 * m_fact, 3),
            np.uint8)  # for odd level
        even_level_boxes_im = np.zeros(
            (dim_length2 * m_fact, dim_width2 * m_fact, 3),
            np.uint8)  # for even level

        # Draw images
        odd_level_boxes_im = draw_boxes_m_fix(
            odd_level_boxes_im,
            sku.box_length,
            sku.box_width,
            sku.arranging_pattern1,
            sku.arranging_pattern1_originx,
            sku.arranging_pattern1_originy,
            color_odd, m_fact)
        even_level_boxes_im = draw_boxes_m_fix(
            even_level_boxes_im,
            sku.box_length,
            sku.box_width,
            sku.arranging_pattern2,
            sku.arranging_pattern2_originx,
            sku.arranging_pattern2_originy,
            color_even,
            m_fact)

        # Make copy
        even_level_boxes_im_copy1 = even_level_boxes_im.copy()
        odd_level_boxes_im_copy1 = odd_level_boxes_im.copy()
        even_level_boxes_im_copy2 = even_level_boxes_im.copy()
        odd_level_boxes_im_copy2 = odd_level_boxes_im.copy()

        # Estimate errors
        error_odd = 0
        error_even = 0

        closeness_odd = 0
        closeness_even = 0

        e1_box_check = np.zeros(len(color_odd))
        e2_box_check = np.zeros(len(color_even))

        for index in range(0, len(x_list)):

            (e1, c1) = get_error(
                sku,
                sku.arranging_pattern1,
                x_list,
                y_list,
                index,
                dim_width1,
                dim_length1,
                m_fact,
                color_odd,
                odd_level_boxes_im_copy1,
                odd_level_boxes_im_copy2,
                e1_box_check)

            (e2, c2) = get_error(
                sku,
                sku.arranging_pattern2,
                x_list,
                y_list,
                index,
                dim_width2,
                dim_length2,
                m_fact,
                color_even,
                even_level_boxes_im_copy1,
                even_level_boxes_im_copy2,
                e2_box_check)

            # Get combined errors
            error_odd = error_odd + e1
            error_even = error_even + e2

            closeness_odd = closeness_odd + c1
            closeness_even = closeness_even + c2

        if flag == 0:

            error_odd_start = error_odd_start + error_odd
            error_even_start = error_even_start + error_even

            closeness_odd_start = closeness_odd_start + closeness_odd
            closeness_even_start = closeness_even_start + closeness_even

        if flag == 1:

            error_odd_start = error_odd_start + error_even
            error_even_start = error_even_start + error_odd

            closeness_odd_start = closeness_odd_start + closeness_even
            closeness_even_start = closeness_even_start + closeness_odd

        flag = abs(flag - 1)

    elapsed_time = time.time() - start_time
    elapsed = str(timedelta(seconds=elapsed_time))
    # print ("Decide Level Pattern - Time Taken = ", elapsed)

    return (
        error_odd_start,
        error_even_start,
        closeness_odd_start,
        closeness_even_start)


def get_hidden_list(color_list_for_boxes,
                    i,
                    level_boxes_im,
                    val,
                    box_midpos):

    """

        Description:-
        Gives information about hidden boxes level wise.

        Arguments:-
        color_list_for_boxes:- List containing box's individual color helps
                               in segregating visible box from hidden box
        i:- Box's Count
        level_boxes_im:- For getting image patch from the level top
                         view image
        val:- For Getting the image patch from the white shawoded
              top view image
        box_midpos:- Numpy array containing box's mid point coordinates

        Returns hidden list level wise

    """

    # Get the color belonges to this box
    color = color_list_for_boxes[i]

    # Get all the indices belong to this
    # particular box
    indexx_list = np.where((level_boxes_im == color).all(axis=2))

    # Get the x and y component of this list
    indexx_list_x = indexx_list[0]
    indexx_list_y = indexx_list[1]

    # Get the min and max of x-coordinates
    indexx_list_x = indexx_list[0]
    min_x = min(indexx_list_x)
    max_x = max(indexx_list_x)

    # Get the min and max of y-coordinates
    indexx_list_y = indexx_list[1]
    min_y = min(indexx_list_y)
    max_y = max(indexx_list_y)

    # Get the image patch from the level top view image
    img_temp1 = level_boxes_im[min_x:max_x, min_y:max_y, :]

    # Get the image patch from the white shawoded top view image
    img_temp2 = val[min_x:max_x, min_y:max_y, :]

    # Box mid pos
    box_midpos[i][0] = min_x
    box_midpos[i][1] = min_y

    # Get all the location of white points from shawoded image patch
    index_val_white = np.where((img_temp2 == [255, 255, 255]).all(axis=2))

    # Get all the location of this particular box
    index_val_color = np.where((img_temp1 == color).all(axis=2))

    # Get the length of each list
    index_val_white_length = len(index_val_white[0])
    index_val_color_length = len(index_val_color[0])

    # Get the percentage overlap
    percentage_overlap = (
        ((index_val_white_length * 1.0)/index_val_color_length) * 100)

    return (percentage_overlap, min_x, min_y)


def get_hidden_boxes(boxes_3d_info,
                     m_fact,
                     box_on_off_list,
                     sku,
                     color_list_for_boxes_odd_a,
                     color_list_for_boxes_even_a,
                     Level_wise_imge_stat,
                     flag):
    """

        Description:-
        Return list of hidden boxes

        Args:
        boxes_3d_info - Boxes 3D refrenced
                        position values
        sku - SKU specific information
        color_list_for_boxes_odd_a:- Color List containing box's
                                     color info for odd flag
        color_list_for_boxes_even_a:- Color List containing box's
                                      color info for even flag
        Level_wise_imge_stat:- List containing level wise box's
                               stacking pattern
        flag:- Flag value helps in deciding odd or even stacking
               pattern of each level
        m_fact - Zoom in factor

        Return list of hidden boxes

    """

    # Get start time
    start_time = time.time()

    # Get level details
    level_details = boxes_3d_info.boxes_details[:, 3]

    # Get maximum level detected
    max_level = max(level_details)

    # Start with first level
    level = 1

    # Variabel to store the mid point of each box
    boxes_midpos_List = []

    # Get image dimension for odd level
    (dim_length1, dim_width1) = get_complete_length(
        sku.arranging_pattern1,
        sku.arranging_pattern1_originx,
        sku.arranging_pattern1_originy,
        sku.length_wise_max_level1,
        sku.width_wise_max_level1,
        sku.box_length,
        sku.box_width)

    dim_length1 = int(math.ceil(dim_length1))
    dim_width1 = int(math.ceil(dim_width1))

    # Get image dimension for even level
    (dim_length2, dim_width2) = get_complete_length(
        sku.arranging_pattern2,
        sku.arranging_pattern2_originx,
        sku.arranging_pattern2_originy,
        sku.length_wise_max_level2,
        sku.width_wise_max_level2,
        sku.box_length,
        sku.box_width)

    dim_length2 = int(math.ceil(dim_length2))
    dim_width2 = int(math.ceil(dim_width2))

    # Create a blank image for each pattern (odd + even)
    odd_level_boxes_im = np.zeros(
        (dim_length1 * m_fact, dim_width1 * m_fact, 3),
        np.uint8)  # for odd level
    even_level_boxes_im = np.zeros(
        (dim_length2 * m_fact, dim_width2 * m_fact, 3),
        np.uint8)  # for even level

    # Draw images
    odd_level_boxes_im = draw_boxes_m_fix(
        odd_level_boxes_im,
        sku.box_length,
        sku.box_width,
        sku.arranging_pattern1,
        sku.arranging_pattern1_originx,
        sku.arranging_pattern1_originy,
        color_list_for_boxes_odd_a,
        m_fact)

    even_level_boxes_im = draw_boxes_m_fix(
        even_level_boxes_im,
        sku.box_length,
        sku.box_width,
        sku.arranging_pattern2,
        sku.arranging_pattern2_originx,
        sku.arranging_pattern2_originy,
        color_list_for_boxes_even_a,
        m_fact)

    # Repeat it for all detected levels
    for val in Level_wise_imge_stat:

        # Variabel to store the mid point of each box
        box_midpos = np.zeros(
            (len(color_list_for_boxes_odd_a), 2),
            dtype="float32")

        # Repeat it for each boxes of this level
        if flag == "odd":

            if level % 2 == 0:

                for i in range(0, len(color_list_for_boxes_even_a)):

                    # Get percentage overlap
                    (percentage_overlap, min_x, min_y) = get_hidden_list(
                        color_list_for_boxes_even_a,
                        i,
                        even_level_boxes_im,
                        val,
                        box_midpos)

                    # If box meet the criteria, put it into hidden list
                    if percentage_overlap > 30.0:
                        if box_on_off_list[level - 1][i] != 1:
                            box_on_off_list[level - 1][i] = 2

            else:

                for i in range(0, len(color_list_for_boxes_odd_a)):

                    # Get percentage overlap
                    (percentage_overlap, min_x, min_y) = get_hidden_list(
                        color_list_for_boxes_odd_a,
                        i,
                        odd_level_boxes_im,
                        val,
                        box_midpos)

                    # If box meet the criteria, put it into hidden list
                    if percentage_overlap > 30.0:
                        if box_on_off_list[level - 1][i] != 1:
                            box_on_off_list[level - 1][i] = 2

        if flag == "even":

            if level % 2 == 0:

                for i in range(0, len(color_list_for_boxes_odd_a)):

                    # Get percentage overlap
                    (percentage_overlap, min_x, min_y) = get_hidden_list(
                        color_list_for_boxes_odd_a,
                        i,
                        odd_level_boxes_im,
                        val,
                        box_midpos)

                    # If box meet the criteria, put it into hidden list
                    if percentage_overlap > 30.0:
                        if box_on_off_list[level - 1][i] != 1:
                            box_on_off_list[level - 1][i] = 2

            else:

                for i in range(0, len(color_list_for_boxes_even_a)):

                    # Get percentage overlap
                    (percentage_overlap, min_x, min_y) = get_hidden_list(
                        color_list_for_boxes_even_a,
                        i,
                        even_level_boxes_im,
                        val,
                        box_midpos)

                    # If box meet the criteria, put it into hidden list
                    if percentage_overlap > 30.0:
                        if box_on_off_list[level - 1][i] != 1:
                            box_on_off_list[level - 1][i] = 2

        # Append boxes mid pos list
        boxes_midpos_List.append(box_midpos)

        # Increment the level
        level = level+1

    elapsed_time = time.time() - start_time
    elapsed = str(timedelta(seconds=elapsed_time))
    # print ("Get Hidden Boxes - Time Taken = ", elapsed)

    return (boxes_midpos_List, box_on_off_list)


def count_total_number_of_boxes(Box_Pos_Info,
                                sku,
                                boxes_midpos_List,
                                box_on_off_list,
                                sub_folder_name,
                                img_name,
                                file_handler,
                                flag,
                                score_matrix,
                                details_dict,
                                Box_Marker, image, temp,stock_dir,pallet_id):

    """

        Description:-
        This module returns total number of boxes present on pallet
        by suming up the total number of boxes present on each level.


        Args:
            Box_Pos_Info - Boxes 3d position information
            sku - SKU related information
            boxes_midpos_List - Mid position
            box_on_off_list - Box status information
            sub_folder_name - Folder name
            img_name - Image name
            file_handler - File handler to write output to text file
            flag:- even or odd, according to that we
            will get a count in a particular level
            score_matrix:- Numpy Array which stores confidence
            score pertaining to each sku
            details_dict:-Json file
            box_marker:- Boxes 3D position information
            image: Input Image
            temp:- Masked and Gamma Corrected Image

        Returns total number of boxes present on pallet

    """

    # Get start time
    image = image
    image2 = temp
    start_time = time.time()

    # Get level details
    level_details = Box_Pos_Info.boxes_details[:, 3]

    # Get maximum level detected
    max_level = max(level_details)

    count_matrix = np.zeros((int(max_level), 1), dtype="int")

    # Set level to 1
    level = 1

    # Set total count to zero
    total_count = 0
    confidance_score_all = 0
    level_details_all = []
    level_count = 0
    y = 10
    # Repet this for each detected level
    for level in range(1, int(max_level+1)):

        level_details = {}

        box_list_all = []

        level_details['Level_ID'] = int(level)
        level_details['SKU_ID'] = int(Box_Marker.unique_ids[0][0])

        # Print level no which is being processed
        print("Level =", level, "-", end=' ')

        # Lists to store boxes of each category
        zeros_listX = []
        zeros_listY = []
        ones_listX = []
        ones_listY = []
        twos_listX = []
        twos_listY = []

        # Initialize all counts to zero
        zero_count = 0
        one_count = 0
        two_count = 0

        # Boxes mid pos list level wise
        boxes_midpos_list_level_wise = boxes_midpos_List[level - 1]

        # Get level stat list
        temp_list = box_on_off_list[level - 1]

        # Get the count and position of absent boxes
        (zeros_list_x, zeros_list_y, zero_count, box_details_all) = \
            get_box_count(
                temp_list,
                sku,
                0,
                level,
                flag,
                score_matrix,
                level_details,
                box_list_all)

        level_details['Absent_Count'] = int(zero_count)

        # Get the count and position of visible boxes
        (onesList_x, onesList_y, one_count, box_details_all) = \
            get_box_count(
                temp_list,
                sku,
                1,
                level,
                flag,
                score_matrix,
                level_details,
                box_list_all)

        level_details['Visible_Count'] = int(one_count)

        # Get the count and position of hidden boxes
        (twosList_x, twosList_y, two_count, box_details_all) = \
            get_box_count(
                temp_list,
                sku,
                2,
                level,
                flag,
                score_matrix,
                level_details,
                box_list_all)

        level_details['Hidden_Count'] = int(two_count)

        level_details['Boxes'] = box_list_all

        # Print boxes info
        print("No. of boxes absent = ", zero_count,
              " No. of boxes visible = ", one_count,
              " No. of boxes hidden = ", two_count,
              end=' ')

        print("Total Count = ", one_count + two_count, end=' ')
        print()
        tr = one_count + two_count
        str1 = "Total Count of Level=" + str(level) + "is: " + str(tr)
        image = cv2.putText(image, str1, (30, 30 + y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)
        image2 = cv2.putText(image2, str1, (30, 30 + y),
                             cv2.FONT_HERSHEY_SIMPLEX,
                             1, (0, 0, 255), 2, cv2.LINE_AA)
        y += 30
        # Get confidance list
        pts = np.where((score_matrix[:, 0] == level))
        required_score = score_matrix[pts[0], 2]
        confidance_score = round(sum(required_score) / len(required_score), 2)

        # Calculate confidance score
        if confidance_score != 0.0:

            level_count = level_count + 1
            confidance_score_all = confidance_score_all + confidance_score

        # print("Error = ", confidance_score)

        dict_index = "Level_" + str(level)

        # Add level count to total count
        total_count = total_count + one_count + two_count

        count_matrix[level - 1] = int(one_count + two_count)

        if flag == "odd":

            if level % 2 == 0:

                level_details['No_Of_Boxes_Per_Level'] = \
                    int(len(sku.arranging_pattern2))
                level_details['Stacking_Pattern'] = "E"

            else:

                level_details['No_Of_Boxes_Per_Level'] = \
                    int(len(sku.arranging_pattern1))
                level_details['Stacking_Pattern'] = "O"

        if flag == "even":

            if level % 2 == 0:

                level_details['No_Of_Boxes_Per_Level'] = \
                    int(len(sku.arranging_pattern1))
                level_details['Stacking_Pattern'] = "O"

            else:

                level_details['No_Of_Boxes_Per_Level'] = \
                    int(len(sku.arranging_pattern2))
                level_details['Stacking_Pattern'] = "E"

        level_details['Estimated_Count'] = int(one_count + two_count)
        level_details['Error_Score'] = float(confidance_score)

        # Visualize the result
        if flag == "odd":

            if level % 2 == 0:

                draw_plot(
                    plt,
                    sku.arranging_pattern2,
                    sku.arranging_pattern2_originx,
                    sku.arranging_pattern2_originy,
                    sku.box_length,
                    sku.box_width)

            else:

                draw_plot(
                    plt,
                    sku.arranging_pattern1,
                    sku.arranging_pattern1_originx,
                    sku.arranging_pattern1_originy,
                    sku.box_length,
                    sku.box_width)

        if flag == "even":

            if level % 2 == 0:

                draw_plot(
                    plt,
                    sku.arranging_pattern1,
                    sku.arranging_pattern1_originx,
                    sku.arranging_pattern1_originy,
                    sku.box_length,
                    sku.box_width)
            else:

                draw_plot(
                    plt,
                    sku.arranging_pattern2,
                    sku.arranging_pattern2_originx,
                    sku.arranging_pattern2_originy,
                    sku.box_length,
                    sku.box_width)

        plt.plot(zeros_list_x, zeros_list_y, 'ro')
        plt.plot(onesList_x, onesList_y, 'go')
        plt.plot(twosList_x, twosList_y, 'bo')
        # plt.show()

        # Save level plot
        strr_plot = (sub_folder_name +
                     "/" +
                     img_name +
                     "_L_" +
                     str(level) +
                     ".PNG")
        plt.savefig(strr_plot)
        plt.close()

        # Write total count to text file
        strr = str(one_count + two_count) + " "
        file_handler.write(strr)

        level_details_all.append(level_details)

    # Print total count
    print("Total Count = ", total_count)
    print()

    confidance_score_all = round(confidance_score_all/level_count, 2)

    # print("Error_All = ", confidance_score_all)

    details_dict['Pallet_confidence'] = confidance_score_all
    details_dict['Levels'] = level_details_all

    strr_1 = "Total Number of Boxes = "+str(total_count)
    strr = str(total_count) + "\n"
    file_handler.write(strr)

    elapsed_time = time.time() - start_time
    elapsed = str(timedelta(seconds=elapsed_time))
    # print ("Count Total Number of Boxes - Time Taken = ", elapsed)
    # Change the location to bottom-left
    image = cv2.putText(image, 'Total-Count of Boxes:-' + str(total_count),
                        (30, 150), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)

    image2 = cv2.putText(image2, 'Total-Count of Boxes:-' + str(total_count),
                         (30, 150), cv2.FONT_HERSHEY_SIMPLEX,
                         1, (0, 0, 255), 2, cv2.LINE_AA)
    """
    cv2.namedWindow("Img", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Img", 400, 400)
    cv2.imshow("Img", image)
    cv2.waitKey(0)
    """
    cv2.imwrite(sub_folder_name + '/img_'+ str(pallet_id)+ '.png', image) 
    final_img_path = ""
    final_img_path = "processed_images" + '/img_'+ str(pallet_id)+ '.png'
    cv2.imwrite(stock_dir + '/img_'+ str(pallet_id)+ '.png', image)

    return (total_count, count_matrix, final_img_path)


def get_score(sku,
              arranging_pattern,
              h_list,
              x_list,
              y_list,
              index,
              dim_width,
              dim_length,
              m_fact,
              color_level,
              level_boxes_im_copy1,
              level_boxes_im_copy2,
              e_box_check,
              level,
              score_matrix,
              count,
              box_status_flag):

    """

        Description:-
        Returns error for individual estimated position

        Arguments:-
        sku:- SKU Specific Information
        Arranging Pattern:- List containing Stacking Pattern of Boxes
        x_list:- List containing the x co-ordinate of box
        index:- Gives an index of a particular box placed on the pallet
        y_list:- List containing the y co-ordinate of box
        dim_length:- Length of top-view image
        dim_width:- Width of top-view image
        m_fact:- Magnifying Factor
        color_level:- List of Colored Level wise boxes
        level_boxes_im_copy1:- Check if point belongs to painted reason or not
        level_boxes_im_copy2:- Just to visualize which box is being processed
        e_box_check:- Error matrix for boxes in numpy array format
        level:- Particular Level present in the pallet
        score_matrix:- Numpy Array which stores confidence score pertaining to
        each sku
        count:- Count of each level
        box_status_flag:- Flag that will be turned on if box is present
        at the respective position

        Returns error for individual estimated position
    """

    # Get (x, y) co-ordinates
    if box_status_flag[index] == 5.0:
        y = int(x_list[index]) * m_fact
        x = int(round(dim_width)) - int(y_list[index])
        x = x * m_fact
    else:
        (y, x) = get_x_and_y_method1(
            x_list[index],
            y_list[index],
            dim_width,
            m_fact)

    # Check if point lies in hollow reason or not
    if (y >= (dim_length * m_fact) or
            y < 0 or x >= (dim_width * m_fact) or
            x < 0):

        return (0, 0)

    if np.array_equal(level_boxes_im_copy1[x, y, :], [0, 0, 0]):

        return (0, 0)

    idd = np.where((color_level == level_boxes_im_copy1[x, y, :]).all(axis=1))
    idd = idd[0][0]

    # Check if box is already marked
    e_box_check[idd] = e_box_check[idd] + 1

    # Get the color belonges to this box
    color = color_level[idd]

    # Get all the indices belong to this particular box
    indexx_list = np.where((level_boxes_im_copy1 == color).all(axis=2))

    # Get the x and y component of this list
    indexx_list_x = indexx_list[0]
    indexx_list_y = indexx_list[1]

    # Get the min and max of x-coordinates
    indexx_list_x = indexx_list[0]
    min_x = min(indexx_list_x)
    max_x = max(indexx_list_x)

    # Get the min and max of y-coordinates
    indexx_list_y = indexx_list[1]
    min_y = min(indexx_list_y)
    max_y = max(indexx_list_y)

    # Refrence point is 1/4th distance away
    # from the base of the box
    if arranging_pattern[idd] == "L":

        x_centre = (
            min_x +
            (int(sku.box_width) - int(sku.box_width/4)) * m_fact)
        y_centre = min_y + int(sku.box_length/2) * m_fact

    if arranging_pattern[idd] == "B":

        x_centre = (
            min_x +
            (int(sku.box_length) - int(sku.box_width/4)) * m_fact)
        y_centre = min_y + int(sku.box_width/2) * m_fact

    # Just to visualize which box is being processed
    level_boxes_im_copy2[x_centre-10:x_centre+10,
                         y_centre-10:y_centre+10, :] = [0, 0, 0]
    level_boxes_im_copy2[x-10:x+10, y-10:y+10, :] = [255, 255, 255]

    # Get refrence values
    if arranging_pattern[idd] == "L":
        reference_height = sku.box_height
        reference_distance = sku.box_length
        reference_depth = sku.box_width
        reference_area = sku.box_length * sku.box_width

    if arranging_pattern[idd] == "B":
        reference_height = sku.box_height
        reference_distance = sku.box_width
        reference_depth = sku.box_length
        reference_area = sku.box_length * sku.box_width

    # Calculate height score
    actual_height = sku.box_height * level
    est_height = h_list[index]
    score_height = np.sqrt((actual_height - est_height)**2)
    base_height = sku.box_height

    score_height_per = round((score_height/reference_height) * 100, 2)

    # Calculate distance score
    actual_distance = round(y_centre/10, 2)
    est_distance = round(y/10, 2)

    score_distance = np.sqrt((actual_distance - est_distance)**2)
    base_distance = round(actual_distance - min_y/10, 2)

    score_distance_per = round((score_distance/reference_distance) * 100, 2)

    # Calculate depth score
    actual_depth = round(x_centre/10, 2)
    est_depth = round(x/10, 2)
    score_depth = np.sqrt((actual_depth - est_depth)**2)
    base_depth = round(max_x/10 - x/10, 2)

    score_depth_per = round((score_depth/reference_depth) * 100, 2)

    # Calculate border score
    border_score = round(
        (np.sqrt((x_centre - x)**2 + (y_centre - y)**2)) / 10,
        2)

    level_boxes_im_copy2[min_x-10:min_x+10,
                         min_y-10:min_y+10, :] = [255, 255, 255]
    level_boxes_im_copy2[max_x-10:max_x+10,
                         max_y-10:max_y+10, :] = [255, 255, 255]

    border_score_ref = round(
        (np.sqrt((max_x - min_x)**2 + (max_y - min_y)**2)) / 10,
        2)

    border_score_per = round((border_score / border_score_ref) * 100, 2)

    # Update score matrix
    score_matrix[count, 1] = idd
    score_matrix[count, 2] = border_score_per
    score_matrix[count, 3] = score_height_per
    score_matrix[count, 4] = score_distance_per
    score_matrix[count, 5] = score_depth_per


def calculate_confidance_score(boxes_3d_info,
                               sku,
                               color_odd,
                               color_even,
                               sub_folder_name,
                               img_name,
                               m_fact,
                               flag):

    """

        Description:-
        Returns confidance score

        Arguments:-
        sku:- SKU Specific Information
        m_fact:- Magnifying Factor
        boxes_3d_info - Boxes 3D refrenced
                        position values
        sku - SKU specific information
        flag:- Flag value helps in deciding odd or even stacking
               pattern of each level
        color_odd - Pre generated List of
                    color for even level
        color_even - Pre generated List of
                     color for odd level

        Returns confidance score

    """

    # Get start time
    start_time = time.time()

    # Get level details
    level_details = boxes_3d_info.boxes_details[:, 3]

    # Get maximum level detected
    max_level = max(level_details)

    # Get image dimension for odd level
    (dim_length1, dim_width1) = get_complete_length(
        sku.arranging_pattern1,
        sku.arranging_pattern1_originx,
        sku.arranging_pattern1_originy,
        sku.length_wise_max_level1,
        sku.width_wise_max_level1,
        sku.box_length,
        sku.box_width)

    dim_length1 = int(math.ceil(dim_length1))
    dim_width1 = int(math.ceil(dim_width1))

    odd_level_boxes_im = np.zeros(
        (dim_length1 * m_fact, dim_width1 * m_fact, 3),
        np.uint8)

    odd_level_boxes_im = draw_boxes_m_fix(
        odd_level_boxes_im,
        sku.box_length,
        sku.box_width,
        sku.arranging_pattern1,
        sku.arranging_pattern1_originx,
        sku.arranging_pattern1_originy,
        color_odd,
        m_fact)

    # Get image dimension for even level
    (dim_length2, dim_width2) = get_complete_length(
        sku.arranging_pattern2,
        sku.arranging_pattern2_originx,
        sku.arranging_pattern2_originy,
        sku.length_wise_max_level2,
        sku.width_wise_max_level2,
        sku.box_length,
        sku.box_width)

    dim_length2 = int(math.ceil(dim_length2))
    dim_width2 = int(math.ceil(dim_width2))

    even_level_boxes_im = np.zeros(
        (dim_length2 * m_fact, dim_width2 * m_fact, 3),
        np.uint8)

    even_level_boxes_im = draw_boxes_m_fix(
        even_level_boxes_im,
        sku.box_length,
        sku.box_width,
        sku.arranging_pattern2,
        sku.arranging_pattern2_originx,
        sku.arranging_pattern2_originy,
        color_even,
        m_fact)

    # Stores statistics related to error calculation
    score_matrix = np.zeros(
        (boxes_3d_info.boxes_details.shape[:1][0], 6),
        dtype="float32")

    # Level wise processing

    count = 0

    for level in range(1, int(max_level+1)):

        # Get all points belong to this cluster
        pts = np.where((level_details == level))

        # Variabels to store info related to all
        # boxes for this particular level
        h_list = []
        x_list = []
        y_list = []
        flag_list = []
        e_box_check = np.zeros(len(color_odd))

        # Get all the boxes belonges to this level
        for pt in pts[0]:
            h_list.append(boxes_3d_info.boxes_details[pt, 1])
            x_list.append(boxes_3d_info.boxes_details[pt, 2])
            y_list.append(boxes_3d_info.boxes_details[pt, 0])
            flag_list.append(boxes_3d_info.boxes_details[pt, 6])

        # Make copy
        odd_level_boxes_im_copy1 = odd_level_boxes_im.copy()
        odd_level_boxes_im_copy2 = odd_level_boxes_im.copy()
        even_level_boxes_im_copy1 = even_level_boxes_im.copy()
        even_level_boxes_im_copy2 = even_level_boxes_im.copy()

        for index in range(0, len(x_list)):

            score_matrix[count, 0] = level

            if flag == "odd":

                if level % 2 == 0:

                    get_score(
                        sku,
                        sku.arranging_pattern2,
                        h_list,
                        x_list,
                        y_list,
                        index,
                        dim_width2,
                        dim_length2,
                        m_fact,
                        color_even,
                        even_level_boxes_im_copy1,
                        even_level_boxes_im_copy2,
                        e_box_check,
                        level,
                        score_matrix,
                        count,
                        flag_list)

                else:

                    get_score(
                        sku,
                        sku.arranging_pattern1,
                        h_list,
                        x_list,
                        y_list,
                        index,
                        dim_width1,
                        dim_length1,
                        m_fact,
                        color_odd,
                        odd_level_boxes_im_copy1,
                        odd_level_boxes_im_copy2,
                        e_box_check,
                        level,
                        score_matrix,
                        count,
                        flag_list)

            if flag == "even":

                if level % 2 == 0:

                    get_score(
                        sku,
                        sku.arranging_pattern1,
                        h_list,
                        x_list,
                        y_list,
                        index,
                        dim_width1,
                        dim_length1,
                        m_fact,
                        color_odd,
                        odd_level_boxes_im_copy1,
                        odd_level_boxes_im_copy2,
                        e_box_check,
                        level,
                        score_matrix,
                        count,
                        flag_list)

                else:

                    get_score(
                        sku,
                        sku.arranging_pattern2,
                        h_list,
                        x_list,
                        y_list,
                        index,
                        dim_width2,
                        dim_length2,
                        m_fact,
                        color_even,
                        even_level_boxes_im_copy1,
                        even_level_boxes_im_copy2,
                        e_box_check,
                        level,
                        score_matrix,
                        count,
                        flag_list)

            count = count + 1

    elapsed_time = time.time() - start_time
    elapsed = str(timedelta(seconds=elapsed_time))
    # print ("Calculate ConfidanceScore - Time Taken = ", elapsed)

    return score_matrix


def draw_box_and_verify(arranging_pattern1,
                        arranging_pattern1_originx,
                        arranging_pattern1_originy,
                        dim_length1,
                        dim_width1,
                        arranging_pattern2,
                        arranging_pattern2_originx,
                        arranging_pattern2_originy,
                        dim_length2,
                        dim_width2,
                        i,
                        j,
                        k,
                        box_width,
                        box_length,
                        box_height,
                        boxes_stat,
                        Box_Pos_Info,
                        Pallet_Marker):

    """

    Checks for any overlap of base box with respect to top level boxes

    Arguments:
        arranging_pattern1: 1st stacking pattern Type(odd)
        arranging_pattern1_originx: 1st stacking pattern types' x-axis
            orientation of boxes
        arranging_pattern1_originy: 1st stacking pattern types' Y-axis
            orientation of boxes
        dim_length1: lenght of pixels in img1(odd level)
        dim_width1: width of pixels in img2
        arranging_pattern2:2nd stacking pattern Type(even)
        arranging_pattern2_originx:2nd stacking pattern types' x-axis
            orientation of boxes
        arranging_pattern2_originy:2nd stacking pattern types' y-axis
            orientation of boxes
        dim_length2:length of pixels in img2(even level)
        dim_width2: width of pixels in img2
        i, j, k:
        box_width: width of boxes
        box_length: lenght of boxes
        box_height:height value of boxes
        boxes_stat:SKU id and assocaited stacking pattern details
        Box_Pos_Info - Boxes 3d position information
        Pallet_marker: pallet marker related info

    Returns:


    """

    # Create blank images
    im1 = np.zeros((dim_length1, dim_width1, 3), np.uint8)
    im2 = np.zeros((dim_length2, dim_width2, 3), np.uint8)
    im3 = np.zeros((dim_length2, dim_width2, 3), np.uint8)

    # Get the co-ordinates of base as well as top box
    if arranging_pattern1[j] == "L":
        box1_x1 = arranging_pattern1_originy[j]
        box1_y1 = arranging_pattern1_originx[j]
        box1_x2 = arranging_pattern1_originy[j] + box_width
        box1_y2 = arranging_pattern1_originx[j] + box_length

    else:
        box1_x1 = arranging_pattern1_originy[j]
        box1_y1 = arranging_pattern1_originx[j]
        box1_x2 = arranging_pattern1_originy[j] + box_length
        box1_y2 = arranging_pattern1_originx[j] + box_width

    if arranging_pattern2[k] == "L":
        box2_x1 = arranging_pattern2_originy[k]
        box2_y1 = arranging_pattern2_originx[k]
        box2_x2 = arranging_pattern2_originy[k] + box_width
        box2_y2 = arranging_pattern2_originx[k] + box_length

        distance = box2_y1 + (box_length/2)

    else:
        box2_x1 = arranging_pattern2_originy[k]
        box2_y1 = arranging_pattern2_originx[k]
        box2_x2 = arranging_pattern2_originy[k] + box_length
        box2_y2 = arranging_pattern2_originx[k] + box_width

        distance = box2_y1 + (box_width/2)

    # Draw boxes on blank images
    cv2.rectangle(
        im1,
        (int(box1_x1), int(box1_y1)),
        (int(box1_x2), int(box1_y2)),
        [255, 255, 255],
        -1)
    cv2.rectangle(
        im2,
        (int(box2_x1), int(box2_y1)),
        (int(box2_x2), int(box2_y2)),
        [255, 255, 255],
        -1)

    # Get percentage overlap
    length_x1 = box1_x2-box1_x1
    length_y1 = box1_y2-box1_y1

    length_x2 = box2_x2-box2_x1
    length_y2 = box2_y2-box2_y1

    length_xa = (max(box1_x1, box2_x1, box1_x2, box2_x2) -
                 min(box1_x1, box2_x1, box1_x2, box2_x2))
    length_ya = (max(box1_y1, box2_y1, box1_y2, box2_y2) -
                 min(box1_y1, box2_y1, box1_y2, box2_y2))

    if ((length_x1+length_x2 > length_xa) and
            (length_y1+length_y2 > length_ya)):

        x1 = min(box1_x2, box2_x2)
        y1 = min(box1_y2, box2_y2)
        x2 = max(box1_x1, box2_x1)
        y2 = max(box1_y1, box2_y1)

        o_area = (x1 - x2) * (y1 - y2)

        cv2.rectangle(
            im1,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            [0, 0, 255],
            -1)

    else:
        o_area = 0

    area_ref = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    per_area = (o_area/area_ref) * 100

    # Rotate images
    im1 = cv2.rotate(im1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    im2 = cv2.rotate(im2, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # If overlap percentage is more than threshold
    # do further processing
    if per_area > 5.0:

        # Check if base box is alrady detected or not
        if (((boxes_stat[i+1][j] == 1) or
             (boxes_stat[i+1][j] == 2)) and
                (boxes_stat[i][k] == 0)):

            # Set the base box stat to "2"
            boxes_stat[i][k] = 2

            # Estimate depth and height value for
            # base box
            depth = box2_x1 + round(box_width/4, 2)
            height = (i+1) * box_height

            # Create a new row
            newrow = np.array(
                [[depth, height, distance, i+1, 0.0, 0.0, 5.0]],
                dtype="float32")

            # Add newley create row to "Box_Pos_Info"
            Box_Pos_Info.boxes_details = np.vstack(
                [Box_Pos_Info.boxes_details, newrow])

            # Visualize newley added point
            cv2.rectangle(
                im3,
                (int(box2_x1), int(box2_y1)),
                (int(box2_x2), int(box2_y2)),
                [255, 255, 255],
                -1)
            cv2.rectangle(
                im3,
                (int(depth - 2), int(distance - 2)),
                (int(depth + 2), int(distance + 2)),
                [255, 0, 0],
                -1)
            cv2.rectangle(
                im3,
                (int(box2_x1 - 10), int(box2_y1 - 10)),
                (int(box2_x1), int(box2_y1)),
                [0, 0, 255],
                -1)
            im3 = cv2.rotate(im3, cv2.ROTATE_90_COUNTERCLOCKWISE)

            """cv2.imshow('im1',im1)
            cv2.imshow('im2',im2)
            cv2.imshow('im3',im3)
            cv2.waitKey(0)"""


def verify_box_presence(x_list,
                        index,
                        y_list,
                        dim_length,
                        dim_width,
                        m_fact,
                        color_list_for_boxes,
                        level_boxes_im_copy1,
                        level_boxes_im_copy2,
                        boxes_stat,
                        count,
                        level):

    """
       Sets flag for visible boxes

       Arguments:
        x_list: List containing the x co-ordinate of box
        index:Gives an index of a particular box placed on the pallet
        y_list: List containing the y co-ordinate of box
        dim_length:Length of top-view image in pixels
        dim_width: Width of top-view image
        m_fact: magnifying factor
        color_list_for_boxes: List that will tell which index belong
            to which color
        level_boxes_im_copy1: Check if point belongs to painted region or not
        level_boxes_im_copy2: Just to visualize which box is being processed
        boxes_stat:For getting SKU id and assocaited stacking pattern details
        count: count of boxes
        level: level of a box

        Returns:
            boolean to verify box presence

    """

    # Get the (y, x) co-ordinates of box
    (y, x) = get_x_and_y_method1(
        x_list[index],
        y_list[index],
        dim_width,
        m_fact)

    # Check if point lies inside the pallet
    if (y >= (dim_length * m_fact) or
            y < 0 or x >= (dim_width * m_fact) or
            x < 0):

        return

    # Get the index that belong to this color
    idd = np.where(
        (color_list_for_boxes == level_boxes_im_copy1[x, y, :]).all(axis=1))

    # Set flag for this box to visible
    boxes_stat[level-1, idd] = 1


def get_added_value_for_xy(x_list,
                           y_list,
                           j,
                           dim_length,
                           dim_width,
                           floor_img_list,
                           level,
                           all_points,
                           all_average,
                           color_list,
                           floor_img_list_ref,
                           m_fact):

    """

        Adds all the co-ordinate values belonging to a box

        Args:
           x_list:List containing the x co-ordinate of box
           y_list:List containing the y co-ordinate of box
           j:
           dim_length: image length in pixels
           dim_width: image width in pixels
           floor_img_list: list of floor plan images
           level: level of a box
           all_points: points from each image on the floor plan
           all_average: avg of the all points plotted on the floor plan
           color_list: list for color indexing of boxes
           floor_img_list_ref: reference of the initial floor img list
           m_fact: magnifying factor

          Returns:

    """
    (y, x) = get_x_and_y_method1(x_list[j], y_list[j], dim_width, m_fact)

    if y < ((dim_length * m_fact) or
            y >= 0 or
            x < (dim_width * m_fact) or
            x >= 0):

        # Get the index that belong to this color
        idd = np.where(
            (color_list == floor_img_list_ref[level - 1][x, y, :]).all(axis=1))

        all_points[level - 1, idd[0], 0] = (
            all_points[level - 1, idd[0], 0] +
            x_list[j])
        all_points[level - 1, idd[0], 1] = (
            all_points[level - 1, idd[0], 1] +
            y_list[j])

        all_average[level - 1, idd[0], 0] = (
            all_average[level - 1, idd[0], 0] +
            1)
        all_average[level - 1, idd[0], 1] = (
            all_average[level - 1, idd[0], 1] +
            1)

        # floor_img_list[level - 1][x-10:x+10, y-10:y+10, :] = [255, 255, 255]
        cv2.putText(
            floor_img_list[level - 1],
            ".",
            (y, x),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            2)


def get_unique_se_of_3d_positions(pallet_img_wise_list,
                                  boxes_img_wise_list,
                                  boxes_3d_info_img_wise_list,
                                  sku,
                                  flag,
                                  color_odd,
                                  color_even,
                                  m_fact,
                                  sub_folder_name,
                                  img_name, debug):

    """

        Combines 3D position values got from all the images
        and returns unique set of 3D positions

        Args:
              pallet_img_wise_list: list of pallets for all images
              boxes_img_wise_list: list of sku's/boxes for images
              boxes_3d_info_img_wise_list: list of all boxes 3D pose info
              sku: sku info
              flag: flag info for even/odd
              color_odd: unique color of pattern1
              color_even: unique colors of patten2
              m_fact: magnifying factor
              sub_folder_name: sub folder directory to save img
              img_name: image name for saving purposes
              debug: debug flag to select writing to directory or not

        Returns:
         unique box 3D pos info

    """

    # Get the number of images
    no_of_images = len(pallet_img_wise_list)

    # Get image dimension for odd level
    (dim_length1, dim_width1) = get_complete_length(
        sku.arranging_pattern1,
        sku.arranging_pattern1_originx,
        sku.arranging_pattern1_originy,
        sku.length_wise_max_level1,
        sku.width_wise_max_level1,
        sku.box_length,
        sku.box_width)
    dim_length1 = int(math.ceil(dim_length1))
    dim_width1 = int(math.ceil(dim_width1))

    # Get image dimension for even level
    (dim_length2, dim_width2) = get_complete_length(
        sku.arranging_pattern2,
        sku.arranging_pattern2_originx,
        sku.arranging_pattern2_originy,
        sku.length_wise_max_level2,
        sku.width_wise_max_level2,
        sku.box_length,
        sku.box_width)
    dim_length2 = int(math.ceil(dim_length2))
    dim_width2 = int(math.ceil(dim_width2))

    # Create blank images
    odd_level_boxes_im = np.zeros(
        (dim_length1 * m_fact, dim_width1 * m_fact, 3),
        np.uint8)  # for odd level
    even_level_boxes_im = np.zeros(
        (dim_length2 * m_fact, dim_width2 * m_fact, 3),
        np.uint8)  # for even level

    # Draw images
    odd_level_boxes_im = draw_boxes_m_fix(
        odd_level_boxes_im,
        sku.box_length,
        sku.box_width,
        sku.arranging_pattern1,
        sku.arranging_pattern1_originx,
        sku.arranging_pattern1_originy,
        color_odd,
        m_fact)
    even_level_boxes_im = draw_boxes_m_fix(
        even_level_boxes_im,
        sku.box_length,
        sku.box_width,
        sku.arranging_pattern2,
        sku.arranging_pattern2_originx,
        sku.arranging_pattern2_originy,
        color_even,
        m_fact)

    # Create one floor plan image for each level
    max_level = 0
    for i in range(0, no_of_images):

        # Get level details
        level_details = boxes_3d_info_img_wise_list[i].boxes_details[:, 3]

        # Find max levels detected among all images
        if max_level < max(level_details):
            max_level = max(level_details)

    print("Max Level Detected = ", max_level)

    floor_img_list = []
    floor_img_list_ref = []
    if flag == "odd":
        for i in range(1, int(max_level) + 1):
            if i % 2 == 1:
                floor_img_list.append(odd_level_boxes_im.copy())
                floor_img_list_ref.append(odd_level_boxes_im.copy())
            else:
                floor_img_list.append(even_level_boxes_im.copy())
                floor_img_list_ref.append(even_level_boxes_im.copy())

    if flag == "even":
        for i in range(1, int(max_level) + 1):
            if i % 2 == 1:
                floor_img_list.append(even_level_boxes_im.copy())
                floor_img_list_ref.append(even_level_boxes_im.copy())
            else:
                floor_img_list.append(odd_level_boxes_im.copy())
                floor_img_list_ref.append(odd_level_boxes_im.copy())

    # Plot point from each image on floor plan
    all_points = np.zeros(
        (int(max_level), len(sku.arranging_pattern1), 2),
        dtype="float32")

    # Get average of all the points
    all_average = np.zeros(
        (int(max_level), len(sku.arranging_pattern1), 2),
        dtype="float32")

    for i in range(0, no_of_images):

        # Get level details
        level_details = boxes_3d_info_img_wise_list[i].boxes_details[:, 3]

        # Get maximum level detected
        max_level = max(level_details)

        # Process position information level wise
        for level in range(1, int(max_level+1)):

            # print ("Level = ", level+1)

            # Get all points belong to this cluster
            pts = np.where((level_details == level))

            # Variabels to store info related to all
            # boxes for this particular level
            x_list = []
            y_list = []

            # Get all the boxes belonges to this particular level
            for pt in pts[0]:
                x_list.append(
                    boxes_3d_info_img_wise_list[i].boxes_details[pt, 2])
                y_list.append(
                    boxes_3d_info_img_wise_list[i].boxes_details[pt, 0])

            for j in range(0, len(x_list)):

                if flag == "odd":

                    if level % 2 == 0:
                        get_added_value_for_xy(
                            x_list,
                            y_list,
                            j,
                            dim_length2,
                            dim_width2,
                            floor_img_list,
                            level,
                            all_points,
                            all_average,
                            color_even,
                            floor_img_list_ref,
                            m_fact)

                    else:
                        get_added_value_for_xy(
                            x_list,
                            y_list,
                            j,
                            dim_length1,
                            dim_width1,
                            floor_img_list,
                            level,
                            all_points,
                            all_average,
                            color_odd,
                            floor_img_list_ref,
                            m_fact)

                if flag == "even":

                    if level % 2 == 0:
                        get_added_value_for_xy(
                            x_list,
                            y_list,
                            j,
                            dim_length1,
                            dim_width1,
                            floor_img_list,
                            level,
                            all_points,
                            all_average,
                            color_odd,
                            floor_img_list_ref,
                            m_fact)

                    else:
                        get_added_value_for_xy(
                            x_list,
                            y_list,
                            j,
                            dim_length2,
                            dim_width2,
                            floor_img_list,
                            level,
                            all_points,
                            all_average,
                            color_even,
                            floor_img_list_ref,
                            m_fact)

    length_non_zero = np.nonzero(all_average[:, :, 0])
    length_non_zero = length_non_zero[0]
    length_non_zero = len(length_non_zero)

    all_average[all_average == 0] = 1

    all_points_average = all_points/all_average

    pos = Box_Pos_Info()
    pos.boxes_details = np.zeros((length_non_zero, 7), dtype="float32")

    level = 0
    i = 0
    for val1 in all_points_average:

        level = level + 1

        for val2 in val1:

            if val2[0] != 0 or val2[1] != 0:

                pos.boxes_details[i][0] = val2[1]
                pos.boxes_details[i][2] = val2[0]
                pos.boxes_details[i][3] = level
                i = i+1

                (y, x) = get_x_and_y_method1(
                    val2[0],
                    val2[1],
                    dim_width1,
                    m_fact)
                cv2.putText(
                    floor_img_list[level - 1],
                    ".",
                    (y, x),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 0, 255),
                    2)

                """# Show result
                cv2.namedWindow("floor_img_list[level]", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("floor_img_list[level]", 600,600)
                cv2.imshow("floor_img_list[level]", floor_img_list[level - 1])
                cv2.waitKey(0)"""

    # Show + save output images
    i = 1
    for floor_img in floor_img_list:

        """print ("Level Image = ", i)
        cv2.namedWindow("floor_img_list[level]", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("floor_img_list[level]", 600,600)
        cv2.imshow("floor_img_list[level]", floor_img)
        cv2.waitKey(0)"""

        strr_plot = (sub_folder_name +
                     "/" +
                     img_name +
                     "_ALL_" +
                     str(i) +
                     ".PNG")
        if debug == "debug":
            cv2.imwrite(strr_plot, floor_img)

        i = i + 1

    # print ("GetUniqueSeOf3DPosition - End")

    return pos

def get_count_pallet_wise(pallet_id, dict_pallet_wise_list,
                          result_dir, sku_ref_pts_list,
                          cornersS_final_list, L_box_ref_moved_list,
                          preprocessing_imglist, depth_img_list,
                          img_name_list, debug, input_image_barcode_list, barcode_method):

    """
        Returns total count pallet wise

        Args:
            pallet_id - Pallet No
            dict_pallet_wise_list - Dictionary of image name array
                    indexed pallet no. wise
            result_dir - Path to write count results
            sku_ref_pts_list: list having the sku reference points(tvecsS)
                    data for all boxes
            cornersS_final_list: list of box corner coordinates for all
                    true boxes detected
            L_box_ref_moved_list: list for having boxes reference points after
                    shifting it up. also has the depth values
            preprocessing_imglist: list of images/frames for which
                    pre-processing checks are satisfied
            depth_img_list: list of the depth images
            img_name_list: list of the string image names
            debug: debugging flag set on/off

        Returns:
            null if empty pallet or writes total count of boxes into file
    """
    # Variable to store image specific information
    m_fact = 10
    pallet_img_wise_list = []
    #rack_no_img_wise_list = []
    boxes_img_wise_list = []
    sku_img_wise_list = []
    boxes_3d_info_img_wise_list = []
    color_odd_img_wise_list = []
    color_even_img_wise_list = []
    flag_img_wise_list = []
    detail_json = collections.OrderedDict()
    stock_dir = "../output/processed_images"
    if not os.path.exists(stock_dir):
        os.mkdir(stock_dir)

    # Create a location inside mission to save
    # pallet wise count results
    sub_folder_name = result_dir+"/"+"pallet_"+str(pallet_id)
    os.makedirs(sub_folder_name,exist_ok=True)

    # Get the image list for the pallet
    pallet_wise_img_list = dict_pallet_wise_list[pallet_id]

    # To keep record of no. of box ids image wise
    ind = 0
    ind_final = 0
    total_box_detected = 0
    # test_dir=result_dir
    # test_dir = test_dir.split("/")
    # test_dir[-1] = "MaskedImages"
    # test_dir = '/'.join(test_dir)
    # depth_img_list = blurdetection.depth_img_list
    #print(depth_img_list)
    count = 0#Counter that will take care of count of images
    # Process each image one by one
    
    #Step-0 Getting SKU Information
    for barcode_img_name in input_image_barcode_list:
        barcode_info = {}
        barcode_img = cv2.imread(barcode_img_name)
        _, barcode_info = detect_barcode_opencv_exp.detect_bar_code(barcode_img,barcode_method)
        print("Dictionary : ",barcode_info)
        cornersP_barcode = []
        barcode_id = []
        for text, corners in barcode_info.items():
            cornersP_barcode.append(corners)
            barcode_id.append('-'.join(text.split('-')[:-1]))

        #print("cornersP_barcode: ",cornersP_barcode)
        #print("barcode_id:",barcode_id)

    
    # csv file name
    filename = "box_count/tools/Dictionary/SKU_marker_mapping.csv"
    df = pd.read_csv(filename)
    SKU_name = df[df['Marker Mapping'] == 2]['SKU code']#This 2 will be replaced by barcode_id[0]
    #print(type(SKU_name))
    L_SKU_name = str(SKU_name).split("\n")
    SKU_name = L_SKU_name[0]
    L_SKU_name = []
    L_SKU_name = str(SKU_name).split(" ")
    #print("L_SKU_name: ",L_SKU_name)
    SKU_name = L_SKU_name[4] + L_SKU_name[5]
    print("SKU_name: ",SKU_name)

    for image_name in pallet_wise_img_list:
        # Process each image belongs to this
        # particular pallet id

        # Get the complete path of the image
        if image_name in img_name_list:
            complete_image_name = img_name_list[count]
        #print("Image Name = ", complete_image_name)
        # Read image and convert it to gray scale
        frame = preprocessing_imglist[count]


        # Apply gaussian blurr
        """image = cv2.GaussianBlur(frame,(5,5), 3)
        frame = cv2.addWeighted(frame, 1.0, image, -0.5, 7)"""
        """
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("frame", 600,600)
        cv2.imshow("frame", frame)
        cv2.waitKey(0)
        """
        #Gray_Scale image is used for detecting pallet aruco amrker
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize and save input image
        output_img_file = (sub_folder_name +
                           "/" +
                           image_name[0:len(image_name)-4] +
                           ".PNG")
        temp = imutils.resize(frame, width=500)
        if debug == "debug":
            cv2.imwrite(output_img_file, temp)

        # File handler to record error, if any occurs
        output_file = (sub_folder_name +
                       "/" +
                       image_name[0:len(image_name)-4] +
                       ".txt")
        file_handler = open(output_file, "w+")

        # Step 0 - Preparation
        json_object = json.dumps(detail_json)
        name_json = sub_folder_name + "/" + str(pallet_id) + "_score.json"

        with open(name_json, "w") as outfile:
            # Create dummy "json" file
            # without any information

            outfile.write(json_object)

        #detail_json['Rack_ID'] = int(0)

        # Step 1 - Pallet aruco marker detection +
        #          Process pallet's aruco marker +
        #          Process rack aruco marker
        pallet = process_pallet_markers(
            gray,
            Dictionary_Marker(),
            Camera_Info(),
            file_handler,
            detail_json)

        if pallet == None:
            continue
        #It is returning
        """
        pallet.pallet_aruco_depth_tvec = tvecsP[0][0][2]*100
        pallet.pallet_aruco_height_tvec = tvecsP[0][0][1]*100
        pallet.pallet_aruco_distance_tvec = tvecsP[0][0][0]*100
        pallet.cornersP = cornersP
        pallet.idsP = idsP
        """
        #rack_no = process_rack_marker(
            #gray,
            #pallet,
            #Dictionary_Marker(),
            #file_handler,
            #detail_json)
        print("Pallets:-",pallet)
        print("Pallet idsP: ",pallet.idsP)
        print("Pallet-aruco marker depth wrt camera:- ",pallet.pallet_aruco_depth_tvec)
        print("Pallet-aruco marker height wrt camera:- ",pallet.pallet_aruco_height_tvec)
        print("Pallet-aruco marker distance wrt camera:- ",pallet.pallet_aruco_distance_tvec)
        print("Pallet-Marker Corners are :- ",pallet.cornersP)

        #pallet_img_wise_list.append(pallet)
        #rack_no_img_wise_list.append(rack_no)

        # Step 2 - SKU aruco markers detection +
        #          Process SKU's aruco markers
        # co = complete_image_name.split('/')
        boxes = process_box_markers(count,
            file_handler,
            detail_json, sku_ref_pts_list,
            cornersS_final_list, L_box_ref_moved_list, 2, SKU_name)#We will pass barcode_id[0] instead of 2

        count+=1
        if boxes == None:
            continue

        if total_box_detected > len(boxes.cornersS):
            ind_final = ind

        ind = ind + 1

        pallet_img_wise_list.append(pallet)
        #rack_no_img_wise_list.append(rack_no)
        boxes_img_wise_list.append(boxes)
        #print("Boxes:-",boxes)
        #print("Box corners: ",boxes.cornersS)
        #print("Box corners.shape: ",(boxes.cornersS[0]).shape)
        #print("Box .rvecsS: ",boxes.rvecsS)
        #print("Box tvecsS: ",(boxes.tvecsS)*100)
        #print("Box TYPE OF tvecsS: ",type(boxes.tvecsS))
        #print("Box Shape OF tvecsS: ",(boxes.tvecsS).shape)
        #print("Box unique_ids: ",boxes.unique_ids)

        # Step 3 - Read dictionary info for this
        #          detected box id and define
        #          threshold values
        sku = read_sku_info_from_dictionary(boxes.unique_ids, file_handler)
        thres_depth = int(sku.box_width/2)
        thres_height = int(sku.box_height/2)
        sku_img_wise_list.append(sku)

        # Step 4 - Process each detected SKU aruco
        #          marker to get the (x, y, z) info
        boxes_3d_info = get_box_pos_information(frame, boxes, pallet, sku)
        #print("Boxes_3d_info as compared to pallet:- ",boxes_3d_info.boxes_details)
        boxes_3d_info_img_wise_list.append(boxes_3d_info)
        #It is returning
        """
        pos.im_point_depth.append((box_depth_detected, 0))
        pos.im_point_height.append((box_height_detected, 0))
        pos.im_point_distance.append((box_distance_detected, 0))

        """
        result_name = (sub_folder_name +
                       "/" +
                       image_name[0:len(image_name)-4] +
                       "_3D.PNG")
        temp = imutils.resize(frame, width=1000)
        if debug == "debug":
            cv2.imwrite(result_name, temp)
        #cv2.namedWindow("Img", cv2.WINDOW_NORMAL)
        #cv2.resizeWindow("Img", 400, 400)
        #cv2.imshow("Final-Output Image:-",temp)
        #cv2.waitKey(0)

        # Step 5 - Apply Clustering algorithm - Level Wise
        cluster_level_wise(
            frame,
            boxes,
            boxes_3d_info,
            thres_height,
            detail_json)

        # Step 7 - Generate unique color for each box
        color_odd = \
            generate_unique_color_for_each_box(len(sku.arranging_pattern1))
        color_even = \
            generate_unique_color_for_each_box(len(sku.arranging_pattern2))
        color_odd_img_wise_list.append(color_odd)
        color_even_img_wise_list.append(color_even)

        # Step 8 - Decide level
        (error_odd_start, error_even_start, closeness_odd_start,
         closeness_even_start) = decide_level_pattern(boxes_3d_info,
                                                      sku,
                                                      color_odd,
                                                      color_even,
                                                      m_fact)

        if error_odd_start == error_even_start:
            flag1 = "no"

        if error_odd_start < error_even_start:
            flag1 = "odd"

        if error_even_start < error_odd_start:
            flag1 = "even"

        if closeness_odd_start == closeness_even_start:
            flag2 = "no"

        if closeness_odd_start > closeness_even_start:
            flag2 = "odd"

        if closeness_even_start > closeness_odd_start:
            flag2 = "even"

        if (flag1 == flag2):
            flag = flag1
        else:
            if (abs(error_odd_start - error_even_start) >
                    abs(closeness_odd_start - closeness_even_start)):
                flag = flag1
            else:
                flag = flag2

        flag_img_wise_list.append(flag)



    # Get the complete path of the image
    #name = pallet_wise_img_list[ind_final]
    #complete_image_name = (input_dir +"/" +name[0:len(name)-4] +".PNG")

    #frame_t = cv2.imread(complete_image_name)

    #input_dir_t = input_dir.split("/")
    #input_dir_t = input_dir_t[0:len(input_dir_t) - 1]
    #input_dir_t = "/".join(input_dir_t)
    #input_dir_t = input_dir_t + "/" + str(pallet_id) + "_thumbnail.PNG"
    #input_dir_t = input_dir_t + "/" + str(pallet_id) + ".jpg"
    #input_dir_o = input_dir_t + "/" + str(pallet_id) + ".JPG"

    #cv2.imwrite(input_dir_o, frame_t)
    #frame_t = imutils.resize(frame_t, width=100)

    #cv2.imwrite(input_dir_t, frame_t)

    # Condition to check empty pallet
    if len(flag_img_wise_list) == 0:

        detail_json['box_count'] = int(0)

        print("Total Count = 0")

        json_object = json.dumps(detail_json)
        name_json = sub_folder_name + "/" + str(pallet_id) + "_score.json"
        with open(name_json, "w") as outfile:
            outfile.write(json_object)

        return

    # Step 9 - Get final flag value and color_odd + color_even
    odd_count = flag_img_wise_list.count("odd")
    even_count = flag_img_wise_list.count("even")

    if odd_count > even_count:
        flag = "odd"
        index_color = flag_img_wise_list.index("odd")
    else:
        flag = "even"
        index_color = flag_img_wise_list.index("even")

    sku_all = sku_img_wise_list[index_color]
    color_odd_all = color_odd_img_wise_list[index_color]
    color_even_all = color_even_img_wise_list[index_color]


    #print("flag = ", flag)

    # Step 10 - Get unique set of 3D position information
    #           from multiple images
    boxes_3d_info_all = get_unique_se_of_3d_positions(
        pallet_img_wise_list,
        boxes_img_wise_list,
        boxes_3d_info_img_wise_list,
        sku_all,
        flag,
        color_odd_all,
        color_even_all,
        m_fact,
        sub_folder_name,
        "level", debug)

    # Step 11 - Downward Verification
    """downward_verification(
        boxes_3d_info_all,
        sku_all,
        color_odd_all,
        color_even_all,
        sub_folder_name,
        "abc",
        m_fact,
        flag,
        pallet)
    """

    # Step 12 - Calculate confidance score
    score_matrix = calculate_confidance_score(
        boxes_3d_info_all,
        sku_all,
        color_odd_all,
        color_even_all,
        sub_folder_name,
        "abc",
        m_fact,
        flag)

    # Step 13 - Generate shadowed image for each level
    (box_on_off_list, Level_wise_img_stat) = generate_shadowed_image(
        boxes_3d_info_all,
        sku_all,
        color_odd_all,
        color_even_all,
        sub_folder_name,
        "level",
        m_fact,
        flag, debug)

    # Step 14 - Get list of hidden boxes
    (boxes_midpos_List, box_on_off_list) = get_hidden_boxes(
        boxes_3d_info_all,
        m_fact,
        box_on_off_list,
        sku_all,
        color_odd_all,
        color_even_all,
        Level_wise_img_stat,
        flag)

    image_output = img_name_list[-1]
    image = cv2.imread(image_output)

    # Step 15 - Count total number of boxes
    (total_count, count_matrix, final_img_path) = count_total_number_of_boxes(
        boxes_3d_info_all,
        sku_all,
        boxes_midpos_List,
        box_on_off_list,
        sub_folder_name,
        "level",
        file_handler,
        flag,
        score_matrix,
        detail_json,
        boxes,image,temp,stock_dir,pallet_id)
    print("*******************************************************")
    print("*******************************************************")
    print("*******************************************************")
    print("final_img_path: ",final_img_path)
    print("*******************************************************")
    print("*******************************************************")
    print("*******************************************************")
    #temp = cv2.putText(temp, 'Total-Count of Boxes:-'+str(total_count), (30,30), cv2.FONT_HERSHEY_SIMPLEX,
                   #1, (0,0,255),2, cv2.LINE_AA)
    if debug == "debug":
        cv2.imwrite(result_name, temp)
    # Step 16 - Save json file
    detail_json['box_count'] = int(total_count)
    detail_json['img_path'] = final_img_path
    json_object = json.dumps(detail_json)
    name_json = sub_folder_name + "/" + str(pallet_id) + "_score.json"
    with open(name_json, "w") as outfile:
        outfile.write(json_object)

    # Step 17 - Delete variables
    del (boxes_3d_info_all,
         sku_all,
         color_odd_all,
         color_even_all,
         score_matrix,
         box_on_off_list,
         Level_wise_img_stat,
         boxes_midpos_List,
         total_count,
         count_matrix)

    # deleting residual files
    if debug != 'debug':
        filelist = [ f for f in os.listdir(sub_folder_name) if f.endswith(".txt") ]
        for f in filelist:
            os.remove(os.path.join(sub_folder_name, f))

    # Step-18 Json Compilation
    #L = json_compile.func()

    # Step-19 Vision Report Sender

