import numpy as np
import cv2.aruco as aruco
from scipy.spatial import distance as dist
import cv2
import os
CAMERAMATRIX = [[1367.27 , 0, 950.995],
                [0, 1365.57, 564.756],
                [0, 0, 1]]
DISTCOEFFS = [[0.13555811, -0.54607789, 0.00108346, -0.00431513, 0.52654226]]
CAMERAMATRIX = np.array(CAMERAMATRIX)
DISTCOEFFS = np.array(DISTCOEFFS)
camera_matrix = [CAMERAMATRIX, DISTCOEFFS]
counter = 0
corners4 = []


def order_points_old(pts):

    """

        Description:-
        For ordering, compute the sum and difference between the points

        Arguments:
        pts:- List containing corner points

        Returns ordered coordinates


    """

    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def pix_to_cm(box_all_cordinates, marker_length):

    """

        Description:-
        Pixel to distance calculator. It's calculating
        the euclidean distance between two corners in pixels
        and dividing by marker length which is in cms

        Arguments:
        box_all_cordinates:- List containing corner points of a particular box
        marker_length:- Marker Length of the pallet marker

        Returns the conversion factor of pixels to cm.

    """

    pts = np.zeros((4, 2), dtype="float32")
    pts[0, :] = box_all_cordinates[0][0]
    pts[1, :] = box_all_cordinates[0][1]
    pts[2, :] = box_all_cordinates[0][2]
    pts[3, :] = box_all_cordinates[0][3]
    rect = order_points_old(pts)

    corner1 = rect[0, :]
    corner2 = rect[1, :]

    e_distance = dist.euclidean(corner1, corner2)
    one_pix_cm = marker_length / e_distance

    return one_pix_cm


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

        Arguments:
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


def get_box_reference_point(corner, depthImage):

    """

        Description:-
        Calculates the reference point after moving the box centre up,
        to the point where sku marker was supposed to be there

        Arguments:-
        corner:- List containing corners of the box
        depthImage:- Depth Image in form of numpy array

        Returns xref,yref and z for a box in cms.

    """

    xcentre, ycentre = get_center(corner)
    vtop1, vtop2 = corner[0][0], corner[0][1]
    z = get_box_depth(depthImage, xcentre, ycentre)
    yhigh = max(vtop1[1], vtop2[1])
    ycorrection = (ycentre-yhigh) * 0.75
    xref, yref = int(xcentre), int(ycentre-ycorrection)
    # print("ycorrection applied:", ycorrection)
    return (xref, yref, z)


def check_ratio(corner):

    """

        Description:-
        Gives insight about bounding boxes by checking the limits of
        ratios of length and width of boxes, so as to remove
        false boxes.

        Arguments:-
        corner:- List containing corners of the box

        Returns in Boolean True or False

    """

    L = 0  # Length is vertical dimension
    W = 0  # Length is horizontal dimension
    L = corner[0][0][1]-corner[0][3][1]
    W = corner[0][0][0]-corner[0][1][0]
    ratio = W/L
    if ((ratio > 2.9) or (ratio < 0.9)):
        return True


def check_depth(z_pallet_center_camera, z):

    """

        Description:-
        Checking whether the range of depth values of each box
        lie in optimum interval or not.

        Arguments:-
        z_pallet_center_camera:- Depth of pallet
        z:- depth of each box

        Returns in Boolean True or False

    """

    if z_pallet_center_camera-50 < z < z_pallet_center_camera+100:
        return True


def isFalseFace(corner, dimg, out_dir,
                cimg, img_name, debug):

    """

        Calculates standard-deviation of depth values
        of a bounding box, to detect is there any side face
        counted as a 3-dimensional box

        Arguments:-
        corner:- List containing corner coordinates of the boxes
        dimg:- Depth Image in  form of numpy array
        out_dir:- Output Directory to store the results
        cimg:- Image to be processed in  form of numpy array
        img_name:- Image name in string format

        Returns in Boolean True or False

    """

    x1 = corner[0][0][0]
    y1 = corner[0][0][1]

    x2 = corner[0][1][0]
    y2 = corner[0][1][1]

    x3 = corner[0][2][0]
    y3 = corner[0][2][1]

    x4 = corner[0][3][0]
    y4 = corner[0][3][1]

    xc, yc = get_center(corner)

    ylow = np.max([y1, y2])
    yhigh = np.min([y3, y4])
    xlow = np.max([x1, x4])
    xhigh = np.min([x2, x3])

    # taking 70% area
    yl = yc - int(0.70 * (np.absolute(yc - ylow)))
    yh = yc + int(0.70 * (np.absolute(yhigh - yc)))
    xl = xc - int(0.70 * (np.absolute(xc - xlow)))
    xh = xc + int(0.70 * (np.absolute(xhigh - xc)))

    patch = dimg[yl:yh, xl:xh]
    patch = patch[~np.isnan(patch)]
    patch = patch[patch != 0]
    patch = patch/10
    sd = np.std(patch)
    diff = (patch.max() - patch.min())
    # print("\ndiff", diff)
    # print("std:",sd)
    # return (std, diff)
    window_name = 'Image'
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (xc - 50, yc)
    org2 = (xc - 50, yc + 30)
    fontScale = 1
    color = (23, 10, 234)
    thickness = 2
    text1 = "std:" + str(np.around(sd, 2))
    text2 = "diff=" + str(np.round(diff, 2))
    image = cv2.putText(cimg, text1, org, font,
                        fontScale, color, thickness, cv2.LINE_AA)
    # image2 = cv2.putText(image, text2, org2, font, fontScale,
    # color, thickness, cv2.LINE_AA)
    # cv2.imshow(window_name, image)
    # cv2.waitKey(0)
    # Displaying the image
    # cv2.imshow(window_name, image)
    if debug == "debug":
        filename = out_dir + "/" + img_name
        cv2.imwrite(filename, image)
    if sd > 6.2:
        return True


def corners_true(file_box_centre_true,
                 file_box_corners_true,
                 corners4, inputImage,
                 depthImage, img_name,
                 z_pallet_center_camera,
                 directory_sideface,
                 counter, debug):

    """

        Description:-
        Gives only True Box's corner coordinates, their
        count and an array of x_box_ref_moved, y_box_ref_moved and depth

        Arguments:-
        cornersP:- Corners of PAllet marker
        file_box_centre_true:- Text file for storing the centre coordinates
                               of only true boxes for further computation
        corners4:- List that will comtain corners of all the boxes
                   generated by box_detecttion model
        inputImage (numpy_array): Real_sense Image captured from camera
        depthImage(numpy_array):- Depth Image
        img_name (String): Image name
        z_pallet_centre_camera:- Depth of pallet marker in cms wrt camera
        counter:- Total number of all the boxes generated
                  by box_detection model
        directory_sideface: Directory to save the results

        Returns cornersS_final(list),counter(int) and L_box_ref_moved
        (numpy_array) which will be used as a preparation for Box_count.py

    """

    # This is in pixels
    L_box_ref_moved = []
    # Temporary list for storing centre coordinates and depth value
    L_box_ref_moved_temp = []
    # List which will store corner coordinates of a true box
    cornersS_final = corners4
    # List that will store unwanted false boxes dimension wise
    L_unwanted_dim = []
    # List that will store unwanted false boxes depth wise
    L_unwanted_depth = []
    # List that will store unwanted false boxes which are side-faces
    L_unwanted_side_face = []

    i = 0
    # Copy of input_image for visualization purpose
    frame = inputImage.copy()
    if debug == "debug":
        file_box_centre_true.write("Centre-Coordinates of True boxes in: " + img_name + ":")
        file_box_centre_true.write("\n")

    for corner in corners4:
        # print(corner[0][0][1])
        (x_center_pix_box, y_center_pix_box, z) = \
            get_box_reference_point(corner, depthImage)

        # print("(x_center_pix_box, y_center_pix_box, z): ",
        # (x_center_pix_box, y_center_pix_box, z))

        if check_ratio(corner):
            L_unwanted_dim.append(i)
            counter = counter-1
            i += 1

        elif isFalseFace(corner, depthImage, directory_sideface,
                         inputImage, img_name, debug):
            L_unwanted_side_face.append(i)
            counter = counter-1
            i += 1

        elif check_depth(z_pallet_center_camera, z):
            L_box_ref_moved_temp.append(x_center_pix_box)
            L_box_ref_moved_temp.append(y_center_pix_box)
            L_box_ref_moved_temp.append(z)
            L_box_ref_moved.append(L_box_ref_moved_temp)
            if debug == "debug":
                file_box_centre_true.write("Centre-Coordinates and Depth of box " + str(i+1) + ":")
                file_box_centre_true.write("\n")
                file_box_centre_true.write(str(x_center_pix_box) + "," + str(y_center_pix_box) + "," + str(z))
                file_box_centre_true.write("\n")

            L_box_ref_moved_temp = []
            i += 1

            # print("Box Center Coordinates and Depth:",
            # x_center_pix_box, y_center_pix_box, z)
        else:
            L_unwanted_depth.append(i)
            counter = counter-1
            i += 1

    # cv2.namedWindow("Img", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Img", 400, 400)
    # cv2.imshow("Img", frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if debug == "debug":

        file_box_centre_true.write(str(L_box_ref_moved))
        file_box_centre_true.write("\n")
        file_box_corners_true.write("Total number of true boxes visible are:-" + str(len((L_box_ref_moved))))
        file_box_corners_true.write("\n")
        file_box_centre_true.write("----------------------------------------")
        file_box_centre_true.write("\n")
        file_box_centre_true.write("----------------------------------------")
        file_box_centre_true.write("\n")

    # For storing the final total list of
    # corners of false boxes that needs to be deleted.
    L_unwanted_total = []
    for i in L_unwanted_dim:
        L_unwanted_total.append(i)

    # print("Corners4:-",corners_final)
    for j in L_unwanted_depth:
        L_unwanted_total.append(j)

    for m in L_unwanted_side_face:
        L_unwanted_total.append(m)

    L_unwanted_total.sort()
    L_unwanted_total = L_unwanted_total[::-1]

    for k in L_unwanted_total:
        cornersS_final.pop(k)

    if debug == "debug":
        file_box_corners_true.write("Corner-Coordinates of True boxes in: " + img_name + ":")
        file_box_corners_true.write("\n")
        file_box_corners_true.write("Total number of true boxes visible are:-" + str(len((cornersS_final))))
        file_box_corners_true.write("\n")
        file_box_corners_true.write(str(cornersS_final))
        file_box_corners_true.write("\n")
        file_box_corners_true.write("---------------------------------")
        file_box_corners_true.write("\n")

    return L_box_ref_moved, cornersS_final, counter


def calculate_tvecsS(inputImage, img_name, depthImage, out_dir,
                     file_box_corners_true,
                     file_box_centre_true,
                     i, corners4, counter, debug):

    """

        Description:-
        Calculate tvecsS i.e. box's 3-d position information
        wrt camera in cms by shifting origin from top-left
        point to camera centre using pallet
        marker position information in cms and pixels.

        Arguments:
        testImage (numpy_array): Real_sense Image captured from camera
        img_name (String): Image name
        depthImage(numpy_array):- Depth Image
        file_box_corners:- Text file for storing
        the corner coordinates and number of visible boxes
        file_box_centre:- Text file for storing
        the centre coordinates for further computation
        out_dir:- Output Image Directory
        i:- Denotes a counter of number of images passed
        file_box_centre_true:- FIle that contains only true box corners

        Returns sku_ref_points and L_box_ref_moved which
        will be used as a preparation for Box_count.py

    """

    # SKU dictionary
    aruco_dict_sku = aruco.Dictionary_get(aruco.DICT_5X5_1000)
    # Pallet dictionary
    aruco_dict_pallet = aruco.Dictionary_get(aruco.DICT_6X6_1000)

    parameters = aruco.DetectorParameters_create()
    # Get pallet info
    PALLET_WIDTH = 121.92  # In cm
    PALLET_BREADTH = 121.92  # In cm

    # Get aruco info
    PALLET_MARKERLENGTH = 0.07  # 7 cm

    cornersP, idsP, rejectedImgPointsP = aruco.detectMarkers(
            inputImage,
            aruco_dict_pallet,
            parameters=parameters)

    (rvecsP, tvecsP, _) = aruco.estimatePoseSingleMarkers(cornersP,
                                                          PALLET_MARKERLENGTH,
                                                          CAMERAMATRIX,
                                                          DISTCOEFFS)

    tvecsP_before = tvecsP * 100
    # print("TvecsP before correction:- ",tvecsP * 100)
    img_name = img_name.split('/')
    img_name = img_name[-1]
    # Correct Points, we can verify this
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
    Y = X * T

    for i in range(len(tvecsP)):
        y = tvecsP[i][0][1]
        z = tvecsP[i][0][2]
        tvecsP[i][0][1] = Y[i, 1]
        tvecsP[i][0][2] = Y[i, 2]
    # print("TvecsP after correction:- ",tvecsP * 100)

    # It is in cms
    tvecsP = tvecsP * 100
    # Pallet coordinates wrt camera in cms
    x_pallet_center_camera = tvecsP[0][0][0]
    y_pallet_center_camera = tvecsP[0][0][1]
    z_pallet_center_camera = tvecsP_before[0][0][2]
    (corners4, counter) = corners4, counter
    # print("corners4: ", corners4)

    directory_boxdetection_corr = out_dir.split("/")
    directory_boxdetection_corr.append("boxdetection_corr")
    directory_boxdetection_corr = '/'.join(directory_boxdetection_corr)
    if debug == "debug":
        os.makedirs(directory_boxdetection_corr, exist_ok=True)

    directory_sideface = out_dir.split("/")
    directory_sideface.append("sideface")
    directory_sideface = '/'.join(directory_sideface)
    if debug == "debug":
        os.makedirs(directory_sideface, exist_ok=True)
    # print("Corners4:-",corners4)
    # print("Cornersp:-",cornersP)
    image_box = inputImage.copy()
    for corners in corners4:
        image_box = cv2.rectangle(image_box,
                                  (int(corners[0][0][0]),
                                   int(corners[0][0][1])),
                                  (int(corners[0][2][0]),
                                   int(corners[0][2][1])),
                                  tuple((0, 0, 255)), 1)

    for corner in cornersP:
        (x_center_pix, y_center_pix) = get_center(corner)
    # print("x_center_pix, y_center_pix",x_center_pix, y_center_pix)

    (L_box_ref_moved, cornersS_final, counter) = \
        corners_true(
                     file_box_centre_true,
                     file_box_corners_true,
                     corners4, inputImage,
                     depthImage, img_name,
                     z_pallet_center_camera,
                     directory_sideface,
                     counter, debug)

    # print("i:",i)
    for corners in cornersS_final:
        image_box = cv2.rectangle(image_box, (int(corners[0][0][0]),
                                  int(corners[0][0][1])),
                                  (int(corners[0][2][0]),
                                  int(corners[0][2][1])),
                                  tuple((0, 255, 0)), 2)
        if debug == "debug":
            filename = directory_boxdetection_corr + "/" + img_name
            cv2.imwrite(filename, image_box)

    # print("L_box_ref_moved:", L_box_ref_moved)
    # print("counter: ", counter)

    cornersP = np.array(cornersP)
    cornersP = cornersP.reshape((1, 4, 2))

    multiplication_factor = pix_to_cm(cornersP, PALLET_MARKERLENGTH*100)

    x_pallet_center_origin_cms = x_center_pix * multiplication_factor
    y_pallet_center_origin_cms = y_center_pix * multiplication_factor

    # Calculating the camera-centre coordinates
    x_camera_ref_cm = x_pallet_center_origin_cms - x_pallet_center_camera
    y_camera_ref_cm = y_pallet_center_origin_cms - y_pallet_center_camera

    # print("x_camera_ref_cm: ",x_camera_ref_cm)
    # print("y_camera_ref_cm: ",y_camera_ref_cm)
    x_camera_ref_pix = x_camera_ref_cm * (1/multiplication_factor)
    y_camera_ref_pix = y_camera_ref_cm * (1/multiplication_factor)
    x_camera_ref_pix = int(x_camera_ref_pix)
    y_camera_ref_pix = int(y_camera_ref_pix)

    # Centre-Coordinates of box-centers in image:
    # This is wrt top-left origin
    sku_ref_pts = []
    sku_ref_pts = np.array(L_box_ref_moved)
    sku_ref_pts = sku_ref_pts.reshape((len(L_box_ref_moved), 1, 3))
    # print(sku_ref_pts)
    # We are calculating the boxes 3-d info wrt camera
    for i in range(0, counter):
        sku_ref_pts[i][0][0] = int(sku_ref_pts[i][0][0])
        sku_ref_pts[i][0][1] = int(sku_ref_pts[i][0][1])
        sku_ref_pts[i][0][2] = int(sku_ref_pts[i][0][2])

    # We are calculating the boxes 3-d info wrt camera
    i=0
    for x in sku_ref_pts: 
        sku_ref_pts[i][0][0]=  x[0][0]- x_camera_ref_pix
        sku_ref_pts[i][0][1]= x[0][1] - y_camera_ref_pix
        i+=1

    """
    for i in range(0, counter):
        for j in range(0, 2):
            if j % 2 == 0:
                sku_ref_pts[i][0][j] = sku_ref_pts[i][0][j] - x_camera_ref_pix
            else:
                sku_ref_pts[i][0][j] = sku_ref_pts[i][0][j] - y_camera_ref_pix
    

    for i in range(0, counter):
        for j in range(0, 2):
            if j % 2 == 0:
                sku_ref_pts[i][0][j] = \
                    sku_ref_pts[i][0][j] * multiplication_factor
            else:
                sku_ref_pts[i][0][j] = \
                    sku_ref_pts[i][0][j] * multiplication_factor
    """
    i=0
    for x in sku_ref_pts:
        sku_ref_pts[i][0][0]=  x[0][0]*(multiplication_factor)
        sku_ref_pts[i][0][1]= x[0][1]* (multiplication_factor)
        i+=1  

    return (sku_ref_pts, cornersS_final, L_box_ref_moved)
