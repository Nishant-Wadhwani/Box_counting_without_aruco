# 1.Blur Detection
# 2.Gamma Correction
# 3.Pallet Marker Presence Condition
# 4.Pallet Dimesnions Check Condition
# 5.Mask Image
# 6.Multiple Pallet Markers Condition check
# 7.Type Error needs to be added for the depth image

# Necessary Libraries Imported
import cv2
import imutils
import numpy as np
import os
import cv2.aruco as aruco
from scipy.spatial import distance as dist


def blur_detection(image, debug, result_dir, image_name):

    # Get edge sharpness
    """

        Description:-
        Returns the Laplacian Operator that computes the Laplacian
        of the image and then return the focus measure,
        which is simply the variance of the Laplacian.

        Arguments:
        image:- Image in form of numpy array
        debug:- Debug Mode or Normal Mode
        result_dir:-  
        image_name

        Returns a boolean flag as 1 if image is non-blur

    """

    img_resize = imutils.resize(image, width=500)
    gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, 50, cv2.CV_64F).var()
    text = "Non-Blurry"
    flag_nonblur = 0

    if laplacian_var < 200:

        # Defining the threshold,
        # The Laplacian highlights regions of an image
        # containing rapid intensity changes
        text = "Blurry"
        # print("Blurry")
        flag_nonblur = 0
        # cv2.putText(
        # image,
        # "{}: {:.2f}".format(text, laplacian_var),
        # (30, 30),
        # cv2.FONT_HERSHEY_SIMPLEX,
        # 1,
        # (255, 0, 0),
        # 5)
        # cv2.namedWindow("Img", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("Img", 600, 600)
        # cv2.imshow("Img", image)
        # cv2.waitKey(0)

    else:

        # print("Non-Blurry")
        flag_nonblur = 1
        # cv2.putText(
        # image,
        # "{}: {:.2f}".format(text, laplacian_var),
        # (30, 30),
        # cv2.FONT_HERSHEY_SIMPLEX,
        # 1,
        # (255, 0, 0),
        # 5)
        # cv2.namedWindow("Img", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("Img", 600, 600)
        # cv2.imshow("Img", image)
        # cv2.waitKey(0)

    if flag_nonblur:
        pass

    else:
        if debug == "debug":
            result_dir_split = result_dir.split("/")
            result_dir_split[-1] = "Blur_Images"
            result_dir = '/'.join(result_dir_split)
            # print("result_dir:",result_dir)
            os.makedirs(result_dir, exist_ok=True)
            strr = result_dir + '/' + image_name
            cv2.imwrite(strr, image)

    return flag_nonblur


def adjust_gamma(image, gamma=1.0):

    """

        Description:-
        Build a lookup table mapping the pixel values [0, 255] to
        their adjusted gamma values

        Arguments:
        image:- numpy array
        gamma:- Parameter that can be tuned according to the environment

        Returns a gamma corrected image

    """

    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


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

    return (x, y)


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


def check_pallet_marker(frame):

    """

        Description:-
        It is calculating length of corners and thereby
        checking the status of pallet marker present condition.

        Arguments:
        frame:- Input Image in form of numpy array

        Returns the boolean flag to 1 if there is pallet marker present.

    """

    # Define dictionary
    aruco_dict_pallet = \
        aruco.Dictionary_get(aruco.DICT_6X6_1000)  # Pallet dictionary
    parameters = aruco.DetectorParameters_create()
    parameters.perspectiveRemoveIgnoredMarginPerCell = 0.4
    parameters.maxErroneousBitsInBorderRate = 0.5
    parameters.errorCorrectionRate = 0.75
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
    parameters.cornerRefinementMaxIterations = 30
    parameters.cornerRefinementMinAccuracy = 0.1

    # Get pallet info
    PALLET_WIDTH = 121.92  # In cm
    PALLET_BREADTH = 121.92  # In cm

    # Get aruco info
    PALLET_MARKERLENGTH = 0.07  # 7 cm

    # Flag that will keep track of
    # images when there is no pallet marker detected.
    flag_palletmarker = 0
    image = cv2.GaussianBlur(frame, (5, 5), 3)
    frame = cv2.addWeighted(frame, 1.0, image, -0.5, 7)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cornersP, idsP, rejectedImgPointsP = aruco.detectMarkers(
        gray,
        aruco_dict_pallet,
        parameters=parameters)  # Lists of ids and the corners

    # Get total number of pallet ids detected
    length_pallet = len(cornersP)
    # print('length_pallet',length_pallet)

    # Condition to check if any pallet is present
    # in image or not
    if length_pallet != 0:
        flag_palletmarker += 1
        # print("Error 1 - No pallet detected")
    # print('count_nopalletmarker',count_nopalletmarker)

    # Write error message to the output file
    return (flag_palletmarker)


def check_multiple_pallet_marker(frame):

    """

        Description:-
        It is calculating length of corners and thereby
        checking the status of multiple pallet markers present condition.

        Arguments:
        frame:- Image in form of numpy array

        Returns the boolean flag to 1 if there is no
        multiple pallet marker present.
        We will use this after masking the image because before masking we
        have applied certain conditions for removing multiple pallets.


    """

    # Define dictionary
    aruco_dict_pallet = \
        aruco.Dictionary_get(aruco.DICT_6X6_1000)  # Pallet dictionary

    parameters = aruco.DetectorParameters_create()
    parameters.perspectiveRemoveIgnoredMarginPerCell = 0.4
    parameters.maxErroneousBitsInBorderRate = 0.5
    parameters.errorCorrectionRate = 0.75
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
    parameters.cornerRefinementMaxIterations = 30
    parameters.cornerRefinementMinAccuracy = 0.1

    # Get pallet info
    PALLET_WIDTH = 121.92  # In cm
    PALLET_BREADTH = 121.92  # In cm

    # Get aruco info
    PALLET_MARKERLENGTH = 0.063  # 7 cm
    # Counter that will keep
    # track of images when there is no pallet marker detected.
    flag_nomultiplepalletmarker = 0

    # Smoothing and Sharpening Image
    image = cv2.GaussianBlur(frame, (5, 5), 3)
    frame = cv2.addWeighted(frame, 1.0, image, -0.5, 7)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cornersP, idsP, rejectedImgPointsP = aruco.detectMarkers(
        gray,
        aruco_dict_pallet,
        parameters=parameters)  # Lists of ids and the corners

    # Get total number of pallet ids detected
    length_pallet = len(cornersP)

    # Condition to check if any pallet is present
    # in image or not

    if length_pallet > 1:
        flag_nomultiplepalletmarker = 0
    else:
        flag_nomultiplepalletmarker = 1

    return flag_nomultiplepalletmarker


def check_depth(dimg, debug, result_dir, image_name):

    """

    checks for bad depth images in the input data/frames

    Arguments: color img, depth img (both in numpy)
    returns: boolean whether img is OK to work with(1)
             or if img is not good to work with(0)

    """

    saturation_perc = \
        (np.count_nonzero(dimg == (np.iinfo(dimg.dtype).max)))/(dimg.size)*100
    zero_perc = (np.count_nonzero(dimg == 0))/(dimg.size)*100
    good_depth = 0
    if (zero_perc > 37):
        good_depth = 0

    elif(saturation_perc > 5):  # can update the Parameter on basis of new data
        good_depth = 0

    else:
        good_depth = 1

    if good_depth:
        return good_depth

    else:
        if debug == "debug":
            result_dir_split = result_dir.split("/")
            result_dir_split[-1] = "Bad_Depth"
            result_dir = '/'.join(result_dir_split)
            # print("result_dir:",result_dir)
            os.makedirs(result_dir, exist_ok=True)
            strr = result_dir + '/' + image_name
            cv2.imwrite(strr, dimg)

        return good_depth


def flag_matrix(image, dimg, debug, result_dir, image_name):

    """

        Description:-
        It is calculating a flag matrix for getting
        image information which can be used as a feedback
        for the user.

        Arguments:
        image:- Image in form of numpy array
        dimg:- Depth Image in form of numpy array
        debug:- Debug Mode or Normal Mode
        result_dir:- Incase of debug mode, save the intermediate results

        Returns the maked and gamma corrected image along with
        flag_matrix

    """

    # It should be 1 for a non-blurry image
    flag_nonblur = blur_detection(image, debug, result_dir, image_name)
    # It should be 1 for a good image
    flag_palletmarker = check_pallet_marker(image)
    flag_good_depth = check_depth(dimg, debug, result_dir, image_name)
    # Donot call these functions again, just pass into the mask_image
    masked_gamma_img, idsP, flag_full_palletview = \
        mask_image(image, dimg, flag_nonblur,
                   flag_good_depth, flag_palletmarker)
    # flag_full_palletview
    # will be 1 if there will be full pallet view visible
    dict_flagmatrix = dict({'flag_nonblur': flag_nonblur,
                            'flag_palletmarker': flag_palletmarker,
                            'flag_full_palletview': flag_full_palletview,
                            'flag_good_depth': flag_good_depth})

    values = dict_flagmatrix.values()
    values_sum = sum(values)

    if debug == "debug":
        result_dir_split = result_dir.split("/")
        result_dir_split[-1] = "Masked_images"
        result_dir = '/'.join(result_dir_split)
        # print("result_dir:",result_dir)
        os.makedirs(result_dir, exist_ok=True)
        if (values_sum == 4):
            strr = result_dir + '/' + image_name
            cv2.imwrite(strr, masked_gamma_img)

    if (values_sum == 4):
        return (masked_gamma_img, idsP, dict_flagmatrix)

    return (masked_gamma_img, idsP, dict_flagmatrix)


def mask_image(frame, dimg, flag_nonblur, flag_good_depth, flag_palletmarker):

    """

        Description:-
        It is masking the image by taking only useful
        portion of the particular pallet, in order to
        remove nearby partial pallets by making use
        of pallet dimensions.

        Arguments:
        frame:- Image in form of numpy array
        dimg:- Depth Image in form of numpy array

        Returns the masked image and the image count as
        only after fulfilling these conditions:-
        1.Non-Blurry Image
        2.Pallet Marker Detected
        3.Full Pallet View Visible
        4.Good Depth Image

    """

    # Define dictionary
    aruco_dict_pallet = \
        aruco.Dictionary_get(aruco.DICT_6X6_1000)  # Pallet dictionary

    flag = 0  # Flag that will take care of all error conditions

    parameters = aruco.DetectorParameters_create()
    parameters.perspectiveRemoveIgnoredMarginPerCell = 0.4
    parameters.maxErroneousBitsInBorderRate = 0.5
    parameters.errorCorrectionRate = 0.75
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
    parameters.cornerRefinementMaxIterations = 30
    parameters.cornerRefinementMinAccuracy = 0.1

    # Get pallet info
    PALLET_WIDTH = 121.92  # In cm
    PALLET_BREADTH = 121.92  # In cm
    # Get aruco info
    PALLET_MARKERLENGTH = 0.063  # 7 cm

    # Mask each image - cover unwanted region
    flag_full_palletview = 0  # Turn-on the flag if full pallet view

    # cv2.namedWindow("frame ", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("frame ", 400,400)
    # cv2.imshow("frame ", frame)
    # cv2.waitKey(0)
    image = cv2.GaussianBlur(frame, (5, 5), 3)
    frame = cv2.addWeighted(frame, 1.0, image, -0.5, 7)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    """
        # Dispay gray scale image
        cv2.namedWindow("blackAndWhiteImage", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("blackAndWhiteImage", 600,600)
        cv2.imshow("blackAndWhiteImage", blackAndWhiteImage)
        cv2.waitKey(0)

    """

    cornersP, idsP, rejectedImgPointsP = aruco.detectMarkers(
        gray,
        aruco_dict_pallet,
        parameters=parameters)  # Lists of ids and the corners

    if flag_nonblur == 1:

        if flag_palletmarker == 1:

            if flag_good_depth == 1:

                print("Masking Images...")
                # Get total number of pallet ids detected
                length_pallet = len(cornersP)
                # print("idsP: ", idsP)
                # Mask unwanted region with respet to each
                # detected pallet marker id

                list_marker_height = []
                max_y = 0  # Height of bottom most pallet marker
                cornersP_final = []  # Corner of bottom most pallet marker
                if length_pallet >= 1:
                    for corner_pallet in cornersP:
                        list_marker_height.append(corner_pallet[0][2][1])
                    max_y = max(list_marker_height)

                    for corner_pallet in cornersP:
                        if max_y == corner_pallet[0][2][1]:
                            cornersP_final = corner_pallet
                            break

                cornersP_final = np.array(cornersP_final)

                cornersP_final_used = []
                cornersP_final_used.append(cornersP_final)

                # print("Corners_final: ",cornersP_final_used)
                # print("CornersP: ",cornersP)
                for corner_pallet in cornersP_final_used:
                    # print("corner_pallet: ",corner_pallet)

                    # print("Pallet Id = ", idsP[i][0])

                    # Get it's co-ordinates
                    (x, y) = get_center(corner_pallet)
                    # print("x",x)
                    # print("y",y)

                    """

                        # Just for visualization purpose
                        cv2.putText(
                            frame,
                            str(i),
                            (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            8,
                            (0, 0, 255),
                            15)

                    """

                    # CM to Pixel conversion
                    one_pix_cm = \
                        pix_to_cm(corner_pallet, PALLET_MARKERLENGTH*100)

                    # Calculate image dimension that need to be masked
                    length_pix = \
                        int(PALLET_WIDTH/one_pix_cm)
                    height_Pix = \
                        int(125/one_pix_cm)  # Assuming rack height is 125 cm
                    offset_pix = \
                        int(((PALLET_MARKERLENGTH*100)/2)/one_pix_cm)

                    # print("length_pix:",length_pix)
                    # print("height_Pix:",height_Pix)
                    # print("offset_pix:",offset_pix)
                    # Get top-left co-ordinate

                    left_x1 = x - int(length_pix/2)
                    top_y1 = y - height_Pix

                    # Get bottom-right co-ordinate
                    right_x2 = x + int(length_pix/2)
                    # bottom_y2 = y + 6*offset_pix
                    bottom_y2 = y + 2*offset_pix

                    h, w = frame.shape[:2]
                    # print("w:",w)
                    # print("h:",h)
                    # Check if complete pallet is covered in image or not
                    if ((left_x1 >= 0) and
                            (right_x2 <= w) and
                            (bottom_y2 > int(height_Pix/2 + height_Pix/4))):

                        # If depth is in negative,
                        # consider depth is zero
                        if top_y1 < 0:
                            top_y1 = 0

                        # Get masked image
                        mask = frame.copy()
                        mask[top_y1:bottom_y2, left_x1:right_x2, :] = [0, 0, 0]
                        mask[np.where(mask != [0, 0, 0])] = 255
                        temp_img = cv2.add(mask, frame)

                        # Get cropped image
                        # cropped = frame.copy()
                        cropped = frame[top_y1:bottom_y2, left_x1:right_x2, :]
                        masked_gamma_img = adjust_gamma(temp_img, 2.0)
                        flag_full_palletview = 1
                        # Display masked image
                        # cv2.namedWindow("temp_img ", cv2.WINDOW_NORMAL)
                        # cv2.resizeWindow("temp_img ", 400,400)
                        # cv2.imshow("temp_img ", temp_img)
                        # cv2.waitKey(0)

                    else:
                        # print("No full pallet view for pallet id ",
                        # idsP[i][0])
                        flag_full_palletview = 0

                if flag_full_palletview == 0:
                    # Condition to check if any pallet is
                    # present in image or not
                    print("Error 1 - No full pallet view")
                    print('--------------------------------------------------')
                    masked_gamma_img = None
                    flag = 1

            else:
                # Bad Depth Image present
                print("Depth Image has noise in certain patches!!")
                print('------------------------------------------------------')
                masked_gamma_img = None
                flag = 1

        else:
            # No pallet marker present
            print("Error 2 - No pallet marker detected")
            print('------------------------------------------------------')
            masked_gamma_img = None
            flag = 1

    else:
        print("Image is Blurry")
        print('------------------------------------------------------')
        masked_gamma_img = None
        flag = 1

    if flag == 1:
        return masked_gamma_img, idsP, flag_full_palletview

    else:
        return masked_gamma_img, idsP, flag_full_palletview

        # print("-----------------------------\n")
