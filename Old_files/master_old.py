import os
import cv2
import time
import numpy as np
# import cv2.aruco as aruco
# from scipy.spatial import distance as dist
import glob
import argparse
import preprocessing_script
import box_count
import inference_engine_box
import box_tvecsS_palletmarker
import pandas as pd
# import sys
# Data-Capture->Pre-processing->Box-DetectionDL->Box_tvecsS->Box_count


def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-d', '--debug', help='Description for Debug argument',
                        default='Normal')
    parser.add_argument('-m', '--realtime', help='realtime or directory/batch \
                        wise operation', default='batch')
    args = vars(parser.parse_args())
    debug = args['debug']
    realtime = args['realtime']

    input_dir = '/home/nishant/Wipro/work/Dataset/data40'
    input_image_list = []
    input_depthimage_list = []
    # Get list of all images from specified location
    count = 0  # Counter that will take care of number of frames
    # processed condition, i.e it will stop the video
    # processing part once we get the 6 frames
    flag_nomultiplepalletmarker = 0  # Flag that will keep track of images
    # when there are multiple pallet markers detected.

    print("Pre-Processing Started")
    img_wise_pallet = []
    fields = list(['Image_Name', 'Pallet_ID', 'Remark'])
    img_wise_pallet.append(fields)
    images = glob.glob(input_dir + "/*.jpg")
    depth_images = glob.glob(input_dir + "/*.png")
    images.sort()
    depth_images.sort()
    for img in images:
        input_image_list.append(img)

    for depth_img in depth_images:
        input_depthimage_list.append(depth_img)

    good_imglist = []
    depth_img_list = []
    img_name_list = []
    L_flagmatrix = []
    # Get current time stamp
    time_stamp = str(time.time())

    input_img_dir = input_dir
    # directory1 is an intermediate variable for getting outer_folder_name
    # i.e Mission Folder
    directory1 = input_img_dir
    directory1 = directory1.split('/')
    directory1.pop()
    directory1.pop()
    directory1.append('output')
    directory1 = '/'.join(directory1)
    outer_folder_name = (directory1 + "/" + "Mission_" + time_stamp)
    os.mkdir(outer_folder_name)
    input_dir = outer_folder_name
    # print(input_dir)
    input_dir_split = input_dir.split("/")
    input_dir_split.append("CountResult")
    result_dir = '/'.join(input_dir_split)
    # print("result_dir:",result_dir)
    os.mkdir(result_dir)

    count_all_images = 0

    for image_name in input_image_list:
        # Get image name alone
        name_break = image_name.split('/')
        img_name = name_break[-1]
        # name = img_name[0:len(img_name)-4]
        print("Image Name = ", img_name)
        depthImage = cv2.imread(input_depthimage_list[count_all_images], -1)

        # Read image
        frame = cv2.imread(image_name)
        # frame = cv2.blur(frame, (5, 5))
        print(image_name)
        masked_gamma_img, idsP, dict_flagmatrix = \
            preprocessing_script.flag_matrix(frame,
                                             depthImage, debug,
                                             result_dir, img_name)
        # print("dict_flagmatrix: ",dict_flagmatrix)
        L_flagmatrix.append(dict_flagmatrix)
        values = dict_flagmatrix.values()
        values_sum = sum(values)
        count_all_images += 1
        """
        if values_sum == 4:
            flag_nomultiplepalletmarker = \
            preprocessing_script.check_multiple_pallet_marker(masked_gamma_img)

        if flag_nomultiplepalletmarker == 0:
            print("Error 3 - Multiple pallet marker detected")
            masked_gamma_img = None
        print('------------------------------------------------------')
        """
        # dict_flagmatrix['flag_nomultiplepalletmarker'] = \
        #                                           flag_nomultiplepalletmarker
        # print("dict_flagmatrix: ",dict_flagmatrix)
        # print(adjusted)
        # print(type(adjusted))
        # print(adjusted.shape)
        # print('flag_good_img',flag_good_img)
        # print('Image:',image)
        # img_path = input_dir + '/Output/' + image_name.split('/')[-1]

        # print(adjusted)
        # cv2.imshow('Image',adjusted)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        if (count_all_images % 8 == 0):
            df_flag_8 = \
                pd.DataFrame(L_flagmatrix[count_all_images-8:count_all_images])
            # print('df_flag_8: ',df_flag_8)
            for i in range(count_all_images-8, count_all_images):  # step of +8
                for index in range(df_flag_8.shape[1]):
                    # print('Column Number : ', index)
                    # Select column by index position using iloc[]
                    columnSeries = df_flag_8.iloc[i:i+8, index]
                    # print('Column Contents : ', columnSeries.values)
                    # check if we hve 8 zeros
                    if(columnSeries[columnSeries == 0].count() == 8):
                        # if (columnSeries.sum() == 0):
                        if(index == 0):
                            print('blur problem')
                        elif (index == 1):
                            print('pallet maker missing')
                            if index == 2:
                                print('no full pallet view issue due to marker\
                                       absent condition')
                        elif(index == 2):
                            print('no full pallet view issue')
                        elif(index == 3):
                            print('Depth issue')
                        # print(df_flagmatrix)
        print()

        if values_sum == 4:
            # cv2.imwrite(img_path, masked_gamma_img)
            masked_gamma_numpy = np.array(masked_gamma_img)
            img_name_list.append(image_name)
            depth_img_name = image_name
            depth_img_name = depth_img_name.split(".")
            depth_img_name[-1] = "png"
            depth_img_name[-2] = "depth"
            depth_img_name = '.'.join(depth_img_name)
            # print("depth_img_name: ",depth_img_name)
            depth_img_list.append(depth_img_name)
            good_imglist.append(masked_gamma_numpy)
            row = [img_name, str(idsP[0][0])]
            img_wise_pallet.append(row)
            # cv2.imshow('Adjusted_numpy',adjusted_numpy)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        else:
            continue

        count = len(good_imglist)

        # print(count)
        print("Completed")

        if count >= 6:
            print('6 frames processed!!!!')
            break

        print('------------------------------------------------------')

    df_flagmatrix = pd.DataFrame(L_flagmatrix)
    print(df_flagmatrix)

    # print("good_imglist: ",good_imglist)
    # print("depth_img_list: ",depth_img_list)
    # print("img_name_list: ",img_name_list)

    print("Pre-Processing Completed")
    print()

    print("Box Detection Model Started...\n")
    counter = 0
    corners4 = []
    sku_ref_pts_tvecs = []
    cornersS_final = []
    L_box_ref_moved = []
    sku_ref_pts_list = []
    cornersS_final_list = []
    L_box_ref_moved_list = []  # Updated box ref points

    directory_boxdetection = result_dir
    directory_boxdetection = directory_boxdetection.split("/")
    directory_boxdetection[-1] = "boxdetection"
    directory_boxdetection = '/'.join(directory_boxdetection)
    file_box_corners = ''
    file_box_centre = ''
    file_box_corners_true = ''
    file_box_centre_true = ''
    # Remove the second condition
    if debug == 'debug':
        os.makedirs(directory_boxdetection, exist_ok=True)
        file_box_corners = open(directory_boxdetection +
                                "/Corner_Box_Coordinates.txt", "a")
        file_box_centre = open(directory_boxdetection +
                               "/Centre_Box_Coordinates.txt", "a")
        file_box_corners_true = open(directory_boxdetection +
                                     "/Corner_Box_Coordinates_true.txt", "a")
        file_box_centre_true = open(directory_boxdetection +
                                    "/Centre_Box_Coordinates_true.txt", "a")

    img_count = 0
    for img in good_imglist:

        depthImage = cv2.imread(depth_img_list[img_count], -1)
        # print("depthImage: ",depthImage)
        img_name = img_name_list[img_count]
        print("Image name: ", img_name)
        corners4, counter =  \
            inference_engine_box.Box_Coordinates(img, img_name, depthImage,
                                                 file_box_corners,
                                                 file_box_centre,
                                                 directory_boxdetection,
                                                 img_count, debug)

        # print("corners4: ",corners4)
        # print("counter: ",counter)
        sku_ref_pts_tvecs, cornersS_final, L_box_ref_moved = \
            box_tvecsS_palletmarker.calculate_tvecsS(img,
                                                     img_name, depthImage,
                                                     directory_boxdetection,
                                                     file_box_corners_true,
                                                     file_box_centre_true,
                                                     img_count, corners4,
                                                     counter, debug)

        sku_ref_pts_list.append(sku_ref_pts_tvecs)
        cornersS_final_list.append(cornersS_final)
        L_box_ref_moved_list.append(L_box_ref_moved)
        img_count = img_count + 1
        # print("img_count: ", img_count)

    # print("sku_ref_pts_list: ",sku_ref_pts_list)
    # print("cornersS_final_list: ",cornersS_final_list)
    # print("L_box_ref_moved_list: ",L_box_ref_moved_list)

    print("Box Detection Model Completed...")
    print()
    print("Updated Box_Ref_pts calculated!!! ")
    print()
    print("Box Count Module Started...\n")

    fields = []
    rows = []
    fields = img_wise_pallet[0]
    # print("img_wise_pallet: ",img_wise_pallet)
    # extracting each data row one by one
    for row in img_wise_pallet[1:]:
        rows.append(row)
    # file11 = open("image_list.txt","w")
    L = []
    st = " "
    for i in rows:
        st = st.join(i)
        L.append(st + " \n")
        st = " "
    # print("rows: ",rows)
    # Aggregate images pallet wise
    index = 0
    for val in rows[:]:

        if int(val[1]) == 0:
            # Discard images for which no
            # pallet information is found
            rows.pop(index)
            index = index - 1

        index = index + 1

    pallet_list = [x[1] for x in rows]
    pallet_list = np.array(pallet_list)
    # print("pallet_list: ",pallet_list)
    pallet_list_unique, pallet_list_indices = np.unique(pallet_list,
                                                        return_index=True)

    name_list = [x[0] for x in rows]
    name_list = np.array(name_list)

    dict_pallet_wise_list = {}  # Dictionary to store image names pallet wise
    # print("pallet_list_unique: ",pallet_list_unique)
    for no in pallet_list_unique:
        # Group images pallet wise
        pts = np.where((pallet_list == no))
        dict_pallet_wise_list[no] = name_list[pts]

    # print("dict_pallet_wise_list: ",dict_pallet_wise_list)

    for pallet_id in dict_pallet_wise_list:
        # Run count code pallet wise

        box_count.get_count_pallet_wise(pallet_id, dict_pallet_wise_list,
                                        result_dir, sku_ref_pts_list,
                                        cornersS_final_list,
                                        L_box_ref_moved_list,
                                        good_imglist, depth_img_list,
                                        img_name_list, debug)
        print("-----------------------------")

    print("Box Count Module Completed...\n")

    print("\nDone\n")


main()