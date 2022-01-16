import numpy as np
import cv2

# color: [ 1920x1080 p[950.995 564.756] f[1367.27 1365.57] Inverse Brown Conrady [0 0 0 0 0] ]

# color: [ 1920x1080 p[947.880 548.147] f[958.069 958.069] Inverse Brown Conrady [0 0 0 0 0] ]

# cameraMatrix = [[878.33202015, 0, 485.74167328],
#                                                 [0, 878.44704215, 323.28120842],
#                                                 [0, 0, 1]]
# cameraMatrix = np.array(cameraMatrix)
# print(np.linalg.inv(cameraMatrix).shape)                                               

# corners = [610, 163, 858, 163, 858, 267, 610, 267]

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
    
    buf = 2
    patch = dimg[y-buf:y+buf, x-buf:x+buf]
    # print(patch)
    # img[y-buf:y+buf, x-buf:x+buf] = 128
    # img[x-buf:x+buf+50,y-buf:y+buf+50] = 255
    sum_patch = np.sum(patch)
    elements_patch = np.count_nonzero(patch)
    avg_depth_patch = sum_patch/elements_patch
    print('Read depth value-----',avg_depth_patch)
    z_cm = np.around((avg_depth_patch/10), decimals=2)
    return(z_cm)

def calculate_XYZ(u,v,d):
        cameraMatrix_1 = [[1367.27, 0, 950.995],[0, 1365.57, 564.756],[0, 0, 1]]
        cameraMatrix_2 = [[958.069, 0, 947.880],[0, 958.069, 548.147],[0, 0, 1]]
        #tvec1
        cameraMatrix = np.array(cameraMatrix_1)
        # print(np.linalg.inv(cameraMatrix))

        uv_1=np.array([[u,v,1]], dtype=np.float32)
        uv_1=uv_1.T
        suv_1=d*10*uv_1
        xyz_c=np.linalg.inv(cameraMatrix).dot(suv_1)

        # xyz_c=xyz_c-self.tvec1
        # XYZ=self.inverse_R_mtx.dot(xyz_c)

        return xyz_c

# rotation: [0.999798, 0.017687, -0.00953747, -0.0176787, 0.999843, 0.000951276, 0.0095528, -0.000782473, 0.999954]
# translation: [0.0148032, 0.000345676, -0.000438305]


# u1 = 264.0
# v1 = 258.0
# xyz1 = calculate_XYZ(u1,v1)

# u2 = 395.0
# v2 = 283.0
# xyz2 = calculate_XYZ(u2,v2)

# print(xyz1,xyz2)

# squared_dist = np.sum((xyz1-xyz2)**2, axis=0)
# dist = np.sqrt(squared_dist)
# print('distance in cm------>',(dist/10).astype(np.float16))

# from scipy.spatial import distance as dist

# def pix_to_cm(box_all_cordinates, marker_length):
#     pts = np.zeros((4, 2), dtype="float32")
#     pts[0, 0] = box_all_cordinates[0]
#     pts[0, 1] = box_all_cordinates[1]
#     pts[1, 0] = box_all_cordinates[2]
#     pts[1, 1] = box_all_cordinates[3]
#     pts[2, 0] = box_all_cordinates[4]
#     pts[2, 1] = box_all_cordinates[5]
#     pts[3, 0] = box_all_cordinates[6]
#     pts[3, 1] = box_all_cordinates[7]
#     # pts[1, :] = box_all_cordinates[0][1]
#     # pts[2, :] = box_all_cordinates[0][2]
#     # pts[3, :] = box_all_cordinates[0][3]
#     # rect = order_points_old(pts)
#     corner1 = pts[0, :]
#     corner2 = pts[1, :]
#     e_distance = dist.euclidean(corner1, corner2)
#     one_pix_cm = marker_length / e_distance
 
#     return one_pix_cm


# one_pix_cm = pix_to_cm(corners, 10*100)
# # print(one_pix_cm)



# (x=264,y=258  4300) (395,283, 4681)
