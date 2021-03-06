import cv2
import torch
import time
# from torch.utils.tensorboard import SummaryWriter
import numpy as np
import warnings
from maskrcnn_benchmark.config import cfg
import os
# default `log_dir` is "runs" - we'll be more specific here
# writer = SummaryWriter('runs/fashion_mnist_experiment_1')
from torchvision import transforms as T
import torch.nn.functional as F
from PIL import Image
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark import layers as L
from maskrcnn_benchmark.utils import cv2_util
import torchvision.transforms as torch_transform
warnings.filterwarnings("ignore")


class InferenceEngine(object):
    # COCO categories for pretty print
    CATEGORIES = ["__background__",  "Carton Box", "Aruco Marker", ]
    # CATEGORIES = [
    #    "__background__",
    #    "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
    # "basketball-court", "ground-track-field", "harbor", "bridge",
    # "large-vehicle", "small-vehicle", "helicopter", "roundabout",
    # "soccer-ball-field", "swimming-pool", "container-crane",]

    def __init__(
        self,
        cfg,
        weights,
        confidence_threshold=0.5,
        min_image_size=864,
    ):
        self.cfg = cfg.clone()
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        self.min_image_size = min_image_size

        save_dir = cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(cfg, self.model,
                                             save_dir=save_dir)
        _ = checkpointer.load(weights)

        self.transforms = self.build_transform()

        # used to make colors for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )

        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(self.min_image_size),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def run_on_opencv_image(self, image, i, img_name):
        """
        Arguments:
            image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        predictions, smap, sal_map, feature = self.compute_prediction(image, i,
                                                                      img_name)
        # ind1=torch.where(sal_map[:,1,:,:] <0)
        # ind2=torch.where(sal_map[:,1,:,:] >=0)
        # print(ind2)
        # print(ind2[0][0].item(),ind2[1][0].item(),ind2[2][0].item())
        top_predictions = self.select_top_predictions(predictions)
        # print(top_predictions.quad_bbox)
        result = image.copy()
        result, quad_boxes = self.overlay_boxes(result, top_predictions)
        result, labels = self.overlay_class_names(result, top_predictions)
        dim = (result.shape[1], result.shape[0])

        # sal_map1=F.softmax(sal_map).cpu()
        trans = torch_transform.ToPILImage()
        # sal_map1=trans(sal_map1[:,1,:,:].cpu())
        # # print(sal_map1.size)
        # sal_map1.save("/home/rl/frameworks/R2CNN.pytorch/datasets/ICDAR2015/results_pva_update/sal_map_1/"+img_name,"JPEG")

        # print((feature).shape)
        ind11 = np.random.permutation(768)
        return result, quad_boxes, labels

    def compute_prediction(self, original_image, i, img_name):
        """
        Arguments:
            original_image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        # apply pre-processing to image
        image = self.transforms(original_image)
        image = image.to(self.device)
        # convert to an ImageList, padded so that it is divisible by
        # cfg.DATALOADER.SIZE_DIVISIBILITY
        image_list = to_image_list(image,
                                   self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # writer.add_graph(self.model, image_list.tensors)
        # compute predictions
        with torch.no_grad():
            predictions, sal_map, feature = self.model(image_list.tensors)
            # predictions= self.model(image_list.tensors, writer)

        # smap=torch.argmax(sal_map,dim=1).cpu()
        # smap=smap.detach().numpy()
        # smap = smap.transpose(1,2,0)
        # smap=((smap*200)+55)
        # smap = smap.astype(np.uint8)

        predictions = [o.to(self.cpu_device) for o in predictions]
        # always single image is passed at a time

        prediction = predictions[0]
        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))
        # print(prediction.quad_bbox)

        # return prediction,smap,sal_map, feature
        return prediction, None, None, None

    def select_top_predictions(self, predictions):
        """
        Select only predictions which have a `score`>self.confidence_threshold
        ,and returns the predictions in descending order of score

        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        scores = predictions.get_field("scores")
        keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        # print(scores)
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        colors = labels[:, None] * self.palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors

    def overlay_boxes(self, image, predictions):
        """
        Adds the predicted boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        labels = predictions.get_field("labels")
        boxes = predictions.bbox
        # print(boxes)
        quad_boxes = predictions.quad_bbox

        # print(image.shape)

        colors = self.compute_colors_for_labels(labels).tolist()
        # colors=(128,128,255)

        for quad_box, box, color in zip(quad_boxes, boxes, colors):
            # box = box.to(torch.int64)
            quad_box = quad_box.to(torch.int64)
            # print(quad_box)
            # print(box)
            # top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            # image = cv2.rectangle(
            # image, tuple(top_left), tuple(bottom_right), tuple((0,0,255)), 3
            # )
            cv2.line(image, (quad_box[0], quad_box[1]),
                            (quad_box[2], quad_box[3]), color, 2)
            cv2.line(image, (quad_box[2], quad_box[3]),
                            (quad_box[4], quad_box[5]), color, 2)
            cv2.line(image, (quad_box[4], quad_box[5]),
                            (quad_box[6], quad_box[7]), color, 2)
            cv2.line(image, (quad_box[6], quad_box[7]),
                            (quad_box[0], quad_box[1]), color, 2)
            cv2.putText(image, "1st", (quad_box[0], quad_box[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "2nd", (quad_box[2], quad_box[3]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "3rd", (quad_box[4], quad_box[5]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "4th", (quad_box[6], quad_box[7]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return image, quad_boxes

    def overlay_boxes_rpn(self, image, predictions):
        """
        Adds the predicted boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        # labels = predictions.get_field("labels")
        boxes = predictions.bbox

        # colors = self.compute_colors_for_labels(labels).tolist()

        for box in zip(boxes):
            # print(box[0])
            box = box[0].to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv2.rectangle(image, tuple(top_left), tuple(bottom_right),
                                  tuple((0, 255, 0)), 1)

        return image

    def overlay_class_names(self, image, predictions):
        """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
        scores = predictions.get_field("scores").tolist()
        # print(scores)
        labels = predictions.get_field("labels").tolist()
        # print(labels)
        labels = [self.CATEGORIES[i] for i in labels]
        # print(labels)
        boxes = predictions.bbox

        template = "{}: {:.2f}"
        for box, score, label in zip(boxes, scores, labels):
            x, y = box[:2]
            s = template.format(label, score)
            cv2.putText(
                image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 1)

        return image, labels


def get_center(corner):
    """ Returns center of provided rectangle (x,y)
        Takes bounding box co-ordinates(numpy array of shape (4,2))
    """
    x1 = corner[0][0]
    y1 = corner[0][1]

    x2 = corner[1][0]
    y2 = corner[1][1]

    x3 = corner[2][0]
    y3 = corner[2][1]

    x4 = corner[3][0]
    y4 = corner[3][1]
    # Taking the average of the corners in order to get the centre coordinates
    x = int((x1+x2+x3+x4)/4)
    y = int((y1+y2+y3+y4)/4)

    return (x, y)


def get_box_depth(dimg, x, y):
    '''
    calculate the depth for a particular bounding boxes centre patch
    @arguments- depth image, x_center, y_center
    returns the z_value or depth for a particular box
    '''

    patch = dimg[y-6:y+6, x-6:x+6]
    # take a small patch of 12*12 around the centre pixel of the box
    sum_patch = np.sum(patch)
    elements_patch = np.count_nonzero(patch)
    avg_depth_patch = sum_patch/elements_patch
    z_cm = np.around((avg_depth_patch/10), decimals=2)
    return(z_cm)  # returns in cm upto 2 decimal places


begin = time.time()

config_file = \
    'box_count/configs/e2e_r2cnn_R_50_FPN_1x_webdemo.yaml'
weights = 'box_count/output/WCCL3/model_0240000.pth'
cfg.merge_from_file(config_file)

detector = InferenceEngine(cfg, weights)


def Box_Coordinates(inputImage, img_name, depthImage, file_box_corners,
                    file_box_centre, out_dir,
                    i, debug):
    """
        Arguments:
            inputImage (numpy_array): Real_sense Image captured from camera
            img_name (String): Image name
            depthImage(numpy_array):- Depth Image
            file_box_corners:- Text file for storing the corner coordinates and
                number of visible boxes
            file_box_centre:- Text file for storing the centre coordinates
                for further computation
            out_dir:- Output Image Directory
            i:- Denotes a counter of number of images passed

        Returns the corner-coordinates and number of visible boxes
            present on the rack + pushes corner-coordinates,centre-coordinates,
            depth value of the bounding box to the text file.
    """
    try:
        # Floor Division : Gives only Fractional Part as Answer
        strt = time.time()
        L1 = []  # List that will store the corners of boxes
        BoxPoseList = []
        # Temporary list that will store box centre coordinates and depth value
        counter = 0  # Count the number of visible boxes in an image
        x_center = 0
        y_center = 0
        Z = 0  # Depth of a box

        L2 = []

        # print(depth_img_name)
        # List that will store the centre coordinates of a box and depth value
        flag = 0
        # FLag that will be turned on if any bad depth image will be found
        img_count = 0  # Counter that will count bad depth images
        strt = time.time()
        # outer_folder_name = blurdetection.outer_folder_name
        # folder_name = outer_folder_name + "/" + "Bad_Depth"
        # os.makedirs(folder_name, exist_ok=True)

        # print(img.shape)
        # print(img.shape)
        # writer.add_image('icdar_input_image', img)
        canvas, quad_boxes, labels = detector.run_on_opencv_image(inputImage,
                                                                  i, img_name)
        # print("Type of quad_boxes:- ", type(quad_boxes))
        # print("Shape of Quad_box:- ", quad_boxes.shape)
        if debug == "debug":
            file_box_centre.write("Centre-Coordinates of boxes in:" + img_name + ":")
            file_box_centre.write("\n")

        # Define a function for returning the coordinates
        for quad_box, label in zip(quad_boxes, labels):
            # print("Type of quad_box:- ", type(quad_box))
            # print("Shape of Quad_box:- ", quad_box.shape)
            if(label == 'Carton Box'):
                a = quad_box.tolist()
                b = np.array(a)
                box = b.reshape(4, 2)
                # print(b)
                # print(type(b))
                (x_center, y_center) = get_center(box)
                Z = get_box_depth(depthImage, x_center, y_center)
                BoxPoseList.append(x_center)
                BoxPoseList.append(y_center)
                BoxPoseList.append(Z)
                if debug == "debug":
                    file_box_centre.write("Centre-Coordinates and Depth of box" + str(counter + 1) + ":")
                    file_box_centre.write("\n")
                    file_box_centre.write(str(x_center) + "," + str(y_center) + "," + str(Z))
                    file_box_centre.write("\n")

                x_center = 0
                y_center = 0
                if np.isnan(Z):
                    flag = 1
                    break
                counter += 1
                L1.append(a)
                L2.append(BoxPoseList)
                BoxPoseList = []
        if debug == "debug":
            file_box_centre.write(str(L2))
            file_box_centre.write("\n")
            file_box_centre.write("-----------------------------------")
            file_box_centre.write("\n")
            file_box_centre.write("-----------------------------------")
            file_box_centre.write("\n")

        if (flag == 0):
            op = np.zeros((counter, 1, 3), dtype=np.float32)
            op = np.array(L2)
            op = op.reshape((counter, 1, 3))
            corners4 = []
            # Final List that will hold all the corner coordinates of a box
            corners3 = L1  # Temporary List
            for corner in corners3:
                corner = np.array(corner)
                corner = corner.reshape((1, 4, 2))
                corners4.append(corner)
            L2 = []
            if debug == "debug":
                file_box_corners.write("Total number of boxes visible are:-" + str(counter))

                file_box_corners.write("\n")
                file_box_corners.write(str(L1))
                file_box_corners.write("\n")
                file_box_corners.write("-----------------------------------------")
                file_box_corners.write("\n")
                # print("time taken for detection:", end-strt)
                # print(out_path)
                img_name = img_name.split('/')
                img_name = img_name[-1]
                out_path = out_dir + '/' + img_name
                # print(out_path)
                cv2.imwrite(out_path, canvas)
                # writer.close()

                L1 = []
            L1 = []
            end = time.time()
            return (corners4,   counter)
        else:
            corners4 = []
            counter = 0
            # bad_depth_colored = folder_name + '/' + img_name + '.jpg'
            # bad_depth = folder_name+ '/' + img_name+'.png'
            # Make the folder here itself and push the image
            # We can move it from the input directory also
            # cv2.imwrite(bad_depth_colored, inputImage)
            # cv2.imwrite(bad_depth,depthImage)
            return (corners4, counter)

    except (TypeError, ZeroDivisionError) as e:
        # bad_depth_colored = img_name + '.jpg'
        # print("Sorry ! This Depth image has noise,\
        #        "edges are not clearly visible... ")
        # cv2.imwrite(bad_depth_colored, inputImage)
        if debug == 'debug':
            image_name = img_name
            image_name = image_name.split('/')
            image_name = image_name[-1]
            depth_img_name = image_name.split('.')
            depth_img_name[-1] = 'png'
            depth_img_name = '.'.join(depth_img_name)
            bad_depth_path = out_dir.split('/')
            bad_depth_path[-1] = 'Bad_depth'
            bad_depth_path = '/'.join(bad_depth_path)
            depth_img_name = bad_depth_path + '/' + depth_img_name
            os.makedirs(depth_img_name, exist_ok=True)
            cv2.imwrite(depth_img_name, depthImage)
        corners4 = []
        counter = 0
        return (corners4, counter)
