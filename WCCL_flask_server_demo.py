import os
from flask import Flask, flash, request, redirect, render_template, url_for
import urllib.request
from werkzeug.utils import secure_filename
import multiprocessing
import time
# import _init_paths
# from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
# from fast_rcnn.test import im_detect
# from fast_rcnn.nms_wrapper import nms
#from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os, sys, cv2
import argparse
import pickle
import json
from flask import send_file
from PIL import Image
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark import layers as L
from maskrcnn_benchmark.utils import cv2_util
from maskrcnn_benchmark.config import cfg
# from server_predict import COCODemo
from inference_engine_box import InferenceEngine


UPLOAD_FOLDER = './box_count/datasets/WCCL/web_demo/upload/'
RESULT_FOLDER = './box_count/datasets/WCCL/web_demo/result/'

ALLOWED_EXTENSIONS = set(['cu', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
INPUT_FILE_FULLNAME = ''
    # Establish communication queues
multiprocessing.set_start_method('spawn',force=True)
tasks = multiprocessing.JoinableQueue()
results = multiprocessing.Queue()
app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    #parser.add_argument(
    #    '--gpu', 
    #    dest='gpu_id', 
    #    help='GPU device id to use [0]',
    #    default=0, type=int
    #)
    #parser.add_argument(
    #    '--cpu', dest='cpu_mode',
    #    help='Use CPU mode (overrides --gpu)',
    #    action='store_true'
    #)
    parser.add_argument(
        '--port', 
        dest='port',
        help = '',
        default=5000,
        type=int
    )
    parser.add_argument(
        "--config",
        default="box_count/configs/e2e_r2cnn_R_50_FPN_1x_webdemo.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.1,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=864,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    #cfg.freeze()
    print(cfg)

    return args

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/detect')
def upload_form():
    return render_template('upload.html')

@app.route('/detect', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # print(file.filename)
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            im = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            INPUT_FILE_FULLNAME = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            # print(type(im))
            if im is None or not im.shape[0]>0 or not im.shape[1]>0 or not im.shape[2]>0:
                return "bad image"
            tasks.put(Task(im,time.time()))
            tasks.join()
            next_result = results.get()
            im2,quad_boxes,label = next_result()
            #im2 = next_result()
            #print("Type of Quadbox: ",type(quad_boxes))
            #print("Type of label: ",type(label))
            f = RESULT_FOLDER + filename
            
            cv2.imwrite(f, im2)
            #print(im2.shape)
            img = im2.tolist()
            #print(len(img))
            #img_2 = np.array(img)
            #print(img_2.shape)
            #return send_file(f, mimetype='image/jpg')
            return json.dumps(list([img,quad_boxes.tolist(),label]))

    return 

class Consumer(multiprocessing.Process):
    
    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.inited = False

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        cfg = self.cfg

        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        transform = T.Compose(
            [
                T.ToPILImage(),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def run(self):
        proc_name = self.name
        if not self.inited:	    
            cfg.TEST.HAS_RPN = True  # Use RPN for proposals
            cfg.TEST.BBOX_REG = True

            self.args = parse_args()
            # Code for instatiating model and loading the pretrained weigths
            # ....
            self.cfg = cfg.clone()

            # prepare object that handles inference plus adds predictions on top of image
            self.WCCL_demo = InferenceEngine(
                self.cfg,
                confidence_threshold=self.args.confidence_threshold,
                min_image_size=self.args.min_image_size,
            )
            print("Loaded the model")
            self.inited = True

        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                print('%s: Exiting' % proc_name)
                self.task_queue.task_done()
                break
            print('%s' % (proc_name))
            answer,name = next_task()
            self.task_queue.task_done()
            scale_factor = 1 

            ## Code copied from webcam.py from the same repository
            start_time = time.time()
            f = INPUT_FILE_FULLNAME
            out_name = f.split('/')[-1]
            # print(out_name)
            composite = self.WCCL_demo.run_on_opencv_image(answer)
            #print("Composite ####: ",composite)
            print("Time: {:.2f} s / img".format(time.time() - start_time))
            #cv2.imshow("COCO detections", composite)

            self.result_queue.put(Result(composite))
            
        return


class Task(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def __call__(self):
        return self.a,self.b
    def __str__(self):
        return '%s * %s' % (self.a, self.b)

class Result(object):
    def __init__(self, a): #, b, c, d):
        self.a = a
        # self.b = b
        # self.c = c
        # self.d = d
    def __call__(self):
        return self.a #,self.b,self.c,self.d
    def __str__(self):
        return '%s' % (self.a,)


if __name__ == '__main__':
    # Start consumers
    args = parse_args()
    num_consumers = 1#multiprocessing.cpu_count() * 2
    print('Creating %d consumers' % num_consumers)
    consumers = [ Consumer(tasks, results)
                  for i in range(num_consumers) ]
    for w in consumers:
        w.start()

    # Wait for all of the tasks to finish
    print("started consumer")
    app.run(
        host='127.0.0.1', 
        port=args.port,
        debug=False,
        use_reloader=False
    )
    print("started app")
    tasks.join()
    
    
