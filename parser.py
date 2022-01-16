import os
import time
import requests
import numpy as np
import pickle
import glob
import json
import torch
 
 
def vision(img_path):
	files = {'file': open(img_path,'rb')}
	r = requests.post('http://127.0.0.1:5000/detect', files=files)
	[img_name,box_coordinates, labels]=json.loads(r.content)
	#print(len(box_coordinates))
	return img_name,box_coordinates, labels
