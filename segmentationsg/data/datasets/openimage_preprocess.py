import json
import math
from pathlib import Path
from math import floor
from PIL import Image, ImageDraw
import random
import os
import torch
import numpy as np
import pickle
import yaml
import pandas as pd
import cv2
from tqdm import tqdm
from pycocotools import mask
from detectron2.config import get_cfg
from detectron2.structures import Instances, Boxes, pairwise_iou, BoxMode


splits = ['train', 'test']
# splits = ['validation']
data_dir= Path('/scratch/hdd001/datasets/openimages')
dev_run = False

orig_id2cat_id = json.load(open('/h/suhail/SceneGraph/data/datasets/openimages/orig_id2cat_id.json', 'r'))

for split in splits:

    #Directory containing the annotation files
    annotation_dir = data_dir / 'annotations'
    #Directory contating the segementation masks
    segmentation_dir = annotation_dir / 'segmentations'
    segmentation_annotation_file = segmentation_dir / split / '{}-annotations-object-segmentation.csv'.format(split)

    #Sort and group the data frame by imageid 
    segmentation_annotion = pd.read_csv(segmentation_annotation_file).sort_values('ImageID')
    grouped_segmentation_annotation = segmentation_annotion.groupby('ImageID')

    dataset_dicts = []
    # i = 0
    for imageid, group in tqdm(grouped_segmentation_annotation):
        
        image_dict = {}

        image_dict['image_id'] = imageid
        image_dict['file_name'] = str(data_dir / split / '{}.jpg'.format(imageid) )

        image = cv2.imread(image_dict['file_name'])
        image_dict['height'], image_dict['width'] = image.shape[0], image.shape[1] 

        objs = []
        for index, rows in group.iterrows():
            obj = {}

            obj["bbox"] = [rows['BoxXMin']*image_dict['width'], rows['BoxYMin']*image_dict['height'], 
                           rows['BoxXMax']*image_dict['width'], rows['BoxYMax']*image_dict['height']]

            obj["bbox_mode"] = BoxMode.XYXY_ABS

            #Reshape mask as the images and segmentation mask have different size
            maskimage = cv2.resize(cv2.imread(str(segmentation_dir / split / '{}-masks-{}'.format(split, imageid[0]) / rows['MaskPath'] )), (image_dict['width'], image_dict['height']))

            #Conver the numpy array to contours
            imgray = cv2.cvtColor(maskimage, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(imgray, 127, 255, 0)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contour_lengths = [x.shape[0] for x in contours]
            
            #Some masks are corrupted and donot have any annotions(Skip these masks)
            if len(contour_lengths) == 0:
                continue

            #In case of multiple contours extracted choose the largest one
            choose_contour = contour_lengths.index(max(contour_lengths)) 
            poly = contours[choose_contour].squeeze().tolist()
            if isinstance(poly[0], int):
                continue
            poly = [p for x in poly for p in x]

            obj["segmentation"] = [poly]
            obj["category_id"] = orig_id2cat_id[rows['LabelName']] - 1
        
            objs.append(obj)
        
        image_dict["annotations"] = objs

        dataset_dicts.append(image_dict)

    print("Svaing the dict for {}".format(split))
    with open('/h/suhail/SceneGraph/data/datasets/openimages/{}-imagedict.pkl'.format(split), 'wb+') as f:
        pickle.dump(dataset_dicts, f)
