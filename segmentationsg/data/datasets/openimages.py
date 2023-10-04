import h5py
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
import cv2
from detectron2.config import get_cfg
from detectron2.structures import Instances, Boxes, pairwise_iou, BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

class OpenImageTrainData():
    '''
    Registed Open Images Dataset in Detectron
    '''
    def __init__(self, cfg, data_dir='/scratch/hdd001/datasets/openimages', split='train'):

        self.cfg = cfg
        valid_splits = ['train', 'validation', 'test']
        assert split in valid_splits, "Invalid split {}. Specify one of {}".format(split, valid_splits)
        self.split = split
        self.data_dir = Path(data_dir)
        self.dataset_dicts = self.get_dataset_dicts()

        self.register_dataset()

    def register_dataset(self):

        DatasetCatalog.register("openimages_" + self.split, lambda: self.dataset_dicts)
        #MetaData for Open Images
    
    def get_dataset_dicts(self):
        '''
        Convert dataset to Detectron format
        '''

        annotation_dir = data_dir / 'annotations'
        segmentation_dir = annotation_dir / 'segmentations'
        # coco_style_annotation_file = annotation_dir / 'coco_style' / '{}-annotation-bbox.json'
        # cs_annotation = json.load(open(coco_style_annotation_file,'r'))

        segmentation_annotation_file = segmentation_dir / split / '{}-annotations-object-segmentation.csv'.format(split)

        segmentation_annotion = pd.read_csv(segmentation_annotation_file).sort_values('ImageID')

        grouped_segmentation_annotation = segmentation_annotion.groupby('ImageID')

        dataset_dicts = []
        for idx, (imageid, group) in tqdm(enumerate(grouped_segmentation_annotation)):
            
            image_dict = {}

            image_dict['image_id'] = imageid
            image_dict['file_name'] = str(data_dir / split / '{}.jpg'.format(imageid) )

            image = cv2.imread(image_dict['file_name'])
            image_dict['height'], image_dict['width'] = image.shape[0], image.shape[1] 

            objs = []
            for index, rows in group.iterrows():
                obj = {}

                obj["bbox"] = [rows['BoxXMin'], rows['BoxYMin'], rows['BoxXMax'], rows['BoxYMax']]
                obj["bbox_mode"] = BoxMode.XYXY_ABS

                maskimage = cv2.resize(cv2.imread(str(segmentation_dir / split / '{}-masks-{}'.format(split, imageid[0]) / rows['MaskPath'] )), (image_dict['width'], image_dict['height']))

                imgray = cv2.cvtColor(maskimage, cv2.COLOR_BGR2GRAY)
                ret, thresh = cv2.threshold(imgray, 127, 255, 0)
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                poly = contours[0].squeeze().tolist()
                poly = [p for x in poly for p in x]

                obj["segmentation"] = [poly]
                obj["category_id"] = orig_id2cat_id[rows['LabelName']] - 1
            
                objs.append(obj)
            
            image_dict["annotations"] = objs

            dataset_dicts.append(image_dict)
        
        return dataset_dicts