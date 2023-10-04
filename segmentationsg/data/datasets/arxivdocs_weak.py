import h5py
import json
import math
from math import floor
from PIL import Image, ImageDraw
import random
import socket
import pathlib
import os
import torch
import numpy as np
import pickle
import yaml
from detectron2.config import get_cfg
from detectron2.structures import Instances, Boxes, pairwise_iou, BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
import logging
import torch.distributed as dist
import copy
from segmentationsg.data.dataset_mapper import SceneGraphDatasetMapper
from tqdm import tqdm

from multiprocessing import Pool
#from segmentationsg.data.tools import is_valid_dataset_dict


import copy
from detectron2.data import detection_utils as utils
from collections import defaultdict
from segmentationsg.data.dataset_mapper import filter_empty_instances
from detectron2.data import transforms as T


logger = logging.getLogger(__name__)

def validate_single_record_dict(input):
    #(dataset_dict, is_train, recompute_boxes, image_format, filter_duplicate_relations) = input
    (dataset_dict, validation_mapper) = input
    return_val = validation_mapper(dataset_dict)
    if return_val is not None:
        record_valid = True
        return [record_valid, dataset_dict]
    else:
        record_valid = False
        # record['is_valid'] = False
        # print("Dataset record for {} has no valid relations after valid bbox filtering! Skipping..".format(record['file_name']))
        removed_record = {'file_name': dataset_dict['file_name'], 'image_id': dataset_dict['image_id']}
        return [record_valid, removed_record]



class ArxivdocsWeakTrainData:
    """
    Register data for Arxivdocs training
    """
    def __init__(self, cfg, split='train'):
        self.cfg = cfg
        self.split = split
        if split == 'train':
            self.mask_location = cfg.DATASETS.ARXIVDOCS_WEAK.TRAIN_MASKS
        elif split == 'val':
            self.mask_location = cfg.DATASETS.ARXIVDOCS_WEAK.VAL_MASKS
        else:
            self.mask_location = cfg.DATASETS.ARXIVDOCS_WEAK.TEST_MASKS
        self.mask_exists = os.path.isfile(self.mask_location)
        self.clamped = True if "clamped" in self.mask_location else ""
        self.clipped = cfg.DATASETS.ARXIVDOCS_WEAK.CLIPPED
        self.precompute = False if (self.cfg.DATASETS.ARXIVDOCS_WEAK.FILTER_EMPTY_RELATIONS or self.cfg.DATASETS.ARXIVDOCS_WEAK.FILTER_NON_OVERLAP) else True
        #ids = segmentationsg.data.datasets.images_to_remove.ARXIVDOCS_WEAK_IMAGES_TO_REMOVE
        #self.ids_to_remove = ids
        self.ids_to_remove = []
        # self._process_data()
        self.dataset_dicts = self._fetch_data_dict(use_cached=True, validate_filtered_relations=False)
        self.register_dataset()
        try:
            self.get_statistics()
        except:
            pass
        
    def register_dataset(self):
        """
        Register datasets to use with Detectron2
        """
        DatasetCatalog.register('ADwk_{}'.format(self.split), lambda: self.dataset_dicts)

        #Get labels
        if self.split == 'train':
            self.mapping_dictionary = json.load(open(self.cfg.DATASETS.ARXIVDOCS_WEAK.TRAIN_MAPPING_DICTIONARY, 'r'))
        elif self.split == 'val':
            self.mapping_dictionary = json.load(open(self.cfg.DATASETS.ARXIVDOCS_WEAK.VAL_MAPPING_DICTIONARY, 'r'))
        elif self.split == 'test':
            self.mapping_dictionary = json.load(open(self.cfg.DATASETS.ARXIVDOCS_WEAK.TEST_MAPPING_DICTIONARY, 'r'))

        self.idx_to_classes = sorted(self.mapping_dictionary['label_to_idx'], key=lambda k: self.mapping_dictionary['label_to_idx'][k])
        self.idx_to_predicates = sorted(self.mapping_dictionary['predicate_to_idx'], key=lambda k: self.mapping_dictionary['predicate_to_idx'][k])
        self.idx_to_attributes = sorted(self.mapping_dictionary['attribute_to_idx'], key=lambda k: self.mapping_dictionary['attribute_to_idx'][k])
        MetadataCatalog.get('ADwk_{}'.format(self.split)).set(thing_classes=self.idx_to_classes, predicate_classes=self.idx_to_predicates, attribute_classes=self.idx_to_attributes)
    
    def _fetch_data_dict(self, use_cached=True, validate_filtered_relations=False):
        """
        Load data in detectron format
        """
        fileName = "tmp/ADwk_{}_data_{}{}{}{}{}.pkl".format(self.split, 'masks' if self.mask_exists else '', '_oi' if 'oi' in self.mask_location else '', "_clamped" if self.clamped else "", "_precomp" if self.precompute else "", "_clipped" if self.clipped else "")
        if os.path.isfile(fileName) and use_cached is True:
            #If data has been processed earlier, load that to save time
            print("Found for cached data file: ", fileName)
            with open(fileName, 'rb') as inputFile:
                dataset_dicts = pickle.load(inputFile)
        else:
            #Process data
            os.makedirs('tmp', exist_ok=True)
            dataset_dicts = self._process_data(validate_filtered_relations=validate_filtered_relations)
            #rank = dist.get_rank()
            if not dist.is_initialized() or dist.get_rank() == 0:
                print("Caching data file: ", fileName)
                with open(fileName, 'wb') as inputFile:
                    pickle.dump(dataset_dicts, inputFile)
        return dataset_dicts
            
    def _process_data(self, validate_filtered_relations=False):
        if self.split == 'train':
            self.ARXIVDOCS_WEAK_attribute_h5 = h5py.File(
                self.cfg.DATASETS.ARXIVDOCS_WEAK.TRAIN_ARXIVDOCS_WEAK_ATTRIBUTE_H5, 'r')
        elif self.split == 'val':
            self.ARXIVDOCS_WEAK_attribute_h5 = h5py.File(
                self.cfg.DATASETS.ARXIVDOCS_WEAK.VAL_ARXIVDOCS_WEAK_ATTRIBUTE_H5, 'r')
        elif self.split == 'test':
            self.ARXIVDOCS_WEAK_attribute_h5 = h5py.File(
                self.cfg.DATASETS.ARXIVDOCS_WEAK.TEST_ARXIVDOCS_WEAK_ATTRIBUTE_H5, 'r')

        # Remove corrupted images
        if self.split == 'train':
            image_data = json.load(open(self.cfg.DATASETS.ARXIVDOCS_WEAK.TRAIN_IMAGE_DATA, 'r'))
        elif self.split == 'val':
            image_data = json.load(open(self.cfg.DATASETS.ARXIVDOCS_WEAK.VAL_IMAGE_DATA, 'r'))
        elif self.split == 'test':
            image_data = json.load(open(self.cfg.DATASETS.ARXIVDOCS_WEAK.TEST_IMAGE_DATA, 'r'))
        #self.corrupted_ims = ['1592', '1722', '4616', '4617']
        self.image_data = []

        #self.corrupted_ims_ADwk =  ['1607.01653_3/1607.01653_3-0.png']

        #NOTE: this is a tweak/workaroudn for custom paths on some clusters (arxivdocs weak had to be adapted due to its large size)
        hostname_string = socket.gethostname()
        affected_hostname_substring = 'eu-'
        remove_subdir_from_file_name = False
        print('checking hostname ({}) for substring: {}'.format(hostname_string, affected_hostname_substring))
        if affected_hostname_substring in hostname_string:
            print(
                'using no subdirectories for train images for this host!'.format(hostname_string,
                                                                                 affected_hostname_substring))
            remove_subdir_from_file_name = True

        for i, img in enumerate(image_data):
        #    if str(img['image_id']) in self.corrupted_ims:
        #        continue
            if remove_subdir_from_file_name is True:
                relative_path = img['file_name']
                relative_path_parts = pathlib.Path(relative_path).parts
                relative_path_without_last_subdir_list = relative_path_parts[:-2] + relative_path_parts[-1:]
                relative_path_without_last_subdir = pathlib.Path(*relative_path_without_last_subdir_list)
                img['file_name'] = relative_path_without_last_subdir
            self.image_data.append(img)


        if self.split == 'train':
            assert(len(self.image_data) == 593586)
        elif self.split != 'train':
            raise NotImplementedError
        self.masks = None
        if self.mask_location != "":
            try:
                with open(self.mask_location, 'rb') as f:
                    self.masks = pickle.load(f)
            except:
                pass

        #valid_dataset_mask = [x['is_valid'] for x in self.dataset_dicts]



        dataset_dicts = self._load_graphs(validate_filtered_relations=validate_filtered_relations)
        print('len of dicts for split {}: {}'.format(self.split, len(dataset_dicts)))




        return dataset_dicts

    def get_statistics(self, eps=1e-3, bboxes_must_overlap_for_relation=False):
        num_object_classes = len(MetadataCatalog.get('ADwk_{}'.format(self.split)).thing_classes) + 1
        num_relation_classes = len(MetadataCatalog.get('ADwk_{}'.format(self.split)).predicate_classes) + 1
        
        fg_matrix = np.zeros((num_object_classes, num_object_classes, num_relation_classes), dtype=np.int64)
        bg_matrix = np.zeros((num_object_classes, num_object_classes), dtype=np.int64)
        for idx, data in enumerate(self.dataset_dicts):
            gt_relations = data['relations']
            gt_classes = np.array([x['category_id'] for x in data['annotations']])
            gt_boxes = np.array([x['bbox'] for x in data['annotations']])
            for (o1, o2), rel in zip(gt_classes[gt_relations[:,:2]], gt_relations[:,2]):
                fg_matrix[o1, o2, rel] += 1
            for (o1, o2) in gt_classes[np.array(box_filter(gt_boxes, must_overlap=bboxes_must_overlap_for_relation), dtype=int)]:
                bg_matrix[o1, o2] += 1
        bg_matrix += 1
        fg_matrix[:, :, -1] = bg_matrix
        pred_dist = np.log(fg_matrix / fg_matrix.sum(2)[:, :, None] + eps)

        result = {
            'fg_matrix': torch.from_numpy(fg_matrix),
            'pred_dist': torch.from_numpy(pred_dist).float(),
            'obj_classes': self.idx_to_classes + ['__background__'],
            'rel_classes': self.idx_to_predicates + ['__background__'],
            'att_classes': self.idx_to_attributes,
        }
        MetadataCatalog.get('ADwk_{}'.format(self.split)).set(statistics=result)
        return result


    def _load_graphs(self, validate_filtered_relations=False):
        """
        Parse examples and create dictionaries
        """
        data_split = self.ARXIVDOCS_WEAK_attribute_h5['split'][:]
        if self.split == 'test':
            split_flag = 2
        elif self.split == 'val':
            split_flag = 1
        elif self.split == 'train':
            split_flag = 0
        #split_flag = 2 if self.split == 'test' else 0
        split_mask = data_split == split_flag


        #valid_dataset_mask = [x['is_valid'] for x in self.dataset_dicts]

        #Filter images without bounding boxes
        split_mask &= self.ARXIVDOCS_WEAK_attribute_h5['img_to_first_box'][:] >= 0
        if self.cfg.DATASETS.ARXIVDOCS_WEAK.FILTER_EMPTY_RELATIONS:
            split_mask &= self.ARXIVDOCS_WEAK_attribute_h5['img_to_first_rel'][:] >= 0

        image_index = np.where(split_mask)[0]

        #NOTE: our data is already pre-split and contains only train, val or test

        split_mask = np.zeros_like(data_split).astype(bool)
        split_mask[image_index] = True
        
        # Get box information
        all_labels = self.ARXIVDOCS_WEAK_attribute_h5['labels'][:, 0]
        all_attributes = self.ARXIVDOCS_WEAK_attribute_h5['attributes'][:, :]
        all_boxes = self.ARXIVDOCS_WEAK_attribute_h5['boxes_{}'.format(self.cfg.DATASETS.ARXIVDOCS_WEAK.BOX_SCALE)][:]  # cx,cy,w,h
        assert np.all(all_boxes[:, :2] >= 0)  # sanity check
        assert np.all(all_boxes[:, 2:] > 0)  # no empty box
        
        # Convert from xc, yc, w, h to x1, y1, x2, y2
        all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
        all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]
        
        first_box_index = self.ARXIVDOCS_WEAK_attribute_h5['img_to_first_box'][split_mask]
        last_box_index = self.ARXIVDOCS_WEAK_attribute_h5['img_to_last_box'][split_mask]
        first_relation_index = self.ARXIVDOCS_WEAK_attribute_h5['img_to_first_rel'][split_mask]
        last_relation_index = self.ARXIVDOCS_WEAK_attribute_h5['img_to_last_rel'][split_mask]

        #Load relation labels
        all_relations = self.ARXIVDOCS_WEAK_attribute_h5['relationships'][:]
        all_relation_predicates = self.ARXIVDOCS_WEAK_attribute_h5['predicates'][:, 0]

        image_indexer = np.arange(len(self.image_data))[split_mask]
        # Iterate over images
        dataset_dicts = []

#        if validate_filtered_relations is True:
            #validation_mapper = SceneGraphDatasetMapper(self.cfg, is_train=self.split == 'train')




        print("loading and validating all images. Phase 1..")
        for idx, _ in enumerate(tqdm(image_index)):
            record = {}
            #Get image metadata
            image_data = self.image_data[image_indexer[idx]]

            if self.split == 'train':
                record['file_name'] = os.path.join(self.cfg.DATASETS.ARXIVDOCS_WEAK.TRAIN_IMAGES, image_data['file_name'])
            elif self.split == 'val':
                record['file_name'] = os.path.join(self.cfg.DATASETS.ARXIVDOCS_WEAK.VAL_IMAGES, image_data['file_name'])
            elif self.split == 'test':
                record['file_name'] = os.path.join(self.cfg.DATASETS.ARXIVDOCS_WEAK.TEST_IMAGES, image_data['file_name'])




            record['image_id'] = image_data['image_id']
            record['height'] = image_data['height']
            record['width'] = image_data['width']
            if self.clipped:
                if image_data['coco_id'] in self.ids_to_remove:
                    continue
            #Get annotations
            boxes = all_boxes[first_box_index[idx]:last_box_index[idx] + 1, :]
            gt_classes = all_labels[first_box_index[idx]:last_box_index[idx] + 1]
            gt_attributes = all_attributes[first_box_index[idx]:last_box_index[idx] + 1, :]

            if first_relation_index[idx] > -1:
                predicates = all_relation_predicates[first_relation_index[idx]:last_relation_index[idx] + 1]
                objects = all_relations[first_relation_index[idx]:last_relation_index[idx] + 1] - first_box_index[idx]
                predicates = predicates - 1
                relations = np.column_stack((objects, predicates))
            else:
                assert not self.cfg.DATASETS.ARXIVDOCS_WEAK.FILTER_EMPTY_RELATIONS
                relations = np.zeros((0, 3), dtype=np.int32)
            
            if self.cfg.DATASETS.ARXIVDOCS_WEAK.FILTER_NON_OVERLAP and self.split == 'train':
                # Remove boxes that don't overlap
                boxes_list = Boxes(boxes)
                ious = pairwise_iou(boxes_list, boxes_list)
                relation_boxes_ious = ious[relations[:,0], relations[:,1]]
                iou_indexes = np.where(relation_boxes_ious > 0.0)[0]
                if iou_indexes.size > 0:
                    relations = relations[iou_indexes]
                else:
                    #Ignore image
                    continue
            #Get masks if possible
            if self.masks is not None:
                try:
                    gt_masks = self.masks[image_data['image_id']]
                except:
                    print (image_data['image_id'])
            record['relations'] = relations
            objects = []
            # if len(boxes) != len(gt_masks):
            mask_idx = 0
            for obj_idx in range(len(boxes)):
                resized_box = boxes[obj_idx] / self.cfg.DATASETS.ARXIVDOCS_WEAK.BOX_SCALE * max(record['height'], record['width'])
                obj = {
                      "bbox": resized_box.tolist(),
                      "bbox_mode": BoxMode.XYXY_ABS,
                      "category_id": gt_classes[obj_idx] - 1,
                      "attribute": gt_attributes[obj_idx],           
                }
                if self.masks is not None:
                    if gt_masks['empty_index'][obj_idx]:
                        refined_poly = []
                        for poly_idx, poly in enumerate(gt_masks['polygons'][mask_idx]):
                            if len(poly) >= 6:
                                refined_poly.append(poly)
                        obj["segmentation"] = refined_poly
                        mask_idx += 1
                    else:
                        obj["segmentation"] = []
                    if len(obj["segmentation"]) > 0:
                        objects.append(obj)
                else:
                    objects.append(obj)
            record['annotations'] = objects

            dataset_dicts.append(record)
        filtered_dataset_dicts = dataset_dicts
#        filtered_dataset_dicts = []
#        removed_records = []
#        if validate_filtered_relations is True:
#            print("loading and validating all images. Phase 2..")
#
#            is_train = self.split == 'train'
#
#            recompute_boxes = False
#            if self.cfg.INPUT.CROP.ENABLED and is_train:
#                recompute_boxes = self.cfg.MODEL.MASK_ON
#            image_format = self.cfg.INPUT.FORMAT
#            filter_duplicate_relations = self.cfg.DATASETS.VISUAL_GENOME.FILTER_DUPLICATE_RELATIONS
#
#            #imap_function_inputs = [(dataset_dict, is_train, recompute_boxes, image_format, filter_duplicate_relations) for dataset_dict in dataset_dicts]
#            validation_mapper = SceneGraphDatasetMapper(self.cfg, is_train=is_train)
#            imap_function_inputs = [(dataset_dict, validation_mapper) for dataset_dict in dataset_dicts]
#            pool_size = 20
#            #output_dict_by_i = dict()
#            with Pool(processes=pool_size) as p:
#                max_ = len(imap_function_inputs)
#                with tqdm(total=max_) as pbar:
#                    for i, output_tuple in enumerate(p.imap(validate_single_record_dict, imap_function_inputs)):
#                        [record_is_valid, record_output] = output_tuple
#                        if record_is_valid:
#                            filtered_dataset_dicts.append(record_output)
#                            #output_dict_by_i[i] = record_output
#                        else:
#                            removed_records.append(record_output)
#                        pbar.update()
#
#        else:
#            filtered_dataset_dicts = dataset_dicts
#
#        if len(removed_records) > 0:
#            output_file_path = os.path.join('tmp', 'invalid_records_ADwk_{}.json'.format(self.split))
#            if not dist.is_initialized() or dist.get_rank() == 0:
#                print('save bad file paths to {}'.format(output_file_path))
#                os.makedirs('tmp', exist_ok=True)
#                with open(output_file_path, 'w') as out_file:
#                    json.dump(removed_records, out_file)
#
#        if len(removed_records) > 0:
#            logger.info("Removed {} samples with no valid relations after filtering".format(len(removed_records)))
        return filtered_dataset_dicts

def box_filter(boxes, must_overlap=False):
    """ Only include boxes that overlap as possible relations. 
    If no overlapping boxes, use all of them."""
    n_cands = boxes.shape[0]

    overlaps = bbox_overlaps(boxes.astype(np.float), boxes.astype(np.float), to_move=0) > 0
    np.fill_diagonal(overlaps, 0)

    all_possib = np.ones_like(overlaps, dtype=np.bool)
    np.fill_diagonal(all_possib, 0)

    if must_overlap:
        possible_boxes = np.column_stack(np.where(overlaps))

        if possible_boxes.size == 0:
            possible_boxes = np.column_stack(np.where(all_possib))
    else:
        possible_boxes = np.column_stack(np.where(all_possib))
    return possible_boxes

def bbox_overlaps(boxes1, boxes2, to_move=1):
    """
    boxes1 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    boxes2 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    """
    #print('boxes1: ', boxes1.shape)
    #print('boxes2: ', boxes2.shape)
    num_box1 = boxes1.shape[0]
    num_box2 = boxes2.shape[0]
    lt = np.maximum(boxes1.reshape([num_box1, 1, -1])[:,:,:2], boxes2.reshape([1, num_box2, -1])[:,:,:2]) # [N,M,2]
    rb = np.minimum(boxes1.reshape([num_box1, 1, -1])[:,:,2:], boxes2.reshape([1, num_box2, -1])[:,:,2:]) # [N,M,2]

    wh = (rb - lt + to_move).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    return inter



