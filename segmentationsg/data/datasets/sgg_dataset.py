import h5py
import json
import math
from math import floor
from PIL import Image, ImageDraw
import random
import os
import torch
import pathlib
import numpy as np
import socket
import pickle
import yaml
from detectron2.config import get_cfg
from detectron2.structures import Instances, Boxes, pairwise_iou, BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
import logging
import torch.distributed as dist
from collections import defaultdict
import copy


class SggDataset:
    """
    Register data for Arxivdocs training
    """

    def __init__(self, cfg, dataset_name, dataset_image_dir, mapping_dictionary_json_path, scenegraph_dataset_h5_path, dataset_image_data_json_path, scenegraph_dictionary_cache_file, dataset_split='train', is_clipped=False, filter_empty_relations=True, filter_duplicate_relations=True, filter_non_overlapping_boxes=False, box_scale=1024, mask_location='', use_basename_for_file_paths=False, remove_subdir_from_file_name=False):
        self.dataset_name = dataset_name
        self.filter_empty_relations = filter_empty_relations
        self.filter_duplicate_relations = filter_duplicate_relations
        self.filter_non_overlapping_boxes = filter_non_overlapping_boxes
        self.scenegraph_dictionary_cache_file  = scenegraph_dictionary_cache_file
        self.box_scale = box_scale
        self.cfg = cfg
        self.split = dataset_split
        self.dataset_image_dir = dataset_image_dir
        self.mapping_dictionary_json_path = mapping_dictionary_json_path
        self.scenegraph_dataset_h5_path = scenegraph_dataset_h5_path
        self.dataset_image_data_json_path = dataset_image_data_json_path
        self.mask_location = mask_location
        self.mask_exists = os.path.isfile(self.mask_location)
        self.clamped = True if "clamped" in self.mask_location else ""
        self.clipped = is_clipped
        self.use_basename_for_file_paths = use_basename_for_file_paths
        self.use_basename_for_file_paths = use_basename_for_file_paths
        self.remove_subdir_from_file_name = remove_subdir_from_file_name

        self.precompute = False if (
                self.filter_empty_relations or self.filter_non_overlapping_boxes) else True
        # self.ids_to_remove = ids
        self.ids_to_remove = []
        # self._process_data()
        self.dataset_dicts = self._fetch_data_dict()
        self.register_dataset(self.dataset_name)
        if self.mapping_dictionary_json_path is not None:
            self.create_metadata_from_mapping_dictionary(self.mapping_dictionary_json_path)
        else:
            print("Mapping dictionary json path is none. Could not create metadata entry for {}".format(self.dataset_name))

        #try:
        print('creating statistics for {}'.format(self.dataset_name))
        self.get_statistics()
        #except:
        #    print('failed creating statistics for {}'.format(self.dataset_name))
        #    pass

    def register_dataset(self, dataset_name):
        """
        Register datasets to use with Detectron2
        """
        DatasetCatalog.register('{}'.format(dataset_name), lambda: self.dataset_dicts)

    def create_metadata_from_mapping_dictionary(self, mapping_dictionary_json_path):
        # Get labels
        self.mapping_dictionary = json.load(
            open(mapping_dictionary_json_path, 'r'))

        self.idx_to_classes = sorted(self.mapping_dictionary['label_to_idx'],
                                     key=lambda k: self.mapping_dictionary['label_to_idx'][k])
        self.idx_to_predicates = sorted(self.mapping_dictionary['predicate_to_idx'],
                                        key=lambda k: self.mapping_dictionary['predicate_to_idx'][
                                            k])
        self.idx_to_attributes = sorted(self.mapping_dictionary['attribute_to_idx'],
                                        key=lambda k: self.mapping_dictionary['attribute_to_idx'][
                                            k])
        
        #TODO: add thing_colors
        
        MetadataCatalog.get('{}'.format(self.dataset_name)).set(thing_classes=self.idx_to_classes,
                                                               predicate_classes=self.idx_to_predicates,
                                                               attribute_classes=self.idx_to_attributes)

    def _fetch_data_dict(self):
        """
        Load data in detectron format
        """

        if self.scenegraph_dictionary_cache_file  is not None:
            print("Using pickled dataset dictionary: ", self.scenegraph_dictionary_cache_file)
            assert os.path.isfile(self.scenegraph_dictionary_cache_file)
            # If data has been processed earlier, load that to save time
            print("Found for cached data file: ", self.scenegraph_dictionary_cache_file)
            with open(self.scenegraph_dictionary_cache_file, 'rb') as inputFile:
                dataset_dicts = pickle.load(inputFile)
        else:

            fileName = "tmp/{}_cached.pkl".format(self.dataset_name)
            # Process data
            os.makedirs('tmp', exist_ok=True)
            dataset_dicts = self._process_data()
            if not dist.is_initialized() or dist.get_rank() == 0:
                print("Caching data file: ", fileName)
                with open(fileName, 'wb') as inputFile:
                    pickle.dump(dataset_dicts, inputFile)
        return dataset_dicts

    def _process_data(self):
        self.scenegraph_dataset_h5_contents = h5py.File(
            self.scenegraph_dataset_h5_path, 'r')

        # Remove corrupted images
        image_data = json.load(open(self.dataset_image_data_json_path, 'r'))
        # self.corrupted_ims = ['1592', '1722', '4616', '4617']


        if self.remove_subdir_from_file_name is True:
            #NOTE: this is a tweak/workaround for custom paths on some clusters (arxivdocs weak had to be adapted due to its large size)
            hostname_string = socket.gethostname()
            affected_hostname_substring = 'eu-'
            print('checking hostname ({}) for substring: {}'.format(hostname_string, affected_hostname_substring))
            if affected_hostname_substring in hostname_string:
                print(
                    'using no subdirectories for train images for this host!'.format(hostname_string,
                                                                                     affected_hostname_substring))
            else:
                self.remove_subdir_from_file_name = False

        self.image_data = []
        for i, img in enumerate(image_data):
            #    if str(img['image_id']) in self.corrupted_ims:
            #        continue
            if self.remove_subdir_from_file_name is True:
                relative_path = img['file_name']
                relative_path_parts = pathlib.Path(relative_path).parts
                relative_path_without_last_subdir_list = relative_path_parts[:-2] + relative_path_parts[-1:]
                relative_path_without_last_subdir = pathlib.Path(*relative_path_without_last_subdir_list)
                img['file_name'] = relative_path_without_last_subdir
            self.image_data.append(img)



        self.masks = None
        if self.mask_location != "":
            try:
                with open(self.mask_location, 'r') as f:
                    segm_data = json.load(f)
                    self.masks = segm_data
            except:
                pass
        dataset_dicts = self._load_graphs()
        print('len of dicts for split {} in {}: {}'.format(self.split, self.dataset_name, len(dataset_dicts)))
        return dataset_dicts

    def get_statistics(self, eps=1e-3, bboxes_must_overlap_for_relation=False):
        num_object_classes = len(
            MetadataCatalog.get('{}'.format(self.dataset_name)).thing_classes) + 1
        num_relation_classes = len(
            MetadataCatalog.get('{}'.format(self.dataset_name)).predicate_classes) + 1

        fg_matrix = np.zeros((num_object_classes, num_object_classes, num_relation_classes),
                             dtype=np.int64)
        bg_matrix = np.zeros((num_object_classes, num_object_classes), dtype=np.int64)
        total_boxes_in_all_images = 0
        total_relations_in_all_images = 0
        for idx, data in enumerate(self.dataset_dicts):
            gt_relations = data['relations']
            gt_classes = np.array([x['category_id'] for x in data['annotations']])
            total_boxes_in_all_images += len(data['annotations'])
            total_relations_in_all_images += len(data['relations'])
            gt_boxes = np.array([x['bbox'] for x in data['annotations']])
            if len(gt_relations) != 0:
                for (o1, o2), rel in zip(gt_classes[gt_relations[:, :2]], gt_relations[:, 2]):
                    fg_matrix[o1, o2, rel] += 1
            else:
                print("Warning: gt_relations is empty for record of img: {}".format(data['image_id']))
            for (o1, o2) in gt_classes[
                np.array(box_filter(gt_boxes, must_overlap=bboxes_must_overlap_for_relation),
                         dtype=int)]:
                bg_matrix[o1, o2] += 1
        avg_boxes_per_image = total_boxes_in_all_images / len(self.dataset_dicts)
        avg_relations_per_image = total_relations_in_all_images / len(self.dataset_dicts)
        bg_matrix += 1
        fg_matrix[:, :, -1] = bg_matrix
        pred_dist = np.log(fg_matrix / fg_matrix.sum(2)[:, :, None] + eps)

        result = {
            'fg_matrix': torch.from_numpy(fg_matrix),
            'pred_dist': torch.from_numpy(pred_dist).float(),
            'obj_classes': self.idx_to_classes + ['__background__'],
            'rel_classes': self.idx_to_predicates + ['__background__'],
            'att_classes': self.idx_to_attributes,
            'avg_boxes_per_image': avg_boxes_per_image,
            'avg_relations_per_image': avg_relations_per_image,
            'total_boxes': total_boxes_in_all_images,
            'total_relations': total_relations_in_all_images
        }
        print('saving statistics for {}'.format(self.dataset_name))
        MetadataCatalog.get('{}'.format(self.dataset_name)).set(statistics=result)
        return result

    def _load_graphs(self):
        """
        Parse examples and create dictionaries
        """
        data_split = self.scenegraph_dataset_h5_contents['split'][:]
        if self.split == 'test':
            split_flag = 2
        elif self.split == 'val':
            split_flag = 1
        elif self.split == 'train':
            split_flag = 0
        # split_flag = 2 if self.split == 'test' else 0
        split_mask = data_split == split_flag

        # Filter images without bounding boxes
        split_mask &= self.scenegraph_dataset_h5_contents['img_to_first_box'][:] >= 0
        if self.filter_empty_relations:
            split_mask &= self.scenegraph_dataset_h5_contents['img_to_first_rel'][:] >= 0
        image_index = np.where(split_mask)[0]


        split_mask = np.zeros_like(data_split).astype(bool)
        split_mask[image_index] = True

        # Get box information
        all_labels = self.scenegraph_dataset_h5_contents['labels'][:, 0]
        all_attributes = self.scenegraph_dataset_h5_contents['attributes'][:, :]
        all_boxes = self.scenegraph_dataset_h5_contents[
                        'boxes_{}'.format(self.box_scale)][
                    :]  # cx,cy,w,h
        assert np.all(all_boxes[:, :2] >= 0)  # sanity check
        assert np.all(all_boxes[:, 2:] > 0)  # no empty box

        # Convert from xc, yc, w, h to x1, y1, x2, y2
        all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
        all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]

        first_box_index = self.scenegraph_dataset_h5_contents['img_to_first_box'][split_mask]
        last_box_index = self.scenegraph_dataset_h5_contents['img_to_last_box'][split_mask]
        first_relation_index = self.scenegraph_dataset_h5_contents['img_to_first_rel'][split_mask]
        last_relation_index = self.scenegraph_dataset_h5_contents['img_to_last_rel'][split_mask]

        # Load relation labels
        all_relations = self.scenegraph_dataset_h5_contents['relationships'][:]
        all_relation_predicates = self.scenegraph_dataset_h5_contents['predicates'][:, 0]

        image_indexer = np.arange(len(self.image_data))[split_mask]
        # Iterate over images
        dataset_dicts = []
        #all_img_ids = []
        for idx, _ in enumerate(image_index):
            record = {}
            # Get image metadata
            image_data = self.image_data[image_indexer[idx]]

            if self.use_basename_for_file_paths is False:
                file_name = image_data['file_name']
            else:
                file_name = os.path.basename(image_data['file_name'])
            full_file_path = os.path.join(self.dataset_image_dir,
                                               file_name)
            record['file_name'] = full_file_path
            record['image_id'] = image_data['image_id']
            record['height'] = image_data['height']
            record['width'] = image_data['width']
            if self.clipped:
                if image_data['coco_id'] in self.ids_to_remove:
                    continue
            # Get annotations
            boxes = all_boxes[first_box_index[idx]:last_box_index[idx] + 1, :]
            gt_classes = all_labels[first_box_index[idx]:last_box_index[idx] + 1]
            gt_attributes = all_attributes[first_box_index[idx]:last_box_index[idx] + 1, :]

            if first_relation_index[idx] > -1:
                predicates = all_relation_predicates[
                             first_relation_index[idx]:last_relation_index[idx] + 1]
                objects = all_relations[first_relation_index[idx]:last_relation_index[idx] + 1] - \
                          first_box_index[idx]
                predicates = predicates - 1
                relations = np.column_stack((objects, predicates))
            else:
                assert not self.filter_empty_relations
                relations = np.zeros((0, 3), dtype=np.int32)

            if self.filter_non_overlapping_boxes and self.split == 'train':
                # Remove boxes that don't overlap
                boxes_list = Boxes(boxes)
                ious = pairwise_iou(boxes_list, boxes_list)
                relation_boxes_ious = ious[relations[:, 0], relations[:, 1]]
                iou_indexes = np.where(relation_boxes_ious > 0.0)[0]
                if iou_indexes.size > 0:
                    relations = relations[iou_indexes]
                else:
                    # Ignore image
                    continue
            # Get masks if possible
            if self.masks is not None:
                try:
                    gt_masks = self.masks[f"{image_data['image_id']}"]
                except:
                    print('could not find mask in dictionary for image id: {}'.format(image_data['image_id']))
            record['relations'] = relations
            objects = []
            # if len(boxes) != len(gt_masks):
            # mask_idx = 0
            for obj_idx in range(len(boxes)):
                resized_box = boxes[obj_idx] / self.box_scale * max(
                    record['height'], record['width'])
                obj = {
                    "bbox": resized_box.tolist(),
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": gt_classes[obj_idx] - 1,
                    "attribute": gt_attributes[obj_idx],
                }
                if self.masks is not None:
                    debug_empty_index = gt_masks['empty_index']
                    if obj_idx > (len(gt_masks['empty_index']) - 1):
                        raise AssertionError("obj_idx {} not contained in gt_masks[emtpy_index] of length {}".format(obj_idx, len(gt_masks['empty_index'])))
                    if gt_masks['empty_index'][obj_idx]:
                        refined_poly = []
                        if len(gt_masks['empty_index']) <= obj_idx:
                            print('gt_masks for empty index is smaller than obj_idx: {} vs {}'.format(len(gt_masks['empty_index']), obj_idx))
                        for poly_idx, poly in enumerate(gt_masks['empty_index'][obj_idx]['segmentation']):
                            if len(poly) >= 6:
                                refined_poly.append(poly)
                        obj["segmentation"] = refined_poly
                        # mask_idx += 1
                    else:
                        obj["segmentation"] = []
                    if len(obj["segmentation"]) > 0:
                        objects.append(obj)
                else:
                    objects.append(obj)
            #all_img_ids.append(image_data['image_id'])
            record['annotations'] = objects
            dataset_dicts.append(record)
        return dataset_dicts



    def get_all_image_ids(self):
        all_img_ids =[]
        for record in self.dataset_dicts:
            all_img_ids.append(record['image_id'])
        return all_img_ids

    def subsample_and_save_with_image_ids(self, image_ids, dataset_dict_out_path):
        current_dataset_dict = self.dataset_dicts
        new_dataset_dict = []
        for record in current_dataset_dict:
            if record['image_id'] in image_ids:
                new_dataset_dict.append(record)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print("Saving subsampled dataset dict to ", dataset_dict_out_path)
            with open(dataset_dict_out_path, 'wb') as inputFile:
                pickle.dump(new_dataset_dict, inputFile)
        return new_dataset_dict

    def subsample_and_save_with_new_annotations(self, image_ids, new_annotations, dataset_dict_out_path, labeling_histories):
        current_dataset_dict = self.dataset_dicts
        new_dataset_dict = []

        new_annotation_image_ids = []
        gt_annotations_to_recover_per_image = dict()
        for labeling_history in labeling_histories:
             img_id = labeling_history['image_id']
             gt_annotations_to_recover_per_image[img_id] = labeling_history['selected_gt_indices']
             new_annotation_image_ids.append(img_id)

        #        for new_annotation in new_annotations:
#            img_id = new_annotation['image_id']
#            gt_annotations_to_recover_per_image[img_id].append(new_annotation)
#            new_annotation_image_ids.append(img_id)

        for record in current_dataset_dict:
            if record['image_id'] in image_ids:
                if record['image_id'] in new_annotation_image_ids:
#                    print("image {} comes from new annotations".format(record['image_id']))
                    filtered_record = copy.deepcopy(record)
                    gt_ids_to_keep = gt_annotations_to_recover_per_image[record['image_id']]
                    old_to_new_id_mappings = dict()
                    filtered_annotations = []
                    new_ann_counter = 0
                    for old_ann_id,ann in enumerate(filtered_record['annotations']):
                        if old_ann_id in gt_ids_to_keep:
                            old_to_new_id_mappings[old_ann_id] = new_ann_counter
                            new_ann_counter += 1
                            filtered_annotations.append(ann)
                    filtered_record['annotations'] = filtered_annotations
                    orig_relations = filtered_record['relations']
#                    print('all relations: {}'.format(orig_relations))
                    remaining_valid_relation_indices = []
                    for orig_rel in orig_relations:
                        head,tail,_ = orig_rel
                        if head in gt_ids_to_keep and tail in gt_ids_to_keep:
                            adjusted_rel = copy.deepcopy(orig_rel)
                            adjusted_rel[0] = old_to_new_id_mappings[adjusted_rel[0]]
                            adjusted_rel[1] = old_to_new_id_mappings[adjusted_rel[1]]
                            remaining_valid_relation_indices.append(adjusted_rel)
                    filtered_record['relations'] = np.asarray(remaining_valid_relation_indices)
#                    print('relations after filtering: {}'.format(filtered_record['relations']))
#                    print('orig relations type: {}'.format(type(record['relations'])))
#                    print('new relations type: {}'.format(type(filtered_record['relations'])))
                    new_dataset_dict.append(filtered_record)
                    #print("preserved {} annotations from originally {}; preserved {} relations from originally {}".format(len(filtered_record['annotations']), len(record['annotations']), len(filtered_record['relations']), len(record['relations'])))
                    #filter record for GT annotations
                else:
                    print("image {} comes from new annotations".format(record['image_id']))
                    new_dataset_dict.append(record)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print("Saving subsampled dataset dict to ", dataset_dict_out_path)
            with open(dataset_dict_out_path, 'wb') as inputFile:
                pickle.dump(new_dataset_dict, inputFile)
        return new_dataset_dict



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
    # print('boxes1: ', boxes1.shape)
    # print('boxes2: ', boxes2.shape)
    num_box1 = boxes1.shape[0]
    num_box2 = boxes2.shape[0]
    lt = np.maximum(boxes1.reshape([num_box1, 1, -1])[:, :, :2],
                    boxes2.reshape([1, num_box2, -1])[:, :, :2])  # [N,M,2]
    rb = np.minimum(boxes1.reshape([num_box1, 1, -1])[:, :, 2:],
                    boxes2.reshape([1, num_box2, -1])[:, :, 2:])  # [N,M,2]

    wh = (rb - lt + to_move).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    return inter


def register_vg_instances(dataset_name, image_root, dataset_split, mapping_dictionary_json_path=None, scenegraph_dataset_h5_path=None, dataset_image_data_json_path=None, scenegraph_dictionary_cache_file=None, use_basename_for_file_paths=False):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(dataset_name, str), dataset_name
    assert scenegraph_dictionary_cache_file is not None or (scenegraph_dataset_h5_path is not None and dataset_image_data_json_path is not None)
    assert not (scenegraph_dictionary_cache_file is not None and (scenegraph_dataset_h5_path is not None or dataset_image_data_json_path is not None))
    if scenegraph_dictionary_cache_file is not None:
        assert isinstance(scenegraph_dictionary_cache_file, (str, os.PathLike)), scenegraph_dictionary_cache_file
    else:
        assert isinstance(scenegraph_dataset_h5_path, (str, os.PathLike)), scenegraph_dataset_h5_path
        assert isinstance(dataset_image_data_json_path, (str, os.PathLike)), dataset_image_data_json_path

    if mapping_dictionary_json_path is not None:
        assert isinstance(mapping_dictionary_json_path,
                          (str, os.PathLike)), mapping_dictionary_json_path
    #metadata is generated from mapping dictionary. scenegraph h5 file and image data json generate the cached pickle file

    assert isinstance(image_root, (str, os.PathLike)), image_root
#    # 1. register a function which returns dicts
#    # dataset_dicts = self._load_graphs()

    #NOTE: creating this dataset instance will register it and add metadata
    sgg_dataset = SggDataset(cfg=None, dataset_name=dataset_name, dataset_image_dir=image_root, mapping_dictionary_json_path=mapping_dictionary_json_path, scenegraph_dataset_h5_path=scenegraph_dataset_h5_path, scenegraph_dictionary_cache_file=scenegraph_dictionary_cache_file, dataset_image_data_json_path=dataset_image_data_json_path, dataset_split=dataset_split, use_basename_for_file_paths=use_basename_for_file_paths)
    return sgg_dataset
#    # 2. Optionally, add metadata about this dataset,
#    # since they might be useful in evaluation, visualization or logging
#    MetadataCatalog.get(name).set(
#        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
#    )

