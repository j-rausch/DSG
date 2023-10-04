import os
import copy
import numpy as np
import torch
from fvcore.common.file_io import PathManager
from PIL import Image

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures.instances import Instances
from detectron2.data import DatasetCatalog, MetadataCatalog, MapDataset, DatasetFromList, DatasetMapper
from collections import defaultdict
from imantics import Polygons, Mask

class SceneGraphDatasetMapper(DatasetMapper):
    def __init__(self, cfg, is_train=True):
        super(SceneGraphDatasetMapper, self).__init__(cfg, is_train=is_train)
        self.is_train=is_train
        #TODO: make this flexible for other non-VG datasets
        self.filter_duplicate_relations = cfg.DATASETS.VISUAL_GENOME.FILTER_DUPLICATE_RELATIONS
    
    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        h, w, _ = image.shape
        if w != dataset_dict['width'] or h != dataset_dict['height']:
            dataset_dict['width'] = w
            dataset_dict['height'] = h
        utils.check_image_size(dataset_dict, image)
        
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None
        
        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg
        image_shape = image.shape[:2]  # h, w

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            #TODO: this also has to account for potential instances being filtered out and potential resulting mismatches
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        # if not self.is_train:
        #     dataset_dict.pop("annotations", None)
        #     dataset_dict.pop("sem_seg_file_name", None)
        #     return dataset_dict
        
        #TODO: debug this. perhaps is_crowd or other parameters influence the tabular/table accuracy drop?
        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )
            

            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"], filter_mask = filter_empty_instances(instances)
            filter_mask_list = filter_mask.to(device='cpu').numpy().tolist()
            mapping_after_filtering = dict()
            new_proposal_id = 0
            for original_proposal_id, is_valid_proposal in enumerate(filter_mask_list):
                if is_valid_proposal:
                    mapping_after_filtering[original_proposal_id] = new_proposal_id
                    new_proposal_id += 1

            # Filter duplicate relations
            rel_present = False
            if "relations" in dataset_dict:
                if self.filter_duplicate_relations and self.is_train:
                    relation_dict = defaultdict(list)
                    for object_0, object_1, relation in dataset_dict["relations"]:
                        relation_dict[(object_0, object_1)].append(relation)
                    #dataset_dict["relations"] = [(k[0], k[1], np.random.choice(v)) for k, v in relation_dict.items()]
                    filtered_and_remapped_relations = []
                    for k, v in relation_dict.items():
                        #new_rel_candidate = (k[0], k[1], np.random.choice(v))
                        if filter_mask_list[k[0]] is True and filter_mask_list [k[1]] is True: #0,4,1) == new_rel:
                            orig_head_id, orig_tail_id = k[0], k[1]
                            new_rel = (mapping_after_filtering[orig_head_id], mapping_after_filtering[orig_tail_id], np.random.choice(v))
                            filtered_and_remapped_relations.append(new_rel)

                        #if filter_mask_list[k[0]] is False or filter_mask_list [k[1]] is False: #0,4,1) == new_rel:
                        #    #debug
                        #    print("Relations became invalid due to instance removal in {}. skipping..".format(
                        #        dataset_dict['file_name']))
                        #    return None

                    if len(filtered_and_remapped_relations) == 0:
                        print("WARNING: NO relations found for img: {}. skipping image..".format(dataset_dict['file_name']))
                        return None
                    else:
                        rel_present = True

                    dataset_dict["relations"] = filtered_and_remapped_relations

                dataset_dict["relations"] = torch.as_tensor(np.ascontiguousarray(dataset_dict["relations"]))


            if rel_present:
                # Add object attributes
                instances.gt_attributes = torch.tensor([obj['attribute'] for obj in annos], dtype=torch.int64)
        return dataset_dict

class MaskLabelDatasetMapper(SceneGraphDatasetMapper):
    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        h, w, _ = image.shape
        if w != dataset_dict['width'] or h != dataset_dict['height']:
            dataset_dict['width'] = w
            dataset_dict['height'] = h
        utils.check_image_size(dataset_dict, image)
        
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None
        
        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg
        image_shape = image.shape[:2]  # h, w

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))
        
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )
        
        dataset_dict["relations"] = torch.as_tensor(np.ascontiguousarray(dataset_dict["relations"]))
        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )
            
            # Add object attributes
            instances.gt_attributes = torch.tensor([obj['attribute'] for obj in annos], dtype=torch.int64)
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["empty_index"] = self.nonempty(instances.gt_boxes.tensor.clone()).data.cpu().numpy()
            #NOTE: it is possible that this filters out instances that are referred to in the gt relations.
            #TODO: either also filter gt relations, or don't remove empty instances
            dataset_dict["instances"], filter_mask = filter_empty_instances(instances)
        return dataset_dict
    
    def nonempty(self, box, threshold=1e-5):
        widths = box[:, 2] - box[:, 0]
        heights = box[:, 3] - box[:, 1]
        keep = (widths > threshold) & (heights > threshold)
        return keep

class ObjectDetectionDatasetMapper(SceneGraphDatasetMapper):
    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        input_dict = copy.deepcopy(dataset_dict)
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        h, w, _ = image.shape
        if w != dataset_dict['width'] or h != dataset_dict['height']:
            dataset_dict['width'] = w
            dataset_dict['height'] = h
        utils.check_image_size(dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]

            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            # dataset_dict["instances"] = utils.filter_empty_instances(instances)
            dataset_dict["instances"], masks = filter_empty_instances(instances)
        return dataset_dict

class MaskRCNNDatasetMapper(DatasetMapper):
    def __init__(self, cfg, is_train=True):
        super(MaskRCNNDatasetMapper, self).__init__(cfg, is_train=is_train)
        self.is_train=is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)
        
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None
        
        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg
        image_shape = image.shape[:2]  # h, w

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        if self.is_train:
            for idx, annotation in enumerate(dataset_dict['annotations']):
                annotation['bbox'] = annotation['bbox'].tolist()
                annotation['segmentation'] = [x.tolist() for x in annotation['segmentation']]

        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict
 
        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict

def filter_empty_instances(instances, by_box=True, by_mask=True, box_threshold=1e-5):
    """
    Filter out empty instances in an `Instances` object.
    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks
        box_threshold (float): minimum width and height to be considered non-empty
    Returns:
        Instances: the filtered instances.
    """
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty(threshold=box_threshold))
    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.nonempty())

    # TODO: can also filter visible keypoints

    if not r:
        return instances
    m = r[0]
    # for x in r[1:]:
    #     m = m & x
    return instances[m], m

