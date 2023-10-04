import copy
import inspect
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import ROI_HEADS_REGISTRY
from detectron2.modeling.roi_heads import StandardROIHeads
from detectron2.modeling.poolers import ROIPooler
#from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from segmentationsg.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from detectron2.layers import Linear, ShapeSpec, batched_nms, cat, nonzero_tuple
from detectron2.modeling.sampling import subsample_labels
from detectron2.utils.events import get_event_storage
from torch import nn
from detectron2.modeling.roi_heads import select_foreground_proposals
from segmentationsg.modeling.matcher import FairMatcher
#from detectron2.modeling.matcher import Matcher
from segmentationsg.modeling.roi_heads.scenegraph_head.doc_heuristics import StructureParser
from segmentationsg.modeling.roi_heads.scenegraph_head.sampling import build_roi_scenegraph_samp_processor

from .fast_rcnn import FastRCNNOutputLayersSG, FastRCNNOutputLayerswithCOCO, FastRCNNOutputLayersSGMaskTransfer, FastRCNNOutputLayersSGEnd2End
from .scenegraph_head import build_scenegraph_head
from .fast_rcnn import fast_rcnn_inference

from segmentationsg.utils.postprocessing import postprocess_raw_tensor,postprocess_prediction_instances

#add tweaked version of StandardROIHeads that randomly selects the GT match for a proposal, if there are multiple ones that are equally good fits
#this can occur, if in the ground truth data multiple bboxes are exactly the same (e.g. two bboxes "human" and "woman" enclosing a person)
@ROI_HEADS_REGISTRY.register()
class StandardROIHeadsRandomTieBreaking(StandardROIHeads):

#    @configurable
#    def __init__(
#            self,
#            *,
#            box_in_features: List[str],
#            box_pooler: ROIPooler,
#            box_head: nn.Module,
#            box_predictor: nn.Module,
#            mask_in_features: Optional[List[str]] = None,
#            mask_pooler: Optional[ROIPooler] = None,
#            mask_head: Optional[nn.Module] = None,
#            keypoint_in_features: Optional[List[str]] = None,
#            keypoint_pooler: Optional[ROIPooler] = None,
#            keypoint_head: Optional[nn.Module] = None,
#            train_on_pred_boxes: bool = False,
#            **kwargs
#    ):
#        super(StandardROIHeadsRandomTieBreaking, self).__init__(box_in_features=box_in_features, box_pooler=box_pooler, box_head=box_head, box_predictor=box_predictor,
#                                                 mask_in_features=mask_in_features, mask_pooler=mask_pooler, mask_head=mask_head, keypoint_in_features=keypoint_in_features,
#                                                 keypoint_pooler=keypoint_pooler, keypoint_head=keypoint_head, train_on_pred_boxes=train_on_pred_boxes, **kwargs)


    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)

#        #overwrite default matcher
        ret["proposal_matcher"] = FairMatcher(
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
            cfg.MODEL.ROI_HEADS.IOU_LABELS,
            allow_low_quality_matches=False,
        )
        return ret

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        if self.training:
            proposals, losses = super().forward(images, features, proposals, targets)
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            #pred_instances, _ = super().forward(images, features, proposals, targets)
            return pred_instances, {}


@ROI_HEADS_REGISTRY.register()
class StandardSGROIHeads(StandardROIHeadsRandomTieBreaking):
    @configurable
    def __init__(self,*, box_in_features, box_pooler, box_head, box_predictor, 
                    mask_in_features=None, mask_pooler=None, mask_head=None, 
                    keypoint_in_features=None, keypoint_pooler=None, keypoint_head=None,
                    scenegraph_in_features=None, scenegraph_head=None, 
                    train_on_pred_boxes=False, use_gt_box=True, use_gt_object_label=True, add_gt=True, freeze_layers=[], use_document_heuristics_for_relations_in_test=False, document_heuristics_dataset_type='ADtgt', use_document_heuristics_with_postprocessing_loop=False, use_grammar_based_postprocessing=False,class_mapping_list=None, **kwargs):
        super(StandardSGROIHeads, self).__init__(box_in_features=box_in_features, box_pooler=box_pooler, box_head=box_head, box_predictor=box_predictor,
                                                    mask_in_features=mask_in_features, mask_pooler=mask_pooler, mask_head=mask_head, keypoint_in_features=keypoint_in_features, 
                                                    keypoint_pooler=keypoint_pooler, keypoint_head=keypoint_head, train_on_pred_boxes=train_on_pred_boxes, **kwargs)
        self.use_gt_box = use_gt_box
        self.use_gt_object_label = use_gt_object_label
        self.add_gt = add_gt
        self.scenegraph_on = scenegraph_in_features is not None
        if self.scenegraph_on:
            self.scenegraph_in_features = scenegraph_in_features
            # self.scenegraph_pooler = scenegraph_pooler
            self.scenegraph_head = scenegraph_head
        self._freeze_layers(layers=freeze_layers)
        self.use_document_heuristics_for_relations_in_test = use_document_heuristics_for_relations_in_test
        self.use_document_heuristics_with_postprocessing_loop = use_document_heuristics_with_postprocessing_loop
        self.document_heuristics_dataset_type = document_heuristics_dataset_type
        if self.use_document_heuristics_for_relations_in_test is True:
            self.legacy_structure_parser = StructureParser(dataset_type = self.document_heuristics_dataset_type)
        self.use_grammar_based_postprocessing = use_grammar_based_postprocessing
        self.class_mapping_list = class_mapping_list
        assert not (self.use_grammar_based_postprocessing is True and self.use_document_heuristics_for_relations_in_test is True)

    def _freeze_layers(self, layers):
        # Freeze layers
        for name, param in self.named_parameters():
            if any(layer == name.split(".")[0] for layer in layers):
                logging.getLogger('detectron2').log(logging.WARN, "Freezed Layer: {}".format(name))
                param.requires_grad = False

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        if inspect.ismethod(cls._init_scene_graph_head):
            ret.update(cls._init_scene_graph_head(cfg, input_shape))
        ret['use_gt_box'] = cfg.MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_BOX
        ret['use_gt_object_label'] = cfg.MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_OBJECT_LABEL
        ret['freeze_layers'] = cfg.MODEL.FREEZE_LAYERS.ROI_HEADS
        ret['add_gt'] = cfg.MODEL.ROI_SCENEGRAPH_HEAD.ADD_GTBOX_TO_PROPOSAL_IN_TRAIN
        ret['use_document_heuristics_for_relations_in_test'] = cfg.TEST.USE_DOCUMENT_HEURISTICS
        ret['use_document_heuristics_with_postprocessing_loop'] = cfg.TEST.DOCUMENT_HEURISTICS_ACTIVATE_POSTPROCESSING_LOOP
        ret['document_heuristics_dataset_type'] = cfg.TEST.USE_DOCUMENT_HEURISTICS_DATASET_TYPE
        ret['use_grammar_based_postprocessing'] = cfg.TEST.USE_GRAMMAR_POSTPROCESSING
        
        class_mapping_list = MetadataCatalog.get(cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused").thing_classes
        ret['class_mapping_list'] = class_mapping_list
        
        assert cfg.TEST.USE_DOCUMENT_HEURISTICS_DATASET_TYPE in ['ADtgt', 'EP']
        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        ret = super()._init_box_head(cfg, input_shape)
        del ret['box_predictor']
        ret['box_predictor'] = FastRCNNOutputLayersSG(cfg, ret['box_head'].output_shape)
        return ret

    @classmethod
    def _init_scene_graph_head(cls, cfg, input_shape):
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        
        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"scenegraph_in_features": in_features}
        shape = input_shape
        ret["scenegraph_head"] = build_scenegraph_head(cfg, shape)

        return ret

    def forward(self, images, features, proposals, targets=None, relations=None):
        del images
        
        with torch.no_grad():
            pred_instances = self._forward_box(features, proposals, targets=targets)
            pred_instances = self._forward_mask(features, pred_instances)
        
        if self.training:
            _, _, losses = self._forward_scenegraph(features, pred_instances, targets, relations)
            return proposals, losses
        else:    
            # # During inference cascaded prediction is used: the mask and keypoints heads are only
            # # applied to the top scoring box detections.
            if self.use_document_heuristics_for_relations_in_test is False:
                _, result, _ = self._forward_scenegraph(features, pred_instances, targets, relations)
            else:
                boxes = [x.pred_boxes for x in pred_instances]
                rel_labels, rel_binarys = None, None
                rel_pair_idxs = self.scenegraph_head.drop_bad_labels.prepare_test_pairs(boxes[0].device, pred_instances)
                result  = self.legacy_structure_parser.forward_document_heuristics_from_detectron2(features=features, instances=pred_instances, targets=targets, relations=relations, rel_pair_idxs=rel_pair_idxs, do_postprocessing=self.use_document_heuristics_with_postprocessing_loop)
                #result = forward_document_heuristics(features=features, instances=pred_instances, targets=targets, relations=relations)
                
            if  self.use_grammar_based_postprocessing is True:
                result_postprocessed = []
                for res in result:
                    
                    tensor_before, tensor_after = postprocess_prediction_instances(res, self.class_mapping_list)
                    #res_postprocessed = postprocess_prediction_with_grammar(res) 
                    result_postprocessed.append(tensor_after) 
                result=result_postprocessed
            return result, {}
            
        del targets
    
    def _forward_mask(self, features, instances):
        if not self.mask_on:
            return instances
        if self.use_gt_box and (not self.use_gt_object_label) and self.training:
            for idx, instance in enumerate(instances):
                mask_pred_classes = copy.deepcopy(instance.mask_pred_classes)
                pred_classes = copy.deepcopy(instance.pred_classes)
                instances[idx].mask_pred_classes = pred_classes
                instances[idx].pred_classes = mask_pred_classes
        if self.mask_pooler is not None:
            features = [features[f] for f in self.mask_in_features]
            boxes = [x.pred_boxes for x in instances]
            features = self.mask_pooler(features, boxes)
        else:
            features = {f: features[f] for f in self.mask_in_features}
        instances = self.mask_head(features, instances)
        if self.use_gt_box and (not self.use_gt_object_label) and self.training:
            for idx, instance in enumerate(instances):
                mask_pred_classes = copy.deepcopy(instance.mask_pred_classes)
                pred_classes = copy.deepcopy(instance.pred_classes)
                instances[idx].mask_pred_classes = pred_classes
                instances[idx].pred_classes = mask_pred_classes
        return instances

    def _forward_box(self, features, proposals, targets=None):

        features = [features[f] for f in self.box_in_features]
        if self.use_gt_box:
            del proposals
            proposals = []
            for idx, target in enumerate(targets):
                instance = Instances(target.image_size)
                instance.proposal_boxes = Boxes(target.gt_boxes.tensor.detach().clone())
                proposals.append(instance)
        else:
            if self.training:
                assert targets, "'targets' argument is required during training"
                #Sgdet add gt boxes
                #FIX ME: Add gt boxes here??
                #if self.add_gt:
                #     proposals = self.add_gt_proposals(proposals, targets)
                proposals = self.label_and_sample_proposals(proposals, targets)
                #if self.add_gt:
                #    #gt_boxes = [x.gt_boxes for x in targets]
                #    proposals = self.add_gt_proposals(proposals, targets)
                
        if not self.use_gt_object_label:
            box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
            box_features = self.box_head(box_features)
            predictions = self.box_predictor(box_features)
            del box_features
        else:
            predictions = None
        pred_instances, _ = self.box_predictor.inference(proposals, predictions=predictions, targets=targets)
        return pred_instances
    
    def add_gt_proposals(self, proposals, targets):
        """
        Arguments:
            proposals: list[BoxList]
            targets: list[BoxList]
        """
        # Get the device we're operating on
        device = proposals[0].proposal_boxes.device

        gt_boxes = [Instances(image_size=target._image_size, proposal_boxes=target.gt_boxes.clone()) for target in targets]
        
        # later cat of bbox requires all fields to be present for all bbox
        # so we need to add a dummy for objectness that's missing
        for gt_box in gt_boxes:
            gt_box.objectness_logits =  torch.full((len(gt_box),), fill_value=100, dtype=proposals[0].objectness_logits.dtype, device=device) #A high value for logits

        proposals = [
            Instances.cat([proposal, gt_box])
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        return proposals
    def _forward_scenegraph(
        self, 
        features: Dict[str, torch.Tensor], 
        instances: List[Instances],
        targets = None, 
        relations=None
        ):
        """
        Forward logic of the mask prediction branch.
        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.
        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.scenegraph_on:
            return {} if self.training else instances

        # if self.scenegraph_pooler is not None:
        #     features = [features[f] for f in self.scenegraph_in_features]
        #     boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
        #     features = self.scenegraph_pooler(features, boxes)
        # else:
        #     features = {f: features[f] for f in self.scenegraph_in_features}

        return self.scenegraph_head(features, instances, targets, relations)


@ROI_HEADS_REGISTRY.register()
class StandardSGROIHeadsEnd2End(StandardROIHeadsRandomTieBreaking):
    @configurable
    def __init__(self, *, box_in_features, box_pooler, box_head, box_predictor,
                 mask_in_features=None, mask_pooler=None, mask_head=None,
                 keypoint_in_features=None, keypoint_pooler=None, keypoint_head=None,
                 scenegraph_in_features=None, scenegraph_head=None,
                 train_on_pred_boxes=False, use_gt_box=True, use_gt_object_label=True, add_gt=True, add_gt_instances_to_inference_predictions_in_train=True, freeze_layers=[], proposal_matcher=None, use_document_heuristics_for_relations_in_test=False, document_heuristics_dataset_type='ADtgt', use_document_heuristics_with_postprocessing_loop=False, use_grammar_based_postprocessing=False,class_mapping_list=None, **kwargs):
        assert freeze_layers == []
        super(StandardSGROIHeadsEnd2End, self).__init__(box_in_features=box_in_features, box_pooler=box_pooler,
                                                        box_head=box_head, box_predictor=box_predictor,
                                                        mask_in_features=mask_in_features, mask_pooler=mask_pooler,
                                                        mask_head=mask_head, keypoint_in_features=keypoint_in_features,
                                                        keypoint_pooler=keypoint_pooler, keypoint_head=keypoint_head,
                                                        train_on_pred_boxes=train_on_pred_boxes, proposal_matcher=proposal_matcher, **kwargs)
        self.use_gt_box = use_gt_box
        self.use_gt_object_label = use_gt_object_label
        #        assert self.use_gt_box is False
        #        assert self.use_gt_object_label is False
        #self.proposal_append_gt = add_gt
        #self.add_gt = add_gt
        self.add_gt_instances_to_inference_predictions_in_train = add_gt_instances_to_inference_predictions_in_train
        self.scenegraph_on = scenegraph_in_features is not None
        if self.scenegraph_on:
            self.scenegraph_in_features = scenegraph_in_features
            # self.scenegraph_pooler = scenegraph_pooler
            self.scenegraph_head = scenegraph_head
        #self._freeze_layers(layers=freeze_layers)
        if proposal_matcher is not None:
            self.proposal_matcher = proposal_matcher
        self.use_document_heuristics_for_relations_in_test = use_document_heuristics_for_relations_in_test
        self.use_document_heuristics_with_postprocessing_loop = use_document_heuristics_with_postprocessing_loop
        self.document_heuristics_dataset_type = document_heuristics_dataset_type
        if self.use_document_heuristics_for_relations_in_test is True:
            self.legacy_structure_parser = StructureParser(dataset_type = self.document_heuristics_dataset_type)
        self.use_grammar_based_postprocessing = use_grammar_based_postprocessing
        self.class_mapping_list = class_mapping_list
        assert not (self.use_grammar_based_postprocessing is True and self.use_document_heuristics_for_relations_in_test is True)
        
    def _freeze_layers(self, layers):
        # Freeze layers
        for name, param in self.named_parameters():
            if any(layer == name.split(".")[0] for layer in layers):
                logging.getLogger('detectron2').log(logging.WARN, "Freezed Layer: {}".format(name))
                param.requires_grad = False

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        if inspect.ismethod(cls._init_scene_graph_head):
            ret.update(cls._init_scene_graph_head(cfg, input_shape))
        ret['use_gt_box'] = cfg.MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_BOX
        ret['use_gt_object_label'] = cfg.MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_OBJECT_LABEL
        ret['freeze_layers'] = cfg.MODEL.FREEZE_LAYERS.ROI_HEADS
        #TODO: double-check and remove 'add_gt'. unused
        ret['add_gt'] = cfg.MODEL.ROI_SCENEGRAPH_HEAD.ADD_GTBOX_TO_PROPOSAL_IN_TRAIN
        ret['add_gt_instances_to_inference_predictions_in_train'] = cfg.MODEL.ROI_SCENEGRAPH_HEAD.ADD_GT_INSTANCES_TO_DETECTOR_TRAIN_PREDICTIONS
        ret['use_document_heuristics_for_relations_in_test'] = cfg.TEST.USE_DOCUMENT_HEURISTICS
        ret['use_document_heuristics_with_postprocessing_loop'] = cfg.TEST.DOCUMENT_HEURISTICS_ACTIVATE_POSTPROCESSING_LOOP
        ret['document_heuristics_dataset_type'] = cfg.TEST.USE_DOCUMENT_HEURISTICS_DATASET_TYPE
        ret['use_grammar_based_postprocessing'] = cfg.TEST.USE_GRAMMAR_POSTPROCESSING
        class_mapping_list = MetadataCatalog.get(cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused").thing_classes
        ret['class_mapping_list'] = class_mapping_list

        #        ret["proposal_matcher"] = Matcher(
#            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
#            cfg.MODEL.ROI_HEADS.IOU_LABELS,
#            allow_low_quality_matches=False,
#        )
        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        ret = super()._init_box_head(cfg, input_shape)
        del ret['box_predictor']
        ret['box_predictor'] = FastRCNNOutputLayersSGEnd2End(cfg, ret['box_head'].output_shape)
        return ret

    @classmethod
    def _init_scene_graph_head(cls, cfg, input_shape):
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"scenegraph_in_features": in_features}
        shape = input_shape
        ret["scenegraph_head"] = build_scenegraph_head(cfg, shape)

        return ret

    def forward(self, images, features, proposals, targets=None, relations=None):
        del images

        if self.training is False:
            with torch.no_grad():
                # TODO: check whether losses need to be returned here (and ignored)
                pred_instances = self._forward_box(features, proposals, targets=targets)
                if isinstance(pred_instances[0], list) and len(pred_instances) > 1 and all(len(x) == 0 for x in pred_instances[1:]):
                    pred_instances = pred_instances[0]

                # TODO: same here as above with losses (adapted it here)
                pred_instances, _ = self._forward_mask(features, pred_instances)
            # # During inference cascaded prediction is used: the mask and keypoints heads are only
            # # applied to the top scoring box detections.
            #TODO: we set a score threshold for instances. however, the scenegraph head assigns new labels and scores, where this threshold is not applied anymore

            if self.use_document_heuristics_for_relations_in_test is False:
                _, result, _ = self._forward_scenegraph(features=features, instances=pred_instances, targets=targets, relations=relations)
            else:

                boxes = [x.pred_boxes for x in pred_instances]
                #rel_labels, rel_binarys = None, None
                rel_pair_idxs = self.scenegraph_head.samp_processor.prepare_test_pairs(boxes[0].device, pred_instances)
                result = self.legacy_structure_parser.forward_document_heuristics_from_detectron2(
                    features=features, instances=pred_instances, targets=targets,
                    relations=relations, rel_pair_idxs=rel_pair_idxs, do_postprocessing=self.use_document_heuristics_with_postprocessing_loop)

            if  self.use_grammar_based_postprocessing is True:
                result_postprocessed = []
                for res in result:
                    
                    tensor_after = postprocess_prediction_instances(res, self.class_mapping_list)
                    #res_postprocessed = postprocess_prediction_with_grammar(res) 
                    result_postprocessed.append(tensor_after) 
                result=result_postprocessed
            return result, {}
        elif self.training is True:
            #NOTE: we now also add GT proposals to the proposals to improve relation classification (in label_an_sample_proposals)
            proposals = self.label_and_sample_proposals(proposals, targets)

#            if self.add_gt:
#                proposals = self.add_gt_proposals(proposals, targets)

            pred_instances, losses = self._forward_box(features, proposals, targets=targets)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            pred_instances, losses_mask = self._forward_mask(features, pred_instances)
            if losses_mask is not None:
                losses.update(losses_mask)
            # losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            #_, _, losses_sg = self._forward_scenegraph(features, proposals, targets, relations)
            #pred_instances = self._add_gt_instances_for_rel_head(pred_instances, targets)
            _, _, losses_sg = self._forward_scenegraph(features=features, instances=pred_instances, targets=targets, relations=relations)
            if losses_sg is not None:
                losses.update(losses_sg)
            return proposals, losses
        del targets


    def _forward_mask(self, features, instances):
        if not self.mask_on:
            return instances, {}

        if self.use_gt_box and (not self.use_gt_object_label) and self.training:
            for idx, instance in enumerate(instances):
                mask_pred_classes = copy.deepcopy(instance.mask_pred_classes)
                pred_classes = copy.deepcopy(instance.pred_classes)
                instances[idx].mask_pred_classes = pred_classes
                instances[idx].pred_classes = mask_pred_classes
        if self.mask_pooler is not None:
            features = [features[f] for f in self.mask_in_features]
            boxes = [x.pred_boxes for x in instances]
            features = self.mask_pooler(features, boxes)
        else:
            features = {f: features[f] for f in self.mask_in_features}

        if self.training:
            instances, losses = self.mask_head(features, instances)
        else:
            instances, _ = self.mask_head(features, instances)

        if self.use_gt_box and (not self.use_gt_object_label) and self.training:
            for idx, instance in enumerate(instances):
                mask_pred_classes = copy.deepcopy(instance.mask_pred_classes)
                pred_classes = copy.deepcopy(instance.pred_classes)
                instances[idx].mask_pred_classes = pred_classes
                instances[idx].pred_classes = mask_pred_classes

        if self.training:
            return instances, losses
        else:
            return instances, {}

    def _forward_box(self, features, proposals, targets=None):

        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)
            #NOTE: even GT proposals sometimes dismissed here, since the decision ultimately depends on the predictor score, not just the proposal box
            #TODO: can we do this differentiable?
            pred_instances, _ = self.box_predictor.inference(proposals, predictions=predictions, targets=targets, add_gt_instances_to_inference_predictions_in_train=self.add_gt_instances_to_inference_predictions_in_train)
            return pred_instances, losses
        else:
            pred_instances, _ = self.box_predictor.inference(proposals, predictions=predictions)
            return pred_instances, {}

    #        if not self.use_gt_object_label:
    #            box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
    #        pred_instances, losses = self.box_predictor.inference(proposals, predictions=predictions, targets=targets)
    #        return pred_instances, losses

    def add_gt_proposals(self, proposals, targets):
        """
        Arguments:
            proposals: list[BoxList]
            targets: list[BoxList]
        """
        # Get the device we're operating on
        device = proposals[0].proposal_boxes.device

        gt_boxes = [Instances(image_size=target._image_size, proposal_boxes=target.gt_boxes.clone()) for target in
                    targets]

        # later cat of bbox requires all fields to be present for all bbox
        # so we need to add a dummy for objectness that's missing
        for gt_box in gt_boxes:
            gt_box.objectness_logits = torch.full((len(gt_box),), fill_value=100,
                                                  dtype=proposals[0].objectness_logits.dtype,
                                                  device=device)  # A high value for logits

        proposals = [
            Instances.cat([proposal, gt_box])
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        return proposals


    def _forward_scenegraph(
            self,
            features: Dict[str, torch.Tensor],
            instances: List[Instances],
            targets=None,
            relations=None
    ):
        """
        Forward logic of the mask prediction branch.
        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.
        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.scenegraph_on:
            return {} if self.training else instances

        # if self.scenegraph_pooler is not None:
        #     features = [features[f] for f in self.scenegraph_in_features]
        #     boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
        #     features = self.scenegraph_pooler(features, boxes)
        # else:
        #     features = {f: features[f] for f in self.scenegraph_in_features}

        return self.scenegraph_head(features=features, proposals=instances, targets=targets, relations=relations)



@ROI_HEADS_REGISTRY.register()
class SGROIHeadsMaskTransfer(StandardROIHeadsRandomTieBreaking):
    @configurable
    def __init__(self,*, box_in_features, box_pooler, box_head, box_predictor, 
                    mask_in_features=None, mask_pooler=None, mask_head=None, 
                    keypoint_in_features=None, keypoint_pooler=None, keypoint_head=None,
                    scenegraph_in_features=None, scenegraph_head=None, 
                    train_on_pred_boxes=False, use_gt_box=True, use_gt_object_label=True, freeze_layers=[], **kwargs):
        self.train_data_name = kwargs['train_data_name']
        self.embeddings_path = kwargs['embeddings_path']
        self.embeddings_path_coco = kwargs['embeddings_path_coco']
        self.num_output_classes = kwargs['num_output_classes']
        self.transfer_data_name = kwargs['transfer_data_name']
        self.lingual_matrix_threshold = kwargs['lingual_matrix_threshold']
        del kwargs['transfer_data_name']
        del kwargs['train_data_name']
        del kwargs['embeddings_path']
        del kwargs['embeddings_path_coco']
        del kwargs['num_output_classes']
        del kwargs['lingual_matrix_threshold']
        super(SGROIHeadsMaskTransfer, self).__init__(box_in_features=box_in_features, box_pooler=box_pooler, box_head=box_head, box_predictor=box_predictor, 
                                                    mask_in_features=mask_in_features, mask_pooler=mask_pooler, mask_head=mask_head, keypoint_in_features=keypoint_in_features, 
                                                    keypoint_pooler=keypoint_pooler, keypoint_head=keypoint_head, train_on_pred_boxes=train_on_pred_boxes, **kwargs)
        self.use_gt_box = use_gt_box
        self.use_gt_object_label = use_gt_object_label
        self.scenegraph_on = scenegraph_in_features is not None
        if self.scenegraph_on:
            self.scenegraph_in_features = scenegraph_in_features
            # self.scenegraph_pooler = scenegraph_pooler
            self.scenegraph_head = scenegraph_head
        pretrained_embeddings = torch.load(self.embeddings_path)['embeddings']
        pretrained_embeddings_coco = torch.load(self.embeddings_path_coco)['embeddings']
        self.embeddings = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=True)
        self.embeddings_coco = nn.Embedding.from_pretrained(pretrained_embeddings_coco, freeze=True)
        self._class_mapper()
        self._freeze_layers(layers=freeze_layers)

    def _freeze_layers(self, layers):
        # Freeze layers
        for name, param in self.named_parameters():
            if any(layer == name.split(".")[0] for layer in layers):
                logging.getLogger('detectron2').log(logging.WARN, "Freezed Layer: {}".format(name))
                param.requires_grad = False

    def _class_mapper(self):
        self.coco_classes = {name.lower():idx for idx, name in enumerate(MetadataCatalog.get(self.transfer_data_name).thing_classes)}
        self.output_classes = {name.lower():idx for idx, name in enumerate(MetadataCatalog.get(self.train_data_name).thing_classes)}
        self.base_classes_indexer = []
        self.novel_classes_indexer = []
        self.output_to_coco_indexer = []
        if 'OI' in self.transfer_data_name:
            transfers = {'bike':'motorcycle', 'phone': 'mobile phone', 'arm': 'human_arm', 'basket': 'picnic basket', 'counter': 'countertop', 'ear': 'human ear', 'cup': 'coffee cup', 'eye': 'human eye', 'face': 'human face', 'guy': 'man', 'hair': 'human hair', 'hand': 'human hand', 'handle': 'door handle', 'head': 'human head', 'jean': 'jeans', 'lady': 'woman', 'leg': 'human leg', 'men': 'man', 'mouth': 'human mouth', 'nose': 'human nose', 'plane': 'airplane', 'pot': 'flowerpot', 'short':'shorts'}
            non_zero_masks = ['tortoise', 'magpie', 'sea turtle', 'football', 'ambulance', 'toy', 'apple', 'beer', 'chopsticks', 'bird', 'traffic light', 'croissant', 'cucumber', 'radish', 'towel', 'skull', 'washing machine', 'glove', 'belt', 'ball', 'backpack', 'surfboard', 'boot', 'hot dog', 'shorts', 'bus', 'boy', 'screwdriver', 'bicycle wheel', 'barge', 'laptop', 'miniskirt', 'drill (tool)', 'dress', 'bear', 'waffle', 'pancake', 'brown bear', 'woodpecker', 'blue jay', 'pretzel', 'bagel', 'teapot', 'person', 'swimwear', 'bat (animal)', 'starfish', 'popcorn', 'burrito', 'balloon', 'wrench', 'vehicle registration plate', 'toaster', 'flashlight', 'limousine', 'carnivore', 'scissors', 'computer keyboard', 'printer', 'traffic sign', 'shirt', 'cheese', 'sock', 'fire hydrant', 'tie', 'suitcase', 'muffin', 'snowmobile', 'clock', 'cattle', 'cello', 'jet ski', 'camel', 'suit', 'cat', 'bronze sculpture', 'juice', 'computer mouse', 'cookie', 'coin', 'calculator', 'cocktail', 'box', 'stapler', 'christmas tree', 'cowboy hat', 'studio couch', 'drink', 'zucchini', 'ladle', 'human mouth', 'dice', 'oven', 'couch', 'cricket ball', 'winter melon', 'spatula', 'whiteboard', 'hat', 'shower', 'eraser', 'fedora', 'guacamole', 'dagger', 'scarf', 'dolphin', 'sombrero', 'mug', 'tap', 'harbor seal', 'human body', 'roller skates', 'coffee cup', 'stop sign', 'volleyball (ball)', 'vase', 'slow cooker', 'coffee', 'paper towel', 'sun hat', 'flying disc', 'skirt', 'barrel', 'kite', 'tart', 'fox', 'flag', 'guitar', 'pillow', 'grape', 'human ear', 'power plugs and sockets', 'panda', 'giraffe', 'woman', 'door handle', 'rhinoceros', 'goldfish', 'goat', 'baseball bat', 'baseball glove', 'mixing bowl', 'light switch', 'horse', 'hammer', 'sofa bed', 'adhesive tape', 'saucer', 'harpsichord', 'heater', 'harmonica', 'hamster', 'kettle', 'drinking straw', 'hair dryer', 'food processor', 'punching bag', 'common fig', 'cocktail shaker', 'jaguar (animal)', 'golf ball', 'alarm clock', 'filing cabinet', 'artichoke', 'kangaroo', 'koala', 'knife', 'bottle', 'bottle opener', 'lynx', 'lighthouse', 'dumbbell', 'bowl', 'lizard', 'billiard table', 'mouse', 'motorcycle', 'swim cap', 'frying pan', 'missile', 'bust', 'man', 'milk', 'mobile phone', 'mushroom', 'pitcher (container)', 'table tennis racket', 'pencil case', 'briefcase', 'kitchen knife', 'nail (construction)', 'tennis ball', 'plastic bag', 'chest of drawers', 'ostrich', 'piano', 'girl', 'potato', 'penguin', 'pumpkin', 'pear', 'polar bear', 'pizza', 'digital clock', 'pig', 'reptile', 'lipstick', 'skateboard', 'raven', 'high heels', 'red panda', 'rose', 'rabbit', 'sculpture', 'saxophone', 'submarine sandwich', 'sword', 'picture frame', 'loveseat', 'squirrel', 'segway', 'snake', 'skyscraper', 'sheep', 'tea', 'tank', 'torch', 'tiger', 'strawberry', 'tomato', 'train', 'cooking spray', 'trousers', 'truck', 'measuring cup', 'handbag', 'wine', 'wheel', 'wok', 'whale', 'zebra', 'jug', 'pizza cutter', 'monkey', 'lion', 'bread', 'platter', 'chicken', 'eagle', 'owl', 'duck', 'turtle', 'hippopotamus', 'crocodile', 'toilet', 'toilet paper', 'clothing', 'lemon', 'frog', 'banana', 'rocket', 'tablet computer', 'waste container', 'dog', 'book', 'elephant', 'shark', 'candle', 'leopard', 'axe', 'hand dryer', 'soap dispenser', 'flower', 'canary', 'cheetah', 'hamburger', 'fish', 'garden asparagus', 'hedgehog', 'airplane', 'spoon', 'otter', 'bull', 'oyster', 'orange', 'beaker', 'goose', 'mule', 'swan', 'peach', 'seat belt', 'raccoon', 'chisel', 'camera', 'squash (plant)', 'racket', 'diaper', 'falcon', 'cabbage', 'carrot', 'mango', 'jeans', 'flowerpot', 'envelope', 'cake', 'common sunflower', 'microwave oven', 'sea lion', 'watch', 'parrot', 'handgun', 'sparrow', 'van', 'corded phone', 'tennis racket', 'dog bed', 'facial tissue holder', 'pressure cooker', 'ruler', 'luggage and bags', 'broccoli', 'pastry', 'grapefruit', 'band-aid', 'bell pepper', 'turkey', 'pomegranate', 'doughnut', 'pen', 'car', 'aircraft', 'skunk', 'teddy bear', 'watermelon', 'cantaloupe', 'flute', 'balance beam', 'sandwich', 'binoculars', 'ipod', 'alpaca', 'taxi', 'canoe', 'remote control', 'rugby ball', 'armadillo']
            zero_indices = [self.coco_classes[x] for x in self.coco_classes if x not in non_zero_masks]
        for idx, class_name in enumerate(self.output_classes.keys()):
            if 'coco' in self.transfer_data_name:
                if class_name == 'bike':
                    class_name = 'motorcycle'
                if class_name == 'ski':
                    class_name = 'skis'
                if class_name == 'phone':
                    class_name = 'cell phone'
                if class_name == 'woman' or class_name == 'men' or class_name == 'man' or class_name == 'lady' or class_name == 'girl' or class_name == 'guy' or class_name == 'boy' or class_name == 'child' or class_name == 'girl' or class_name == 'kid' or class_name == 'people' or class_name == 'player' or class_name == 'head' or class_name == 'arm' or class_name == 'face' or class_name == 'jacket' or class_name == 'jean' or class_name == 'leg' or class_name == 'pant' or class_name == 'short':
                    class_name = 'person'
                if class_name == 'table':
                    class_name = 'dining table'
                if class_name == 'plane':
                    class_name = 'airplane'
                if class_name == 'plant' or class_name == 'branch':
                    class_name = 'potted plant'
                if class_name == 'screen':
                    class_name = 'tv'
            else:
                if class_name in transfers:
                    if transfers[class_name] in non_zero_masks:
                        class_name = transfers[class_name]
                if class_name not in non_zero_masks:
                    class_name = class_name + "_nomask"
                    print (class_name)
            if class_name in self.coco_classes:
                self.base_classes_indexer.append(idx)
                self.output_to_coco_indexer.append(self.coco_classes[class_name])
            else:
                print (class_name)
                self.novel_classes_indexer.append(idx)
        self.base_classes_indexer = np.array(self.base_classes_indexer)
        self.novel_classes_indexer = np.array(self.novel_classes_indexer)
        self.output_to_coco_indexer = np.array(self.output_to_coco_indexer)

        self.base_classes_indexer_tensor = torch.tensor(self.base_classes_indexer).long()
        self.novel_classes_indexer_tensor = torch.tensor(self.novel_classes_indexer).long()
        self.output_to_coco_indexer_tensor = torch.tensor(self.output_to_coco_indexer).long()

        self.coco_class_embeddings = self.embeddings_coco(torch.arange(len(self.coco_classes)).long())
        self.output_class_embeddings = self.embeddings(torch.arange(len(self.output_classes)).long())
        self.novel_to_coco_similarity_matrix = torch.mm(torch.index_select(self.output_class_embeddings, 0, self.novel_classes_indexer_tensor), self.coco_class_embeddings.transpose(0,1))
        if 'OI' in self.transfer_data_name:
            self.novel_to_coco_similarity_matrix[:,zero_indices] = -np.inf
            # Bug in training code where classes were set to 600. 
            self.novel_to_coco_similarity_matrix = self.novel_to_coco_similarity_matrix[:,:-1]
        self.novel_to_coco_similarity_matrix = nn.functional.softmax(self.novel_to_coco_similarity_matrix, -1)
        self.novel_to_coco_similarity_matrix[self.novel_to_coco_similarity_matrix < self.lingual_matrix_threshold] = 0.0
        self.novel_to_coco_similarity_matrix = self.novel_to_coco_similarity_matrix / torch.sum(self.novel_to_coco_similarity_matrix, dim=-1, keepdim=True)
        # max_similar = self.novel_to_coco_similarity_matrix.argsort(-1, descending=True)
        # for idx, class_idx in enumerate(self.novel_classes_indexer):
        #     print (MetadataCatalog.get(self.train_data_name).thing_classes[class_idx],":",[MetadataCatalog.get('coco_2017_train').thing_classes[max_similar[idx][x]] for x in range(5)])
        

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        if inspect.ismethod(cls._init_scene_graph_head):
            ret.update(cls._init_scene_graph_head(cfg, input_shape))
        ret['use_gt_box'] = cfg.MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_BOX
        ret['use_gt_object_label'] = cfg.MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_OBJECT_LABEL
        ret['freeze_layers'] = cfg.MODEL.FREEZE_LAYERS.ROI_HEADS
        ret['train_data_name'] = cfg.DATASETS.TRAIN[0]
        ret['embeddings_path'] = cfg.MODEL.ROI_HEADS.EMBEDDINGS_PATH
        ret['embeddings_path_coco'] = cfg.MODEL.ROI_HEADS.EMBEDDINGS_PATH_COCO
        ret['num_output_classes'] = cfg.MODEL.ROI_HEADS.NUM_OUTPUT_CLASSES
        ret['transfer_data_name'] = cfg.DATASETS.TRANSFER[0]
        ret['lingual_matrix_threshold'] = cfg.MODEL.ROI_HEADS.LINGUAL_MATRIX_THRESHOLD
        return ret
    
    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        ret = super()._init_box_head(cfg, input_shape)
        del ret['box_predictor']
        ret['box_predictor'] = FastRCNNOutputLayersSGMaskTransfer(cfg, ret['box_head'].output_shape)
        return ret

    @classmethod
    def _init_scene_graph_head(cls, cfg, input_shape):
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"scenegraph_in_features": in_features}
        shape = input_shape
        ret["scenegraph_head"] = build_scenegraph_head(cfg, shape)

        return ret

    def forward(self, images, features, proposals, targets=None, relations=None):
        if not self.novel_to_coco_similarity_matrix.is_cuda:
            device = next(self.box_predictor.parameters()).device
            self.novel_to_coco_similarity_matrix = self.novel_to_coco_similarity_matrix.to(device)
            self.base_classes_indexer_tensor = self.base_classes_indexer_tensor.to(device)
            self.novel_classes_indexer_tensor = self.novel_classes_indexer_tensor.to(device)
            self.output_to_coco_indexer_tensor = self.output_to_coco_indexer_tensor.to(device)
        del images
        with torch.no_grad():
            pred_instances = self._forward_box(features, proposals, targets=targets)
            pred_instances = self._forward_mask(features, pred_instances)
        if self.training:
            _, _, losses = self._forward_scenegraph(features, pred_instances, targets, relations)
            return proposals, losses
        else:    
            # # During inference cascaded prediction is used: the mask and keypoints heads are only
            # # applied to the top scoring box detections.
            _, result, _ = self._forward_scenegraph(features, pred_instances, targets, relations)
            return result, {}
        
    
    def _forward_mask(self, features, instances):
        if not self.mask_on:
            return instances
        if self.use_gt_box and (not self.use_gt_object_label) and self.training:
            for idx, instance in enumerate(instances):
                mask_pred_classes = copy.deepcopy(instance.mask_pred_classes)
                pred_classes = copy.deepcopy(instance.pred_classes)
                instances[idx].mask_pred_classes = pred_classes
                instances[idx].pred_classes = mask_pred_classes
        if self.mask_pooler is not None:
            features = [features[f] for f in self.mask_in_features]
            boxes = [x.pred_boxes for x in instances]
            features = self.mask_pooler(features, boxes)
        else:
            features = {f: features[f] for f in self.mask_in_features}
        instances = self.mask_head(features, instances, self.novel_to_coco_similarity_matrix, self.base_classes_indexer_tensor, self.novel_classes_indexer_tensor, self.output_to_coco_indexer_tensor)
        if self.use_gt_box and (not self.use_gt_object_label) and self.training:
            for idx, instance in enumerate(instances):
                mask_pred_classes = copy.deepcopy(instance.mask_pred_classes)
                pred_classes = copy.deepcopy(instance.pred_classes)
                instances[idx].mask_pred_classes = pred_classes
                instances[idx].pred_classes = mask_pred_classes
        return instances

    def _forward_box(self, features, proposals, targets=None):
        features = [features[f] for f in self.box_in_features]
        if self.use_gt_box:
            del proposals
            proposals = []
            for idx, target in enumerate(targets):
                instance = Instances(target.image_size)
                instance.proposal_boxes = Boxes(target.gt_boxes.tensor.detach().clone())
                proposals.append(instance)
        else:
            if self.training:
                assert targets, "'targets' argument is required during training"
                proposals = self.label_and_sample_proposals(proposals, targets)
        if not self.use_gt_object_label:
            box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
            box_features = self.box_head(box_features)
            predictions = self.box_predictor(box_features, is_transfer=True)
            del box_features
        else:
            predictions = None
        pred_instances, _ = self.box_predictor.inference(proposals, predictions=predictions, targets=targets)
        return pred_instances

    def _forward_scenegraph(
        self, 
        features: Dict[str, torch.Tensor], 
        instances: List[Instances],
        targets = None, 
        relations=None
        ):
        """
        Forward logic of the mask prediction branch.
        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.
        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.scenegraph_on:
            return {} if self.training else instances

        # if self.scenegraph_pooler is not None:
        #     features = [features[f] for f in self.scenegraph_in_features]
        #     boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
        #     features = self.scenegraph_pooler(features, boxes)
        # else:
        #     features = {f: features[f] for f in self.scenegraph_in_features}

        return self.scenegraph_head(features, instances, targets, relations)

@ROI_HEADS_REGISTRY.register()
class SGSegmentationROIHeadsMaskTransfer(SGROIHeadsMaskTransfer):
    @configurable
    def __init__(self,*, box_in_features, box_pooler, box_head, box_predictor, 
                    mask_in_features=None, mask_pooler=None, mask_head=None, 
                    keypoint_in_features=None, keypoint_pooler=None, keypoint_head=None,
                    scenegraph_in_features=None, scenegraph_head=None, 
                    train_on_pred_boxes=False, use_gt_box=True, use_gt_object_label=True, freeze_layers=[], **kwargs):
        self.refine_seg_mask = kwargs['refine_seg_mask']
        self.mask_num_classes = kwargs['mask_num_classes']
        self.segmentation_step_mask_refine = kwargs['segmentation_step_mask_refine']
        del kwargs['refine_seg_mask']
        del kwargs['mask_num_classes']
        del kwargs['segmentation_step_mask_refine']
        super(SGSegmentationROIHeadsMaskTransfer, self).__init__(box_in_features=box_in_features, box_pooler=box_pooler, box_head=box_head, box_predictor=box_predictor, 
                    mask_in_features=mask_in_features, mask_pooler=mask_pooler, mask_head=mask_head, 
                    keypoint_in_features=keypoint_in_features, keypoint_pooler=keypoint_pooler, keypoint_head=keypoint_head,
                    scenegraph_in_features=scenegraph_in_features, scenegraph_head=scenegraph_head, 
                    train_on_pred_boxes=train_on_pred_boxes, use_gt_box=use_gt_box, use_gt_object_label=use_gt_object_label, freeze_layers=freeze_layers, **kwargs)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret['refine_seg_mask'] = cfg.MODEL.ROI_HEADS.REFINE_SEG_MASKS
        ret['mask_num_classes'] =  cfg.MODEL.ROI_HEADS.MASK_NUM_CLASSES
        ret['segmentation_step_mask_refine'] = cfg.MODEL.ROI_HEADS.SEGMENTATION_STEP_MASK_REFINE
        return ret

    def _sample_proposals(
        self, matched_idxs, matched_labels, gt_classes, segmentation_step=False
    ):
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            if not segmentation_step:
                gt_classes[matched_labels == 0] = self.num_classes
            else:
                gt_classes[matched_labels == 0] = self.mask_num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            if not segmentation_step:
                gt_classes = torch.zeros_like(matched_idxs) + self.num_classes
            else:
                gt_classes = torch.zeros_like(matched_idxs) + self.mask_num_classes

        if not segmentation_step:
            sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
                gt_classes, self.batch_size_per_image, self.positive_fraction, self.num_classes
            )
        else:
            sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
                gt_classes, self.batch_size_per_image, self.positive_fraction, self.mask_num_classes
            )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals, targets, segmentation_step=False
    ):
        gt_boxes = [x.gt_boxes for x in targets]
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes, segmentation_step=segmentation_step
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label.

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def forward(self, images, features, proposals, targets=None, relations=None, segmentation_step=False):
        if not self.novel_to_coco_similarity_matrix.is_cuda:
            device = next(self.box_predictor.parameters()).device
            self.novel_to_coco_similarity_matrix = self.novel_to_coco_similarity_matrix.to(device)
            self.base_classes_indexer_tensor = self.base_classes_indexer_tensor.to(device)
            self.novel_classes_indexer_tensor = self.novel_classes_indexer_tensor.to(device)
            self.output_to_coco_indexer_tensor = self.output_to_coco_indexer_tensor.to(device)
        del images
        with torch.no_grad():
            pred_instances = self._forward_box(features, proposals, targets=targets, segmentation_step=segmentation_step)
            pred_instances = self._forward_mask(features, pred_instances, segmentation_step=segmentation_step)
            if (not segmentation_step) and self.refine_seg_mask and (not self.training):
                features_copy = copy.deepcopy(features)
                residual_masks = self._forward_scenegraph(features_copy, pred_instances, targets, relations, segmentation_step=segmentation_step, return_masks=True)
                pred_instances = self._forward_mask(features, pred_instances, segmentation_step=segmentation_step, residual_masks=residual_masks)
        if self.training:
            _, _, losses = self._forward_scenegraph(features, pred_instances, targets, relations, segmentation_step=segmentation_step)
            return proposals, losses
        else:    
            # # During inference cascaded prediction is used: the mask and keypoints heads are only
            # # applied to the top scoring box detections.
            if ((len(pred_instances[0].pred_boxes) == 0) or (not self.segmentation_step_mask_refine)) and segmentation_step and (not self.training):
                if (len(pred_instances[0].pred_boxes) > 0):
                    print ("Here")
                    try:
                        pred_mask_logits = pred_instances[0].pred_masks
                        num_masks = pred_mask_logits.shape[0]
                        class_pred = cat([i.pred_classes for i in pred_instances])
                        indices = torch.arange(num_masks, device=class_pred.device)
                        mask_probs_pred = pred_mask_logits[indices, class_pred][:, None]
                        num_boxes_per_image = [len(i) for i in pred_instances]
                        mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)
                        for prob, instances in zip(mask_probs_pred, pred_instances):
                            instances.pred_masks = prob  # (1, Hmask, Wmask)
                    except:
                        import ipdb; ipdb.set_trace()
                return pred_instances, {}
            _, result, _ = self._forward_scenegraph(features, pred_instances, targets, relations, segmentation_step=segmentation_step)
            return result, {}    
        del targets
    
    def _forward_box(self, features, proposals, targets=None, segmentation_step=False):
        features = [features[f] for f in self.box_in_features]
        if self.use_gt_box:
            if not self.training:
                is_zero = np.any([len(target.gt_boxes)==0 for target in targets])
                if is_zero:
                    proposals_copy = copy.deepcopy(proposals)
            del proposals
            proposals = []
            for idx, target in enumerate(targets):
                instance = Instances(target.image_size)
                instance.proposal_boxes = Boxes(target.gt_boxes.tensor.detach().clone())
                if (len(instance.proposal_boxes) == 0) and segmentation_step and (not self.training):
                    box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals_copy])
                    box_features = self.box_head(box_features)
                    predictions = self.box_predictor(box_features, is_transfer=(not segmentation_step))
                    scores = self.box_predictor.predict_probs(predictions, proposals_copy)
                    boxes = self.box_predictor.predict_boxes(predictions, proposals_copy)
                    image_shapes = [x.image_size for x in proposals_copy]
                    results, indices =  fast_rcnn_inference(
                                boxes,
                                scores,
                                image_shapes,
                                self.box_predictor.test_score_thresh,
                                self.box_predictor.test_nms_thresh,
                                self.box_predictor.test_topk_per_image,
                                nms_filter_duplicates=False
                            )
                    instance = Instances(target.image_size)
                    instance.proposal_boxes = Boxes(results[0].pred_boxes.tensor.detach().clone())
                    if self.use_gt_object_label:
                        instance.scores = results[0].scores
                        instance.pred_scores = results[0].pred_scores
                        instance.pred_classes = results[0].pred_classes
                if segmentation_step and self.training:
                    instance.gt_boxes = Boxes(target.gt_boxes.tensor.detach().clone())
                    instance.gt_masks = copy.deepcopy(target.gt_masks)
                    instance.gt_classes = target.gt_classes.detach().clone()
                proposals.append(instance)
        else:
            if self.training:
                assert targets, "'targets' argument is required during training"
                proposals = self.label_and_sample_proposals(proposals, targets, segmentation_step=segmentation_step)
        if not self.use_gt_object_label:
            box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
            box_features = self.box_head(box_features)
            predictions = self.box_predictor(box_features, is_transfer=(not segmentation_step))
            del box_features
        else:
            predictions = None
        pred_instances, _ = self.box_predictor.inference(proposals, predictions=predictions, targets=targets, segmentation_step=segmentation_step)
        return pred_instances

    def _forward_mask(self, features, instances, segmentation_step=False, residual_masks=None):
        if not self.mask_on:
            return instances
        if self.use_gt_box and (not self.use_gt_object_label) and self.training:
            for idx, instance in enumerate(instances):
                mask_pred_classes = copy.deepcopy(instance.mask_pred_classes)
                pred_classes = copy.deepcopy(instance.pred_classes)
                instances[idx].mask_pred_classes = pred_classes
                instances[idx].pred_classes = mask_pred_classes
        if self.mask_pooler is not None:
            features = [features[f] for f in self.mask_in_features]
            boxes = [x.pred_boxes for x in instances]
            features = self.mask_pooler(features, boxes)
        else:
            features = {f: features[f] for f in self.mask_in_features}
        instances = self.mask_head(features, instances, self.novel_to_coco_similarity_matrix, self.base_classes_indexer_tensor, self.novel_classes_indexer_tensor, self.output_to_coco_indexer_tensor, segmentation_step=segmentation_step, residual_masks=residual_masks)
        if self.use_gt_box and (not self.use_gt_object_label) and self.training:
            for idx, instance in enumerate(instances):
                mask_pred_classes = copy.deepcopy(instance.mask_pred_classes)
                pred_classes = copy.deepcopy(instance.pred_classes)
                instances[idx].mask_pred_classes = pred_classes
                instances[idx].pred_classes = mask_pred_classes
        return instances

    def _forward_scenegraph(
        self, 
        features: Dict[str, torch.Tensor], 
        instances: List[Instances],
        targets = None, 
        relations=None,
        segmentation_step=False,
        return_masks=False
        ):
        """
        Forward logic of the mask prediction branch.
        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.
        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.scenegraph_on:
            return {} if self.training else instances

        # if self.scenegraph_pooler is not None:
        #     features = [features[f] for f in self.scenegraph_in_features]
        #     boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
        #     features = self.scenegraph_pooler(features, boxes)
        # else:
        #     features = {f: features[f] for f in self.scenegraph_in_features}

        return self.scenegraph_head(features, instances, targets, relations, segmentation_step=segmentation_step, return_masks=return_masks)

@ROI_HEADS_REGISTRY.register()
class StandardMaskLabelROIHead(StandardROIHeadsRandomTieBreaking):
    @configurable
    def __init__(self,*, box_in_features, box_pooler, box_head, box_predictor, mask_in_features=None, mask_pooler=None, mask_head=None, 
                keypoint_in_features=None, keypoint_pooler=None, keypoint_head=None, train_on_pred_boxes=False, **kwargs):
        self.train_data_name = kwargs['train_data_name']
        self.embeddings_path = kwargs['embeddings_path']
        self.embeddings_path_coco = kwargs['embeddings_path_coco']
        self.num_output_classes = kwargs['num_output_classes']
        self.transfer_data_name = kwargs['transfer_data_name']
        self.lingual_matrix_threshold = kwargs['lingual_matrix_threshold']
        del kwargs['transfer_data_name']
        del kwargs['train_data_name']
        del kwargs['embeddings_path']
        del kwargs['embeddings_path_coco']
        del kwargs['num_output_classes']
        del kwargs['lingual_matrix_threshold']
        super(StandardMaskLabelROIHead, self).__init__(box_in_features=box_in_features, box_pooler=box_pooler, box_head=box_head, box_predictor=box_predictor, 
                                                        mask_in_features=mask_in_features, mask_pooler=mask_pooler, mask_head=mask_head, 
                                                        keypoint_in_features=keypoint_in_features, keypoint_pooler=keypoint_pooler, keypoint_head=keypoint_head, 
                                                        train_on_pred_boxes=train_on_pred_boxes, **kwargs)

        pretrained_embeddings = torch.load(self.embeddings_path)['embeddings']
        pretrained_embeddings_coco = torch.load(self.embeddings_path_coco)['embeddings']
        self.embeddings = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=True)
        self.embeddings_coco = nn.Embedding.from_pretrained(pretrained_embeddings_coco, freeze=True)
        self._class_mapper()

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret['train_data_name'] = cfg.DATASETS.TRAIN[0]
        ret['embeddings_path'] = cfg.MODEL.ROI_HEADS.EMBEDDINGS_PATH
        ret['embeddings_path_coco'] = cfg.MODEL.ROI_HEADS.EMBEDDINGS_PATH_COCO
        ret['num_output_classes'] = cfg.MODEL.ROI_HEADS.NUM_OUTPUT_CLASSES
        ret['transfer_data_name'] = cfg.DATASETS.TRANSFER[0]
        ret['lingual_matrix_threshold'] = cfg.MODEL.ROI_HEADS.LINGUAL_MATRIX_THRESHOLD
        return ret

    def _class_mapper(self):
        self.coco_classes = {name.lower():idx for idx, name in enumerate(MetadataCatalog.get(self.transfer_data_name).thing_classes)}
        self.output_classes = {name.lower():idx for idx, name in enumerate(MetadataCatalog.get(self.train_data_name).thing_classes)}
        self.base_classes_indexer = []
        self.novel_classes_indexer = []
        self.output_to_coco_indexer = []
        if 'OI' in self.transfer_data_name:
            transfers = {'bike':'motorcycle', 'phone': 'mobile phone', 'arm': 'human_arm', 'basket': 'picnic basket', 'counter': 'countertop', 'ear': 'human ear', 'cup': 'coffee cup', 'eye': 'human eye', 'face': 'human face', 'guy': 'man', 'hair': 'human hair', 'hand': 'human hand', 'handle': 'door handle', 'head': 'human head', 'jean': 'jeans', 'lady': 'woman', 'leg': 'human leg', 'men': 'man', 'mouth': 'human mouth', 'nose': 'human nose', 'plane': 'airplane', 'pot': 'flowerpot', 'short':'shorts'}
            non_zero_masks = ['tortoise', 'magpie', 'sea turtle', 'football', 'ambulance', 'toy', 'apple', 'beer', 'chopsticks', 'bird', 'traffic light', 'croissant', 'cucumber', 'radish', 'towel', 'skull', 'washing machine', 'glove', 'belt', 'ball', 'backpack', 'surfboard', 'boot', 'hot dog', 'shorts', 'bus', 'boy', 'screwdriver', 'bicycle wheel', 'barge', 'laptop', 'miniskirt', 'drill (tool)', 'dress', 'bear', 'waffle', 'pancake', 'brown bear', 'woodpecker', 'blue jay', 'pretzel', 'bagel', 'teapot', 'person', 'swimwear', 'bat (animal)', 'starfish', 'popcorn', 'burrito', 'balloon', 'wrench', 'vehicle registration plate', 'toaster', 'flashlight', 'limousine', 'carnivore', 'scissors', 'computer keyboard', 'printer', 'traffic sign', 'shirt', 'cheese', 'sock', 'fire hydrant', 'tie', 'suitcase', 'muffin', 'snowmobile', 'clock', 'cattle', 'cello', 'jet ski', 'camel', 'suit', 'cat', 'bronze sculpture', 'juice', 'computer mouse', 'cookie', 'coin', 'calculator', 'cocktail', 'box', 'stapler', 'christmas tree', 'cowboy hat', 'studio couch', 'drink', 'zucchini', 'ladle', 'human mouth', 'dice', 'oven', 'couch', 'cricket ball', 'winter melon', 'spatula', 'whiteboard', 'hat', 'shower', 'eraser', 'fedora', 'guacamole', 'dagger', 'scarf', 'dolphin', 'sombrero', 'mug', 'tap', 'harbor seal', 'human body', 'roller skates', 'coffee cup', 'stop sign', 'volleyball (ball)', 'vase', 'slow cooker', 'coffee', 'paper towel', 'sun hat', 'flying disc', 'skirt', 'barrel', 'kite', 'tart', 'fox', 'flag', 'guitar', 'pillow', 'grape', 'human ear', 'power plugs and sockets', 'panda', 'giraffe', 'woman', 'door handle', 'rhinoceros', 'goldfish', 'goat', 'baseball bat', 'baseball glove', 'mixing bowl', 'light switch', 'horse', 'hammer', 'sofa bed', 'adhesive tape', 'saucer', 'harpsichord', 'heater', 'harmonica', 'hamster', 'kettle', 'drinking straw', 'hair dryer', 'food processor', 'punching bag', 'common fig', 'cocktail shaker', 'jaguar (animal)', 'golf ball', 'alarm clock', 'filing cabinet', 'artichoke', 'kangaroo', 'koala', 'knife', 'bottle', 'bottle opener', 'lynx', 'lighthouse', 'dumbbell', 'bowl', 'lizard', 'billiard table', 'mouse', 'motorcycle', 'swim cap', 'frying pan', 'missile', 'bust', 'man', 'milk', 'mobile phone', 'mushroom', 'pitcher (container)', 'table tennis racket', 'pencil case', 'briefcase', 'kitchen knife', 'nail (construction)', 'tennis ball', 'plastic bag', 'chest of drawers', 'ostrich', 'piano', 'girl', 'potato', 'penguin', 'pumpkin', 'pear', 'polar bear', 'pizza', 'digital clock', 'pig', 'reptile', 'lipstick', 'skateboard', 'raven', 'high heels', 'red panda', 'rose', 'rabbit', 'sculpture', 'saxophone', 'submarine sandwich', 'sword', 'picture frame', 'loveseat', 'squirrel', 'segway', 'snake', 'skyscraper', 'sheep', 'tea', 'tank', 'torch', 'tiger', 'strawberry', 'tomato', 'train', 'cooking spray', 'trousers', 'truck', 'measuring cup', 'handbag', 'wine', 'wheel', 'wok', 'whale', 'zebra', 'jug', 'pizza cutter', 'monkey', 'lion', 'bread', 'platter', 'chicken', 'eagle', 'owl', 'duck', 'turtle', 'hippopotamus', 'crocodile', 'toilet', 'toilet paper', 'clothing', 'lemon', 'frog', 'banana', 'rocket', 'tablet computer', 'waste container', 'dog', 'book', 'elephant', 'shark', 'candle', 'leopard', 'axe', 'hand dryer', 'soap dispenser', 'flower', 'canary', 'cheetah', 'hamburger', 'fish', 'garden asparagus', 'hedgehog', 'airplane', 'spoon', 'otter', 'bull', 'oyster', 'orange', 'beaker', 'goose', 'mule', 'swan', 'peach', 'seat belt', 'raccoon', 'chisel', 'camera', 'squash (plant)', 'racket', 'diaper', 'falcon', 'cabbage', 'carrot', 'mango', 'jeans', 'flowerpot', 'envelope', 'cake', 'common sunflower', 'microwave oven', 'sea lion', 'watch', 'parrot', 'handgun', 'sparrow', 'van', 'corded phone', 'tennis racket', 'dog bed', 'facial tissue holder', 'pressure cooker', 'ruler', 'luggage and bags', 'broccoli', 'pastry', 'grapefruit', 'band-aid', 'bell pepper', 'turkey', 'pomegranate', 'doughnut', 'pen', 'car', 'aircraft', 'skunk', 'teddy bear', 'watermelon', 'cantaloupe', 'flute', 'balance beam', 'sandwich', 'binoculars', 'ipod', 'alpaca', 'taxi', 'canoe', 'remote control', 'rugby ball', 'armadillo']
            zero_indices = [self.coco_classes[x] for x in self.coco_classes if x not in non_zero_masks]
        for idx, class_name in enumerate(self.output_classes.keys()):
            if 'coco' in self.transfer_data_name:
                if class_name == 'bike':
                    class_name = 'motorcycle'
                if class_name == 'ski':
                    class_name = 'skis'
                if class_name == 'phone':
                    class_name = 'cell phone'
                if class_name == 'woman' or class_name == 'men' or class_name == 'man' or class_name == 'lady' or class_name == 'girl' or class_name == 'guy' or class_name == 'boy' or class_name == 'child' or class_name == 'girl' or class_name == 'kid' or class_name == 'people':
                    class_name = 'person'
            else:
                if class_name in transfers:
                    if transfers[class_name] in non_zero_masks:
                        class_name = transfers[class_name]
                if class_name not in non_zero_masks:
                    class_name = class_name + "_nomask"
                    print (class_name)
            if class_name in self.coco_classes:
                self.base_classes_indexer.append(idx)
                self.output_to_coco_indexer.append(self.coco_classes[class_name])
            else:
                print (class_name)
                self.novel_classes_indexer.append(idx)
        self.base_classes_indexer = np.array(self.base_classes_indexer)
        self.novel_classes_indexer = np.array(self.novel_classes_indexer)
        self.output_to_coco_indexer = np.array(self.output_to_coco_indexer)

        self.base_classes_indexer_tensor = torch.tensor(self.base_classes_indexer).long()
        self.novel_classes_indexer_tensor = torch.tensor(self.novel_classes_indexer).long()
        self.output_to_coco_indexer_tensor = torch.tensor(self.output_to_coco_indexer).long()

        self.coco_class_embeddings = self.embeddings_coco(torch.arange(len(self.coco_classes)).long())
        self.output_class_embeddings = self.embeddings(torch.arange(len(self.output_classes)).long())
        self.novel_to_coco_similarity_matrix = torch.mm(torch.index_select(self.output_class_embeddings, 0, self.novel_classes_indexer_tensor), self.coco_class_embeddings.transpose(0,1))
        if 'OI' in self.transfer_data_name:
            self.novel_to_coco_similarity_matrix[:,zero_indices] = -np.inf
            # Bug in training code where classes were set to 600. 
            self.novel_to_coco_similarity_matrix = self.novel_to_coco_similarity_matrix[:,:-1]
        self.novel_to_coco_similarity_matrix = nn.functional.softmax(self.novel_to_coco_similarity_matrix, -1)
        self.novel_to_coco_similarity_matrix[self.novel_to_coco_similarity_matrix < self.lingual_matrix_threshold] = 0.0
        self.novel_to_coco_similarity_matrix = self.novel_to_coco_similarity_matrix / torch.sum(self.novel_to_coco_similarity_matrix, dim=-1, keepdim=True)

        # max_similar = self.novel_to_coco_similarity_matrix.argsort(-1, descending=True)
        # for idx, class_idx in enumerate(self.novel_classes_indexer):
        #     print (MetadataCatalog.get(self.train_data_name).thing_classes[class_idx],":",[MetadataCatalog.get('coco_2017_train').thing_classes[max_similar[idx][x]] for x in range(5)])
        
    def _forward_box(self, features, proposals):
        features = [features[f] for f in self.box_in_features]
        if self.training:
            raise NotImplementedError
        else:
            pred_instances = []
            for idx, proposal in enumerate(proposals):
                pred_instances.append(Instances(proposal.image_size))
                pred_instances[idx].pred_boxes = proposal.gt_boxes.clone()
                pred_instances[idx].pred_classes = proposal.gt_classes.clone()
                pred_instances[idx].score = torch.ones(len(proposal)).to(pred_instances[idx].pred_classes.device)
            return pred_instances

    def forward(self, images, features, proposals, targets=None, relations=None):
        if not self.novel_to_coco_similarity_matrix.is_cuda:
            device = next(self.box_predictor.parameters()).device
            self.novel_to_coco_similarity_matrix = self.novel_to_coco_similarity_matrix.to(device)
            self.base_classes_indexer_tensor = self.base_classes_indexer_tensor.to(device)
            self.novel_classes_indexer_tensor = self.novel_classes_indexer_tensor.to(device)
            self.output_to_coco_indexer_tensor = self.output_to_coco_indexer_tensor.to(device)

        del images
        del targets
        
        if self.training:
            raise NotImplementedError
        else:
            pred_instances = self._forward_box(features, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(self, features, instances):
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_mask(features, instances)
        instances = self._forward_keypoint(features, instances)
        return instances

    def _forward_mask(self, features, instances):
        if not self.mask_on:
            return {} if self.training else instances

        if self.mask_pooler is not None:
            features = [features[f] for f in self.mask_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.mask_pooler(features, boxes)
        else:
            features = {f: features[f] for f in self.mask_in_features}
        return self.mask_head(features, instances, self.novel_to_coco_similarity_matrix, self.base_classes_indexer_tensor, self.novel_classes_indexer_tensor, self.output_to_coco_indexer_tensor)

@ROI_HEADS_REGISTRY.register()
class StandardROIHeadsWithCOCO(StandardROIHeadsRandomTieBreaking):
    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeadsRandomTieBreaking is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        box_predictor = FastRCNNOutputLayerswithCOCO(cfg, box_head.output_shape)
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    def forward(
        self,
        images,
        features,
        proposals,
        targets,
        is_transfer = True
    ):
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets, "'targets' argument is required during training"
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals, is_transfer=is_transfer)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals, is_transfer=is_transfer)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def _forward_box(self, features, proposals, is_transfer=True):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.
        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".
        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features, is_transfer=is_transfer)
        del box_features

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances

    def _forward_mask(self, features: Dict[str, torch.Tensor], instances: List[Instances]):
        """
        Forward logic of the mask prediction branch.
        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.
        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        if self.training:
            # head is only trained on positive proposals.
            instances, _ = select_foreground_proposals(instances, self.num_classes)

        if self.mask_pooler is not None:
            features = [features[f] for f in self.mask_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.mask_pooler(features, boxes)
        else:
            features = {f: features[f] for f in self.mask_in_features}
        return self.mask_head(features, instances)