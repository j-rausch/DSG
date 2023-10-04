import copy
import inspect
import logging
import os
import sys
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from detectron2.config import CfgNode, configurable
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, Instances
from detectron2.utils.registry import Registry
from torch import nn
from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from .box_feature_extractor import build_box_feature_extractor
from .inference import build_roi_scenegraph_post_processor, build_roi_scenegraph_post_processor_with_grammar
from .loss import build_roi_scenegraph_loss_evaluator, build_roi_scenegraph_loss_evaluator_with_grammar
from .relation_feature_extractor import build_relation_feature_extractor
from .sampling import build_roi_scenegraph_samp_processor
from .scenegraph_predictor import build_roi_scenegraph_predictor
from detectron2.modeling.poolers import ROIPooler

ROI_SCENEGRAPH_HEAD_REGISTRY = Registry("ROI_SCENEGRAPH_HEAD_REGISTRY")

@ROI_SCENEGRAPH_HEAD_REGISTRY.register()
class SceneGraphHead(nn.Module):
    '''
    SceneGraphHeads predict scene graph from detected regions
    It uses the extracted feature , preforms context aggreagation and predict object and relation labels
    '''
    
    def __init__(self, cfg: CfgNode, input_shape: int):

        super(SceneGraphHead, self).__init__()
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)

        self.mode = self.cfg.MODEL.ROI_SCENEGRAPH_HEAD.MODE
        self.mask_on = self.cfg.MODEL.MASK_ON
        self.disable_masks_in_scenegraph_head = self.cfg.MODEL.ROI_SCENEGRAPH_HEAD.DISABLE_MASKS_IN_SCENEGRAPH_HEAD
        self.use_gt_box = self.cfg.MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_BOX
        self.use_gt_label = self.cfg.MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_OBJECT_LABEL
        self.use_union_box = self.cfg.MODEL.ROI_SCENEGRAPH_HEAD.PREDICT_USE_VISION
        self.box_feature_mask_logits = self.cfg.MODEL.ROI_BOX_FEATURE_EXTRACTORS.CLASS_LOGITS_WITH_MASK

        #Sample realtion proposal during training
        self.samp_processor = build_roi_scenegraph_samp_processor(cfg)

        #Featuer extractor for graph nodes and edges
        self.box_feature_extractor = build_box_feature_extractor(cfg, input_shape)

        feat_dim = self.box_feature_extractor.out_channels
        self.union_feature_extractor = build_relation_feature_extractor(cfg, input_shape)

        #Generate scene graph classification scores from extracted features
        self.predictor = build_roi_scenegraph_predictor(cfg, feat_dim)
        # From classification scores and box regression compute postprocessed box and apply nms
        self.post_processor = build_roi_scenegraph_post_processor(cfg)
        # #Compute loss for generated scene graph
        self.loss_evaluator = build_roi_scenegraph_loss_evaluator(cfg)
    
    def forward(self, features: Dict[str, torch.Tensor], proposals: List[Instances], targets=None, relations=None, segmentation_step=False, return_masks=False):
        '''
        Params:
        -------
            features     : feature maps
            proposals    : List of detected instances
            targets      : Ground truth (used during training)
        '''
        masks = None
        logits = None
        if self.mask_on and (not self.disable_masks_in_scenegraph_head):
            masks = [x.pred_masks for x in proposals]
            if self.box_feature_mask_logits:
                logits = [x.pred_scores for x in proposals]
        
        if self.training:
            with torch.no_grad():
                if self.use_gt_box:
                    #Create a list of bounding boxes
                    boxes = [x.pred_boxes for x in proposals]
                    boxes, rel_labels, rel_pair_idxs, rel_binarys = self.samp_processor.gtbox_relsample(boxes, targets, relations)
                else:
                    proposals, rel_labels, rel_pair_idxs, rel_binarys = self.samp_processor.detect_relsample(proposals, targets, relations)
                    boxes = [x.pred_boxes for x in proposals]
        else:
            boxes = [x.pred_boxes for x in proposals]
            rel_labels, rel_binarys = None, None
            rel_pair_idxs = self.samp_processor.prepare_test_pairs(boxes[0].device, proposals)
        cat_boxes = Boxes.cat(boxes)
        if len(cat_boxes) == 0:
            #TODO: cases where no/few proposals are passed are currently badly supported
            self.logger.warning("No valid proposal boxes passed to scenegraph head! returning None")
            if self.training:
                return None, (None, None, rel_pair_idxs, proposals), None #if no boxes are passed to scenegraph head, nothing can be done
            else:
                roi_features = self.box_feature_extractor(features, boxes, masks=masks,
                                                          logits=logits)
                results = []
                for prp in proposals:
                    result = prp
                    result._rel_pair_idxs = None
                    result._pred_rel_scores = None
                    results.append(result)
                return roi_features, results, {}

        # # use box_head to extract features that will be fed to the later predictor processing
        roi_features = self.box_feature_extractor(features, boxes, masks=masks, logits=logits)
        # roi_features = self.box_feature_extractor(features, boxes, masks=None)
        
        if self.use_union_box:
            union_features, viz_outputs = self.union_feature_extractor(features, boxes, rel_pair_idxs, masks=masks, proposals=proposals, return_seg_masks=self.cfg.MODEL.ROI_SCENEGRAPH_HEAD.RETURN_SEG_MASKS)
        else:
            union_features = None
        #Context aggragation followed by label predcition
        refine_logits, relation_logits, add_losses = self.predictor(proposals, boxes, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, self.logger)

        if not self.training:
            img_sizes = [proposal.image_size for proposal in proposals]
            
            if self.use_gt_box:
                result = self.post_processor((relation_logits, refine_logits), rel_pair_idxs, boxes=boxes, img_sizes=img_sizes, segmentation_vis=self.cfg.MODEL.ROI_SCENEGRAPH_HEAD.RETURN_SEG_MASKS)
            else:
                #TODO: add score thresholding of instances and subsequent filtering of predicted relationsgT
                result = self.post_processor((relation_logits, refine_logits), rel_pair_idxs, boxes=boxes, img_sizes=img_sizes, segmentation_vis=self.cfg.MODEL.ROI_SCENEGRAPH_HEAD.RETURN_SEG_MASKS, proposals=proposals)
            del proposals
            if self.cfg.MODEL.ROI_SCENEGRAPH_HEAD.RETURN_SEG_MASKS:
                for idx in range(len(result)):
                    for key in viz_outputs:
                        viz_outputs[key] = viz_outputs[key][result[idx]._sorting_idx]
                    result[idx]._viz_outputs = viz_outputs
            
            return roi_features, result, {}

        # #Compute loss and create loss dict
        if self.use_gt_box:
            loss_relation, loss_refine = self.loss_evaluator(targets, rel_labels, relation_logits, refine_logits)
        else:
            loss_relation, loss_refine = self.loss_evaluator(proposals, rel_labels, relation_logits, refine_logits)
        
        output_losses = dict(loss_rel=loss_relation, loss_refine_obj=loss_refine)

        output_losses.update(add_losses)
        return roi_features, (relation_logits, refine_logits, rel_pair_idxs, proposals), output_losses


@ROI_SCENEGRAPH_HEAD_REGISTRY.register()
class SceneGraphHeadWithGrammar(nn.Module):
    '''
    SceneGraphHeads predict scene graph from detected regions
    It uses the extracted feature , preforms context aggreagation and predict object and relation labels
    '''

    def __init__(self, cfg: CfgNode, input_shape: int):

        super(SceneGraphHeadWithGrammar, self).__init__()
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)

        self.mode = self.cfg.MODEL.ROI_SCENEGRAPH_HEAD.MODE
        self.mask_on = self.cfg.MODEL.MASK_ON
        self.use_gt_box = self.cfg.MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_BOX
        self.use_gt_label = self.cfg.MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_OBJECT_LABEL
        self.use_union_box = self.cfg.MODEL.ROI_SCENEGRAPH_HEAD.PREDICT_USE_VISION
        self.box_feature_mask_logits = self.cfg.MODEL.ROI_BOX_FEATURE_EXTRACTORS.CLASS_LOGITS_WITH_MASK

        # Sample realtion proposal during training
        self.samp_processor = build_roi_scenegraph_samp_processor(cfg)

        # Featuer extractor for graph nodes and edges
        self.box_feature_extractor = build_box_feature_extractor(cfg, input_shape)

        feat_dim = self.box_feature_extractor.out_channels
        self.union_feature_extractor = build_relation_feature_extractor(cfg, input_shape)

        # Generate scene graph classification scores from extracted features
        self.predictor = build_roi_scenegraph_predictor(cfg, feat_dim)
        # From classification scores and box regression compute postprocessed box and apply nms
        self.post_processor = build_roi_scenegraph_post_processor_with_grammar(cfg)
        # #Compute loss for generated scene graph
        self.loss_evaluator = build_roi_scenegraph_loss_evaluator_with_grammar(cfg)

    def forward(self, features: Dict[str, torch.Tensor], proposals: List[Instances], targets=None, relations=None,
                segmentation_step=False, return_masks=False):
        '''
        Params:
        -------
            features     : feature maps
            proposals    : List of detected instances
            targets      : Ground truth (used during training)
        '''
        masks = None
        logits = None
        if self.mask_on:
            masks = [x.pred_masks for x in proposals]
            if self.box_feature_mask_logits:
                logits = [x.pred_scores for x in proposals]

        if self.training:
            with torch.no_grad():
                if self.use_gt_box:
                    # Create a list of bounding boxes
                    boxes = [x.pred_boxes for x in proposals]
                    boxes, rel_labels, rel_pair_idxs, rel_binarys = self.samp_processor.gtbox_relsample(boxes, targets,
                                                                                                        relations)
                else:
                    proposals, rel_labels, rel_pair_idxs, rel_binarys = self.samp_processor.detect_relsample(proposals,
                                                                                                             targets,
                                                                                                             relations)
                    boxes = [x.pred_boxes for x in proposals]
        else:
            boxes = [x.pred_boxes for x in proposals]
            rel_labels, rel_binarys = None, None
            rel_pair_idxs = self.samp_processor.prepare_test_pairs(boxes[0].device, proposals)
        cat_boxes = Boxes.cat(boxes)
        if len(cat_boxes) == 0:
            # TODO: cases where no/few proposals are passed are currently badly supported
            self.logger.warning("No valid proposal boxes passed to scenegraph head! returning None")
            return None, (None, None, rel_pair_idxs,
                          proposals), None  # if no boxes are passed to scenegraph head, nothing can be done
        # # use box_head to extract features that will be fed to the later predictor processing

        roi_features = self.box_feature_extractor(features, boxes, masks=masks, logits=logits)
        # roi_features = self.box_feature_extractor(features, boxes, masks=None)

        if self.use_union_box:
            union_features, viz_outputs = self.union_feature_extractor(features, boxes, rel_pair_idxs, masks=masks,
                                                                       proposals=proposals,
                                                                       return_seg_masks=self.cfg.MODEL.ROI_SCENEGRAPH_HEAD.RETURN_SEG_MASKS)
        else:
            union_features = None
        # Context aggragation followed by label predcition
        refine_logits, relation_logits, add_losses, grammar_outputs = self.predictor(proposals, boxes, rel_pair_idxs,
                                                                                     rel_labels, rel_binarys,
                                                                                     roi_features, union_features,
                                                                                     self.logger)

        grammar_outputs['rel_pair_idxs'] = rel_pair_idxs

        if not self.training:
            img_sizes = [proposal.image_size for proposal in proposals]

            if self.use_gt_box:
                # TODO: add grammar loss implementation for postprocessing
                result = self.post_processor((relation_logits, refine_logits), rel_pair_idxs, boxes, img_sizes,
                                             segmentation_vis=self.cfg.MODEL.ROI_SCENEGRAPH_HEAD.RETURN_SEG_MASKS,
                                             grammar_outputs=grammar_outputs)
            else:
                result = self.post_processor((relation_logits, refine_logits), rel_pair_idxs, proposals, img_sizes,
                                             segmentation_vis=self.cfg.MODEL.ROI_SCENEGRAPH_HEAD.RETURN_SEG_MASKS,
                                             grammar_outputs=grammar_outputs)
            del proposals
            if self.cfg.MODEL.ROI_SCENEGRAPH_HEAD.RETURN_SEG_MASKS:
                for idx in range(len(result)):
                    for key in viz_outputs:
                        viz_outputs[key] = viz_outputs[key][result[idx]._sorting_idx]
                    result[idx]._viz_outputs = viz_outputs

            return roi_features, result, {}

        # #Compute loss and create loss dict
        if self.use_gt_box:
            loss_relation, loss_refine = self.loss_evaluator(targets, rel_labels, relation_logits, refine_logits,
                                                             grammar_outputs=grammar_outputs)
        else:
            loss_relation, loss_refine = self.loss_evaluator(proposals, rel_labels, relation_logits, refine_logits,
                                                             grammar_outputs=grammar_outputs)

        output_losses = dict(loss_refine_obj=loss_refine)
        #output_losses = dict(loss_rel=loss_relation, loss_refine_obj=loss_refine)
        if isinstance(loss_relation, dict):
            output_losses.update(loss_relation)
        else:
            output_losses['loss_rel'] = loss_relation

        output_losses.update(add_losses)
        return roi_features, (relation_logits, refine_logits, rel_pair_idxs, proposals), output_losses


@ROI_SCENEGRAPH_HEAD_REGISTRY.register()
class SceneGraphHeadVGG(SceneGraphHead):
    def __init__(self, cfg: CfgNode, input_shape: int):
        if cfg.MODEL.BACKBONE.NAME == 'VGG':
            input_shape['vgg_conv'] = ShapeSpec(channels=input_shape['vgg_conv'].channels//2, height=input_shape['vgg_conv'].height, width=input_shape['vgg_conv'].width, stride=input_shape['vgg_conv'].stride) 
        super(SceneGraphHeadVGG, self).__init__(cfg, input_shape)
        self.backbone_name = cfg.MODEL.BACKBONE.NAME
        if self.backbone_name == 'VGG':
            conv_norm = cfg.MODEL.ROI_MASK_HEAD.NORM
            conv_dim = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
            num_conv = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
            conv_dims=[conv_dim] * (num_conv + 1)
            self.feature_reducer = Conv2d(
                conv_dims[-1]*2,
                conv_dims[-1],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dims[-1]),
                activation=nn.ReLU(),
            )

    def forward(self, features: Dict[str, torch.Tensor], proposals: List[Instances], targets=None, relations=None, segmentation_step=False, return_masks=False):
        '''
        Params:
        -------
            features     : feature maps
            proposals    : List of detected instances
            targets      : Ground truth (used during training)
        '''
        if self.backbone_name == 'VGG':
            features['vgg_conv'] = self.feature_reducer(features['vgg_conv'])
        masks = None
        logits = None
        if self.mask_on:
            masks = [x.pred_masks for x in proposals]
            if self.box_feature_mask_logits:
                logits = [x.pred_scores for x in proposals]
        
        if self.training:
            with torch.no_grad():
                if self.use_gt_box:
                    #Create a list of bounding boxes
                    boxes = [x.pred_boxes for x in proposals]
                    boxes, rel_labels, rel_pair_idxs, rel_binarys = self.samp_processor.gtbox_relsample(boxes, targets, relations)
                else:
                    proposals, rel_labels, rel_pair_idxs, rel_binarys = self.samp_processor.detect_relsample(proposals, targets, relations)
                    boxes = [x.pred_boxes for x in proposals]
        else:
            boxes = [x.pred_boxes for x in proposals]
            rel_labels, rel_binarys = None, None
            rel_pair_idxs = self.samp_processor.prepare_test_pairs(boxes[0].device, proposals)

        # # use box_head to extract features that will be fed to the later predictor processing

        roi_features = self.box_feature_extractor(features, boxes, masks=masks, logits=logits)
        # roi_features = self.box_feature_extractor(features, boxes, masks=None)
        
        if self.use_union_box:
            union_features, viz_outputs = self.union_feature_extractor(features, boxes, rel_pair_idxs, masks=masks, proposals=proposals, return_seg_masks=self.cfg.MODEL.ROI_SCENEGRAPH_HEAD.RETURN_SEG_MASKS)
        else:
            union_features = None
        #Context aggragation followed by label predcition
        refine_logits, relation_logits, add_losses = self.predictor(proposals, boxes, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, self.logger)

        if not self.training:
            img_sizes = [proposal.image_size for proposal in proposals]
            
            if self.use_gt_box:
                result = self.post_processor((relation_logits, refine_logits), rel_pair_idxs, boxes, img_sizes, segmentation_vis=self.cfg.MODEL.ROI_SCENEGRAPH_HEAD.RETURN_SEG_MASKS)
            else:
                result = self.post_processor((relation_logits, refine_logits), rel_pair_idxs, proposals, img_sizes, segmentation_vis=self.cfg.MODEL.ROI_SCENEGRAPH_HEAD.RETURN_SEG_MASKS) 
            del proposals
            if self.cfg.MODEL.ROI_SCENEGRAPH_HEAD.RETURN_SEG_MASKS:
                for idx in range(len(result)):
                    for key in viz_outputs:
                        viz_outputs[key] = viz_outputs[key][result[idx]._sorting_idx]
                    result[idx]._viz_outputs = viz_outputs
            return roi_features, result, {}

        # #Compute loss and create loss dict
        if self.use_gt_box:
            loss_relation, loss_refine = self.loss_evaluator(targets, rel_labels, relation_logits, refine_logits)
        else:
            loss_relation, loss_refine = self.loss_evaluator(proposals, rel_labels, relation_logits, refine_logits)
        
        output_losses = dict(loss_rel=loss_relation, loss_refine_obj=loss_refine)

        output_losses.update(add_losses)
        return roi_features, (relation_logits, refine_logits, rel_pair_idxs, proposals), output_losses

@ROI_SCENEGRAPH_HEAD_REGISTRY.register()
class SceneGraphSegmentationHead(SceneGraphHead):
    def __init__(self, cfg: CfgNode, input_shape: int):
        if cfg.MODEL.BACKBONE.NAME == 'VGG':
            input_shape['vgg_conv'] = ShapeSpec(channels=input_shape['vgg_conv'].channels//2, height=input_shape['vgg_conv'].height, width=input_shape['vgg_conv'].width, stride=input_shape['vgg_conv'].stride) 
        super(SceneGraphSegmentationHead, self).__init__(cfg, input_shape)

        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_FEATURE_EXTRACTORS.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_FEATURE_EXTRACTORS.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_FEATURE_EXTRACTORS.POOLER_TYPE
        mask_on           = cfg.MODEL.MASK_ON
        use_mask_in_box_features = cfg.MODEL.ROI_BOX_FEATURE_EXTRACTORS.BOX_FEATURE_MASK
        self.pooler = ROIPooler(
                            output_size=pooler_resolution,
                            scales=pooler_scales,
                            sampling_ratio=sampling_ratio,
                            pooler_type=pooler_type
                            )
        self.backbone_name = cfg.MODEL.BACKBONE.NAME
        if self.backbone_name == 'VGG':
            conv_norm = cfg.MODEL.ROI_MASK_HEAD.NORM
            conv_dim = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
            num_conv = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
            conv_dims=[conv_dim] * (num_conv + 1)
            self.feature_reducer = Conv2d(
                conv_dims[-1]*2,
                conv_dims[-1],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dims[-1]),
                activation=nn.ReLU(),
            )
        self.in_features = in_features
        self.segmentation_criterion_loss = nn.CrossEntropyLoss()
        self.seg_bbox_loss_multiplier = cfg.MODEL.ROI_SCENEGRAPH_HEAD.SEG_BBOX_LOSS_MULTIPLIER
        if self.cfg.MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

    def forward(self, features: Dict[str, torch.Tensor], proposals: List[Instances], targets=None, relations=None, segmentation_step=False, return_masks=False):
        '''
        Params:
        -------
            features     : feature maps
            proposals    : List of detected instances
            targets      : Ground truth (used during training)
        '''
        if self.backbone_name == 'VGG':
            features['vgg_conv'] = self.feature_reducer(features['vgg_conv'])
        masks = None
        logits = None
        if self.mask_on:
            masks = [x.pred_masks for x in proposals]
            if self.box_feature_mask_logits:
                logits = [x.pred_scores for x in proposals]
        if self.training and (not segmentation_step):
            with torch.no_grad():
                if self.use_gt_box:
                    #Create a list of bounding boxes
                    boxes = [x.pred_boxes for x in proposals]
                    boxes, rel_labels, rel_pair_idxs, rel_binarys = self.samp_processor.gtbox_relsample(boxes, targets, relations)
                else:
                    proposals, rel_labels, rel_pair_idxs, rel_binarys = self.samp_processor.detect_relsample(proposals, targets, relations)
                    boxes = [x.pred_boxes for x in proposals]
        else:
            boxes = [x.pred_boxes for x in proposals]
            rel_labels, rel_binarys = None, None
            rel_pair_idxs = self.samp_processor.prepare_test_pairs(boxes[0].device, proposals)

        # # use box_head to extract features that will be fed to the later predictor processing
        roi_features = self.box_feature_extractor(features, boxes, masks=masks, logits=logits, segmentation_step=segmentation_step)
        # roi_features = self.box_feature_extractor(features, boxes, masks=None)
        if self.use_union_box and (not segmentation_step) and (not return_masks):
            union_features, viz_outputs = self.union_feature_extractor(features, boxes, rel_pair_idxs, masks=masks, proposals=proposals, return_seg_masks=self.cfg.MODEL.ROI_SCENEGRAPH_HEAD.RETURN_SEG_MASKS)
        else:
            union_features = None
        if segmentation_step or return_masks:
            mask_features = [features[f] for f in self.in_features]
            mask_box_features = self.pooler(mask_features, boxes)
        else:
            mask_box_features = None
        #Context aggragation followed by label predcition
        if return_masks:
            return self.predictor(proposals, boxes, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, self.logger, mask_box_features=mask_box_features, masks=masks, segmentation_step=segmentation_step, return_masks=return_masks)
        refine_logits, relation_logits, add_losses, mask_losses, proposals = self.predictor(proposals, boxes, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, self.logger, mask_box_features=mask_box_features, masks=masks, segmentation_step=segmentation_step, return_masks=False)

        if not self.training:
            img_sizes = [proposal.image_size for proposal in proposals]
            if not segmentation_step:
                if self.use_gt_box:
                    result = self.post_processor((relation_logits, refine_logits), rel_pair_idxs, boxes, img_sizes, segmentation_vis=(self.cfg.MODEL.ROI_SCENEGRAPH_HEAD.RETURN_SEG_MASKS or self.cfg.MODEL.ROI_SCENEGRAPH_HEAD.RETURN_SEG_ANNOS))
                else:
                    result = self.post_processor((relation_logits, refine_logits), rel_pair_idxs, proposals, img_sizes, segmentation_vis=(self.cfg.MODEL.ROI_SCENEGRAPH_HEAD.RETURN_SEG_MASKS or self.cfg.MODEL.ROI_SCENEGRAPH_HEAD.RETURN_SEG_ANNOS)) 
                if self.cfg.MODEL.ROI_SCENEGRAPH_HEAD.RETURN_SEG_ANNOS:
                    for idx in range(len(result)):
                        pred_mask_logits = proposals[0].pred_masks.detach().clone()
                        num_masks = pred_mask_logits.shape[0]
                        class_pred = result[idx].pred_classes
                        indices = torch.arange(num_masks, device=class_pred.device)
                        mask_probs_pred = pred_mask_logits[indices, class_pred][:, None]
                        result[idx].pred_masks = mask_probs_pred
                del proposals
                if self.cfg.MODEL.ROI_SCENEGRAPH_HEAD.RETURN_SEG_MASKS:
                    for idx in range(len(result)):
                        for key in viz_outputs:
                            viz_outputs[key] = viz_outputs[key][result[idx]._sorting_idx]
                        result[idx]._viz_outputs = viz_outputs
                return roi_features, result, {}
            else:
                return roi_features, proposals, {}

        # #Compute loss and create loss dict
        if (not segmentation_step):
            if self.use_gt_box:
                loss_relation, loss_refine = self.loss_evaluator(targets, rel_labels, relation_logits, refine_logits)
            else:
                loss_relation, loss_refine = self.loss_evaluator(proposals, rel_labels, relation_logits, refine_logits)
            output_losses = dict(loss_rel=loss_relation, loss_refine_obj=loss_refine)
        else:
            # Classification Loss
            # if refine_logits is not None:
            #     if self.use_gt_box:
            #         gt_classes = torch.cat([x.gt_classes for x in targets], dim=0)
            #     else:
            #         gt_classes = torch.cat([x.gt_classes for x in proposals], dim=0)
            #     # import ipdb; ipdb.set_trace()
            #     loss_segmentation_cls = self.segmentation_criterion_loss(refine_logits, gt_classes) * self.seg_bbox_loss_multiplier
            #     output_losses = dict(loss_seg_cls=loss_segmentation_cls)
            # else:
            output_losses = {}

        output_losses.update(add_losses)
        output_losses.update(mask_losses)
        return roi_features, (relation_logits, refine_logits, rel_pair_idxs, proposals), output_losses


@ROI_SCENEGRAPH_HEAD_REGISTRY.register()
class SceneGraphSegmentationHeadEnd2End(SceneGraphSegmentationHead):
    """
    SceneGraphSegmentationHeadEnd2End implements the instance-segmentation refinement step from SceneGraphSegmentationHead
    using a single call of the scene-graph head.

    Requirement: need to use a predictor similar to MotifSegmentationPredictor that implements a segmentation-step.
    """

    def __init__(self, cfg: CfgNode, input_shape: int):
        super(SceneGraphSegmentationHeadEnd2End, self).__init__(cfg, input_shape)
        self.mask_refinement = cfg.MODEL.ROI_SCENEGRAPH_HEAD.MASK_REFINEMENT
        self.disable_masks_in_scenegraph_head = self.cfg.MODEL.ROI_SCENEGRAPH_HEAD.DISABLE_MASKS_IN_SCENEGRAPH_HEAD

    def forward(self, features: Dict[str, torch.Tensor], proposals: List[Instances], targets=None, relations=None,
                segmentation_step=False, return_masks=False):
        masks = None
        logits = None
        if self.mask_on and (not self.disable_masks_in_scenegraph_head):
            masks = [x.pred_masks for x in proposals]
            if self.box_feature_mask_logits:
                logits = [x.pred_scores for x in proposals]

        # Mask-refinement
        if self.mask_on and (not self.disable_masks_in_scenegraph_head) and self.mask_refinement:
            boxes = [x.pred_boxes for x in proposals]
            rel_labels, rel_binarys = None, None
            rel_pair_idxs = self.samp_processor.prepare_test_pairs(boxes[0].device, proposals)

            roi_features = self.box_feature_extractor(features, boxes, masks=masks, logits=logits,
                                                      segmentation_step=True)

            union_features = None

            mask_features = [features[f] for f in self.in_features]
            mask_box_features = self.pooler(mask_features, boxes)

            _, _, _, mask_losses, proposals = self.predictor(proposals, boxes,
                                                             rel_pair_idxs,
                                                             rel_labels,
                                                             rel_binarys,
                                                             roi_features,
                                                             union_features,
                                                             self.logger,
                                                             mask_box_features=mask_box_features,
                                                             masks=masks,
                                                             segmentation_step=True,
                                                             return_masks=False)

            # update masks and logits (necessary?)
            masks = [x.pred_masks for x in proposals]
            if self.box_feature_mask_logits:
                logits = [x.pred_scores for x in proposals]

        else:
            mask_losses = {}

        # Scenegraph Prediction
        if self.training:
            with torch.no_grad():
                if self.use_gt_box:
                    # Create a list of bounding boxes
                    boxes = [x.pred_boxes for x in proposals]
                    boxes, rel_labels, rel_pair_idxs, rel_binarys = self.samp_processor.gtbox_relsample(boxes, targets,
                                                                                                        relations)
                else:
                    proposals, rel_labels, rel_pair_idxs, rel_binarys = self.samp_processor.detect_relsample(proposals,
                                                                                                             targets,
                                                                                                             relations)
                    boxes = [x.pred_boxes for x in proposals]
        else:
            boxes = [x.pred_boxes for x in proposals]
            rel_labels, rel_binarys = None, None
            rel_pair_idxs = self.samp_processor.prepare_test_pairs(boxes[0].device, proposals)
        cat_boxes = Boxes.cat(boxes)
        if len(cat_boxes) == 0:
            # TODO: cases where no/few proposals are passed are currently badly supported
            self.logger.warning("No valid proposal boxes passed to scenegraph head! returning None")
            return None, (None, None, rel_pair_idxs,
                          proposals), None  # if no boxes are passed to scenegraph head, nothing can be done
        # # use box_head to extract features that will be fed to the later predictor processing

        roi_features = self.box_feature_extractor(features, boxes, masks=masks, logits=logits,
                                                  segmentation_step=False)
        # roi_features = self.box_feature_extractor(features, boxes, masks=None)

        if self.use_union_box:
            union_features, viz_outputs = self.union_feature_extractor(features, boxes, rel_pair_idxs, masks=masks,
                                                                       proposals=proposals,
                                                                       return_seg_masks=self.cfg.MODEL.ROI_SCENEGRAPH_HEAD.RETURN_SEG_MASKS)
        else:
            union_features = None
        # Context aggragation followed by label predcition
        refine_logits, relation_logits, add_losses, _, _ = self.predictor(proposals, boxes, rel_pair_idxs, rel_labels,
                                                                          rel_binarys, roi_features, union_features,
                                                                          self.logger)

        if not self.training:
            img_sizes = [proposal.image_size for proposal in proposals]

            if self.use_gt_box:
                result = self.post_processor((relation_logits, refine_logits), rel_pair_idxs, boxes, img_sizes,
                                             segmentation_vis=self.cfg.MODEL.ROI_SCENEGRAPH_HEAD.RETURN_SEG_MASKS)
            else:
                # TODO: add score thresholding of instances and subsequent filtering of predicted relationsgT
                result = self.post_processor((relation_logits, refine_logits), rel_pair_idxs, proposals, img_sizes,
                                             segmentation_vis=self.cfg.MODEL.ROI_SCENEGRAPH_HEAD.RETURN_SEG_MASKS)
            del proposals
            if self.cfg.MODEL.ROI_SCENEGRAPH_HEAD.RETURN_SEG_MASKS:
                for idx in range(len(result)):
                    for key in viz_outputs:
                        viz_outputs[key] = viz_outputs[key][result[idx]._sorting_idx]
                    result[idx]._viz_outputs = viz_outputs

            return roi_features, result, {}

        # #Compute loss and create loss dict
        if self.use_gt_box:
            loss_relation, loss_refine = self.loss_evaluator(targets, rel_labels, relation_logits, refine_logits)
        else:
            loss_relation, loss_refine = self.loss_evaluator(proposals, rel_labels, relation_logits, refine_logits)

        output_losses = dict(loss_rel=loss_relation, loss_refine_obj=loss_refine)

        output_losses.update(add_losses)
        output_losses.update(mask_losses)
        return roi_features, (relation_logits, refine_logits, rel_pair_idxs, proposals), output_losses


def build_scenegraph_head(cfg, input_shape):

    name = cfg.MODEL.ROI_SCENEGRAPH_HEAD.NAME
    return ROI_SCENEGRAPH_HEAD_REGISTRY.get(name)(cfg, input_shape)
