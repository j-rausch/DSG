import logging
import torch
import copy
from torch import nn
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.layers import batched_nms
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes, Instances
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.layers import Linear, ShapeSpec, batched_nms, cat, nonzero_tuple
from detectron2.modeling.roi_heads.roi_heads import select_foreground_proposals
from segmentationsg.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals

def fast_rcnn_inference(boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image, nms_filter_duplicates):
    """
    Call `fast_rcnn_inference_single_image` for all images.
    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.
    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image, nms_filter_duplicates
        )
        for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def fast_rcnn_inference_single_image(
    boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image, nms_filter_duplicates
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).
    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.
    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    scores_backup = scores
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4
    boxes_per_class = boxes.clone()
    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    #print(sum(filter_mask))
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]

    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)

    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
   
    # if nms_filter_duplicates:
    #     inds_all = torch.zeros_like(scores_backup)
    #     inds_all[filter_inds[:,0], filter_inds[:,1]] = 1
    #     dist_scores = scores_backup * inds_all.float()
    #     scores_pre, labels_pre = dist_scores.max(1)
    #     final_inds = scores_pre.nonzero()
    #     assert final_inds.dim() != 0
    #     final_inds = final_inds.squeeze(1)

    #     scores_pre = scores_pre[final_inds]
    #     labels_pre = labels_pre[final_inds]

    #     boxes = boxes_per_class[final_inds, labels_pre]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    # result.scores = scores_pre
    # result.pred_scores = scores_backup[final_inds]
    # result.pred_classes = labels_pre
    # result.boxes_per_cls = boxes_per_class[final_inds]

    # return result, final_inds
    result.scores = scores
    result.pred_scores = scores_backup[filter_inds[:,0]]
    result.pred_classes = filter_inds[:, 1]
    result.boxes_per_cls = boxes_per_class[filter_inds[:,0]]
    return result, filter_inds[:, 0]


class FastRCNNOutputLayersSG(FastRCNNOutputLayers):
    """
    Two linear layers for predicting Fast R-CNN outputs:
    1. proposal-to-detection box regression deltas
    2. classification scores
    """
    @configurable
    def __init__(self, input_shape, *, box2box_transform, num_classes, num_output_classes, 
                test_score_thresh=0.0, test_nms_thresh=0.5, train_nms_thresh=0.7, test_topk_per_image=100,
                cls_agnostic_bbox_reg=False, smooth_l1_beta=0.0,box_reg_loss_type="smooth_l1", loss_weight=1.0, 
                use_gt_box=True, use_gt_object_label=True, nms_filter_duplicates=True):
        super(FastRCNNOutputLayersSG, self).__init__(input_shape=input_shape, box2box_transform=box2box_transform, num_classes=num_classes, test_score_thresh=test_score_thresh, test_nms_thresh=test_nms_thresh, test_topk_per_image=test_topk_per_image, cls_agnostic_bbox_reg=cls_agnostic_bbox_reg, smooth_l1_beta=smooth_l1_beta, box_reg_loss_type=box_reg_loss_type, loss_weight=loss_weight)
        self.use_gt_box = use_gt_box
        self.use_gt_object_label = use_gt_object_label
        self.num_classes = num_classes
        self.nms_filter_duplicates = nms_filter_duplicates
        self.train_nms_thresh = train_nms_thresh

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
        "input_shape": input_shape,
        "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
        # fmt: off
        "num_classes"           : cfg.MODEL.ROI_HEADS.NUM_CLASSES,
        "num_output_classes"    : cfg.MODEL.ROI_HEADS.NUM_OUTPUT_CLASSES,
        "cls_agnostic_bbox_reg" : cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
        "smooth_l1_beta"        : cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
        "test_score_thresh"     : cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
        "test_nms_thresh"       : cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
        "test_topk_per_image"   : cfg.TEST.DETECTIONS_PER_IMAGE,
        "box_reg_loss_type"     : cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE,
        "loss_weight"           : {"loss_box_reg": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT},
        "use_gt_box"            : cfg.MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_BOX,
        "use_gt_object_label"   : cfg.MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_OBJECT_LABEL,
        "nms_filter_duplicates" : cfg.MODEL.ROI_SCENEGRAPH_HEAD.NMS_FILTER_DUPLICATES
        # fmt: on
                }

    def inference(self, proposals, predictions=None, targets=None):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.
        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        if self.use_gt_box:
            outputs = []
            for idx, proposal in enumerate(proposals):
                instance = Instances(proposal.image_size)
                instance.pred_boxes = Boxes(proposal.proposal_boxes.tensor.clone().detach())
                if not self.use_gt_object_label:
                    scores = self.predict_probs(predictions, proposals)
                    instance.pred_scores = scores[idx]
                    instance.pred_classes = torch.argmax(scores[idx][:,:-1], -1)
                    if self.training:
                        instance.mask_pred_classes = targets[idx].gt_classes.clone().detach()
                else:
                    instance.pred_classes = targets[idx].gt_classes.clone().detach()
                    instance.pred_scores = torch.zeros(instance.pred_boxes.tensor.size(0), self.num_classes + 1).to(instance.pred_boxes.device).scatter(1, instance.pred_classes.unsqueeze(1), 1.)
                outputs.append(instance)
            return outputs, None
        else:
            
            scores = self.predict_probs(predictions, proposals)
            boxes = self.predict_boxes(predictions, proposals)
            image_shapes = [x.image_size for x in proposals]
            nms_thresh = self.test_nms_thresh
            if self.training:
                nms_thresh = self.train_nms_thresh
            results, indices = fast_rcnn_inference(
                        boxes,
                        scores,
                        image_shapes,
                        self.test_score_thresh,
                        nms_thresh,
                        self.test_topk_per_image,#should be 80
                        self.nms_filter_duplicates,
                    )
            if self.training:
                for i, (result, index) in enumerate(zip(results, indices)):
                    result.gt_classes = proposals[i].gt_classes[index]
                #Do post processing here to remove duplicate

            return results, indices


class FastRCNNOutputLayersSGEnd2End(FastRCNNOutputLayers):
    """
    Two linear layers for predicting Fast R-CNN outputs:
    1. proposal-to-detection box regression deltas
    2. classification scores
    """

    @configurable
    def __init__(self, input_shape, *, box2box_transform, num_classes, num_output_classes,
                 test_score_thresh=0.0, test_nms_thresh=0.5, train_nms_thresh=0.7, test_topk_per_image=100,
                 cls_agnostic_bbox_reg=False, smooth_l1_beta=0.0, box_reg_loss_type="smooth_l1", loss_weight=1.0,
                 use_gt_box=True, use_gt_object_label=True, nms_filter_duplicates=True, mask_on=False):
        super(FastRCNNOutputLayersSGEnd2End, self).__init__(input_shape=input_shape, box2box_transform=box2box_transform,
                                                            num_classes=num_classes, test_score_thresh=test_score_thresh,
                                                            test_nms_thresh=test_nms_thresh,
                                                            test_topk_per_image=test_topk_per_image,
                                                            cls_agnostic_bbox_reg=cls_agnostic_bbox_reg,
                                                            smooth_l1_beta=smooth_l1_beta, box_reg_loss_type=box_reg_loss_type,
                                                            loss_weight=loss_weight)
        self.mask_on = mask_on
        self.use_gt_box = use_gt_box
        self.use_gt_object_label = use_gt_object_label
        self.num_classes = num_classes
        self.nms_filter_duplicates = nms_filter_duplicates
        self.train_nms_thresh = train_nms_thresh

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            # fmt: off
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "num_output_classes": cfg.MODEL.ROI_HEADS.NUM_OUTPUT_CLASSES,
            "cls_agnostic_bbox_reg": cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "smooth_l1_beta": cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh": cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh": cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "box_reg_loss_type": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE,
            "loss_weight": {"loss_box_reg": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT},
            "use_gt_box": cfg.MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_BOX,
            "use_gt_object_label": cfg.MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_OBJECT_LABEL,
            "nms_filter_duplicates": cfg.MODEL.ROI_SCENEGRAPH_HEAD.NMS_FILTER_DUPLICATES,
            "mask_on": cfg.MODEL.MASK_ON
            # fmt: on
        }

    def inference(self, proposals, predictions=None, targets=None, gt_proposal_indices=None, add_gt_instances_to_inference_predictions_in_train=False):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.
        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        #        if self.use_gt_box:
        #            outputs = []
        #            for idx, proposal in enumerate(proposals):
        #                instance = Instances(proposal.image_size)
        #                instance.pred_boxes = Boxes(proposal.proposal_boxes.tensor.clone().detach())
        #                if not self.use_gt_object_label:
        #                    scores = self.predict_probs(predictions, proposals)
        #                    instance.pred_scores = scores[idx]
        #                    instance.pred_classes = torch.argmax(scores[idx][:, :-1], -1)
        #                    if self.training:
        #                        instance.mask_pred_classes = targets[idx].gt_classes.clone().detach()
        #                else:
        #                    instance.pred_classes = targets[idx].gt_classes.clone().detach()
        #                    instance.pred_scores = torch.zeros(instance.pred_boxes.tensor.size(0), self.num_classes + 1).to(
        #                        instance.pred_boxes.device).scatter(1, instance.pred_classes.unsqueeze(1), 1.)
        #                outputs.append(instance)
        #            return outputs, None
        #        else:


        if add_gt_instances_to_inference_predictions_in_train and self.training:
            assert targets is not None
            additional_proposals = []
            for idx, target in enumerate(targets):
                additional_instance = Instances(target.image_size)
                additional_instance.proposal_boxes = Boxes(target.gt_boxes.tensor.detach().clone())
                additional_proposals.append(additional_instance)

            #additional_outputs = []
            additional_scores_per_img = []
            additional_boxes_per_img = []
            for idx, additional_proposal in enumerate(additional_proposals):
                additional_instance = Instances(additional_proposal.image_size)
                additional_instance.pred_boxes = Boxes(additional_proposal.proposal_boxes.tensor.clone().detach())
                additional_instance.pred_classes = targets[idx].gt_classes.clone().detach()
                additional_instance.pred_scores = torch.zeros(additional_instance.pred_boxes.tensor.size(0), self.num_classes + 1).to(
                    additional_instance.pred_boxes.device).scatter(1, additional_instance.pred_classes.unsqueeze(1), 0.85) #Set score of added GT predictions to 0.85 such that they can be replaced by very confident predictions by detector (add some bbox variation to relation training)
                #additional_outputs.append(additional_instance)

                additional_boxes = additional_proposal.proposal_boxes.tensor.clone().detach().repeat(1, self.num_classes) #add dummy repetitions for box predictions. in the end, only the correcct one will be chosen, based on the 1.0 score
                additional_scores = torch.zeros(additional_instance.pred_boxes.tensor.size(0), self.num_classes + 1).to(
                    additional_instance.pred_boxes.device).scatter(1, additional_instance.pred_classes.unsqueeze(1), 0.85)

                additional_boxes_per_img.append(additional_boxes)
                additional_scores_per_img.append(additional_scores)


        scores = self.predict_probs(predictions, proposals)
        boxes = self.predict_boxes(predictions, proposals)


        if add_gt_instances_to_inference_predictions_in_train and self.training:
            new_scores = []
            new_boxes = []
            for scores_for_img, additional_scores_for_img in zip(list(scores), additional_scores_per_img):
                new_scores_for_img =  torch.cat((scores_for_img, additional_scores_for_img), dim=0)
                new_scores.append(new_scores_for_img)
            for boxes_for_img, additional_boxes_for_img in zip(list(boxes), additional_boxes_per_img):
                new_boxes_for_img =  torch.cat((boxes_for_img, additional_boxes_for_img), dim=0)
                new_boxes.append(new_boxes_for_img)
            boxes = tuple(new_boxes)
            scores = tuple(new_scores)

            new_proposals = add_ground_truth_to_proposals(targets, proposals)
            proposals = new_proposals

        #image_shapes = [x.image_size for x in proposals]
        image_shapes = [x.image_size for x in proposals]
        nms_thresh = self.test_nms_thresh
        if self.training:
            nms_thresh = self.train_nms_thresh
        results, indices = fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            nms_thresh,
            self.test_topk_per_image,  # should be 80
            self.nms_filter_duplicates,
        )
        if self.training:
            for i, (result, index) in enumerate(zip(results, indices)):
                result.gt_classes = proposals[i].gt_classes[index]
                if self.mask_on:
                    result.gt_masks = proposals[i].gt_masks[index]
            # Do post processing here to remove duplicate

        return results, indices

    def append_predictions_from_gt(self, predictions, targets):

        for idx, target in enumerate(targets):
            instance = Instances(target.image_size)
            instance.proposal_boxes = Boxes(target.gt_boxes.tensor.detach().clone())
            instance.pred_boxes = Boxes(target.gt_boxes.tensor.clone().detach())
            instance.pred_classes = targets[idx].gt_classes.clone().detach()
            instance.pred_scores = torch.zeros(instance.pred_boxes.tensor.size(0), self.num_classes + 1).to(
                instance.pred_boxes.device).scatter(1, instance.pred_classes.unsqueeze(1), 1.)
            #NOTE: this could lead to duplicate prediction instances
            predictions[idx] = Instances.cat([predictions[idx], instance])
        #return outputs, None


class FastRCNNOutputLayersSGMaskTransfer(FastRCNNOutputLayers):
    """
    Two linear layers for predicting Fast R-CNN outputs:
    1. proposal-to-detection box regression deltas
    2. classification scores
    """
    @configurable
    def __init__(self, input_shape, *, box2box_transform, num_classes, mask_num_classes, num_output_classes, test_score_thresh=0.0, test_nms_thresh=0.5, test_topk_per_image=100, cls_agnostic_bbox_reg=False, smooth_l1_beta=0.0,box_reg_loss_type="smooth_l1", loss_weight=1.0, use_gt_box=True, use_gt_object_label=True, use_only_fg_proposals=True):
        super(FastRCNNOutputLayersSGMaskTransfer, self).__init__(input_shape=input_shape, box2box_transform=box2box_transform, num_classes=num_classes, test_score_thresh=test_score_thresh, test_nms_thresh=test_nms_thresh, test_topk_per_image=test_topk_per_image, cls_agnostic_bbox_reg=cls_agnostic_bbox_reg, smooth_l1_beta=smooth_l1_beta, box_reg_loss_type=box_reg_loss_type, loss_weight=loss_weight)
        self.use_gt_box = use_gt_box
        self.use_gt_object_label = use_gt_object_label
        del self.cls_score
        del self.bbox_pred
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)       
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        mask_num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else mask_num_classes
        box_dim = len(box2box_transform.weights)

        self.num_classes = num_classes 
        self.mask_num_classes = mask_num_classes
        self.cls_score = Linear(input_size, mask_num_classes + 1)
        self.bbox_pred = Linear(input_size, mask_num_bbox_reg_classes * box_dim)
        self.transfer_cls_score = Linear(input_size, num_classes + 1)
        self.transfer_bbox_pred = Linear(input_size, num_bbox_reg_classes * box_dim)
        self.use_only_fg_proposals = use_only_fg_proposals

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.normal_(self.transfer_cls_score.weight, std=0.01)
        nn.init.normal_(self.transfer_bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred, self.transfer_cls_score, self.transfer_bbox_pred]:
            nn.init.constant_(l.bias, 0)
        
        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image
        self.box_reg_loss_type = box_reg_loss_type
        if isinstance(loss_weight, float):
            loss_weight = {"loss_cls": loss_weight, "loss_box_reg": loss_weight}
        self.loss_weight = loss_weight

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
        "input_shape": input_shape,
        "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
        # fmt: off
        "num_classes"           : cfg.MODEL.ROI_HEADS.NUM_CLASSES,
        "mask_num_classes"      : cfg.MODEL.ROI_HEADS.MASK_NUM_CLASSES,
        "num_output_classes"    : cfg.MODEL.ROI_HEADS.NUM_OUTPUT_CLASSES,
        "cls_agnostic_bbox_reg" : cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
        "smooth_l1_beta"        : cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
        "test_score_thresh"     : cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
        "test_nms_thresh"       : cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
        "test_topk_per_image"   : cfg.TEST.DETECTIONS_PER_IMAGE,
        "box_reg_loss_type"     : cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE,
        "loss_weight"           : {"loss_box_reg": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT},
        "use_gt_box"            : cfg.MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_BOX,
        "use_gt_object_label"   : cfg.MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_OBJECT_LABEL,
        "use_only_fg_proposals" : cfg.MODEL.ROI_SCENEGRAPH_HEAD.USE_ONLY_FG_PROPOSALS
        # fmt: on
                }

    def forward(self, x, is_transfer=True):
        """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.
        Returns:
            (Tensor, Tensor):
            First tensor: shape (N,K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class.
            Second tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4),
            or (N,4) for class-agnostic regression.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        if not is_transfer:
            scores = self.cls_score(x)
            proposal_deltas = self.bbox_pred(x)
        else:
            scores = self.transfer_cls_score(x)
            proposal_deltas = self.transfer_bbox_pred(x)
        return scores, proposal_deltas

    def inference(self, proposals, predictions=None, targets=None, segmentation_step=False):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.
        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        if self.use_gt_box:
            outputs = []
            if not self.use_gt_object_label:
                scores = self.predict_probs(predictions, proposals)
            for idx, proposal in enumerate(proposals):
                instance = Instances(proposal.image_size)
                instance.pred_boxes = Boxes(proposal.proposal_boxes.tensor.clone().detach())
                if segmentation_step and self.training:
                    if proposal.has('gt_boxes'):
                        instance.gt_boxes = Boxes(proposal.gt_boxes.tensor.clone().detach())
                    if proposal.has('gt_masks'):
                        instance.gt_masks = copy.deepcopy(proposal.gt_masks)
                    if proposal.has('gt_classes'):
                        instance.gt_classes = proposal.gt_classes.clone().detach()
                if not self.use_gt_object_label:
                    if segmentation_step:
                        try:
                            instance.scores = torch.gather(scores[idx], 1, targets[idx].gt_classes.unsqueeze(1)).squeeze(1)
                        except:
                            instance.scores = torch.gather(scores[idx], 1, torch.argmax(scores[idx][:,:-1], -1).unsqueeze(1)).squeeze(1)
                    instance.pred_scores = scores[idx]
                    try:
                        instance.pred_classes = torch.argmax(scores[idx][:,:-1], -1)
                    except:
                        instance.pred_classes = scores[idx]
                    if self.training:
                        instance.mask_pred_classes = targets[idx].gt_classes.clone().detach()
                else:
                    try:
                        instance.pred_classes = targets[idx].gt_classes.clone().detach()
                        instance.pred_scores = torch.zeros(instance.pred_boxes.tensor.size(0), self.num_classes + 1).to(instance.pred_boxes.device).scatter(1, instance.pred_classes.unsqueeze(1), 1.)
                        if segmentation_step:
                            instance.scores = torch.ones(len(instance.pred_boxes)).to(targets[idx].gt_classes.device)
                    except:
                        instance.pred_classes = proposal.pred_classes
                        instance.pred_scores = proposal.pred_scores
                        if segmentation_step:
                            instance.scores = proposal.scores                    
                outputs.append(instance)
            return outputs, None
        else:
            if segmentation_step and self.training and self.use_only_fg_proposals:
                proposals, fg_indices = select_foreground_proposals(proposals, self.mask_num_classes)
                fg_indices = torch.cat(fg_indices, 0)
                predictions = (predictions[0][fg_indices], predictions[1][fg_indices])
            scores = self.predict_probs(predictions, proposals)
            boxes = self.predict_boxes(predictions, proposals)
            image_shapes = [x.image_size for x in proposals]
            if segmentation_step and self.training and self.use_only_fg_proposals:
                results = []
                indices = []
                for idx, proposal in enumerate(proposals):
                    instance = Instances(image_shapes[idx])
                    instance.pred_boxes = Boxes(proposal.proposal_boxes.tensor.clone().detach())
                    instance.pred_scores = scores[idx]
                    instance.scores = torch.gather(scores[idx], 1, torch.argmax(scores[idx][:,:-1], -1).unsqueeze(1)).squeeze(1)
                    instance.pred_classes = torch.argmax(scores[idx][:,:-1], -1)
                    instance.gt_classes = proposal.gt_classes
                    if segmentation_step:
                        if proposal.has('gt_boxes'):
                            instance.gt_boxes = Boxes(proposal.gt_boxes.tensor.clone().detach())
                        if proposal.has('gt_masks'):
                            instance.gt_masks = copy.deepcopy(proposal.gt_masks)
                    results.append(instance)
            else:
                results, indices =  fast_rcnn_inference(
                            boxes,
                            scores,
                            image_shapes,
                            self.test_score_thresh,
                            self.test_nms_thresh,
                            self.test_topk_per_image,
                            nms_filter_duplicates=False
                        )
                if self.training:
                    for i, (result, index) in enumerate(zip(results, indices)):
                        result.gt_classes = proposals[i].gt_classes[index]
                        if segmentation_step:
                            if proposals[i].has('gt_boxes'):
                                result.gt_boxes = Boxes(proposals[i].gt_boxes.tensor[index].clone().detach())
                            if proposals[i].has('gt_masks'):
                                result.gt_masks = copy.deepcopy(proposals[i].gt_masks[index])
            return results, indices

class FastRCNNOutputLayerswithCOCO(FastRCNNOutputLayers):
    @configurable
    def __init__(
        self,
        input_shape,
        *,
        box2box_transform,
        num_classes,
        mask_num_classes,
        test_score_thresh = 0.0,
        test_nms_thresh = 0.5,
        test_topk_per_image = 100,
        cls_agnostic_bbox_reg = False,
        smooth_l1_beta = 0.0,
        box_reg_loss_type = "smooth_l1",
        loss_weight = 1.0,
    ):
        super(FastRCNNOutputLayerswithCOCO, self).__init__(input_shape=input_shape, box2box_transform=box2box_transform, num_classes=num_classes, test_score_thresh=test_score_thresh, test_nms_thresh=test_nms_thresh, test_topk_per_image=test_topk_per_image, cls_agnostic_bbox_reg=cls_agnostic_bbox_reg, smooth_l1_beta=smooth_l1_beta, box_reg_loss_type=box_reg_loss_type, loss_weight=loss_weight)
        del self.cls_score
        del self.bbox_pred
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)       
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        mask_num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else mask_num_classes
        box_dim = len(box2box_transform.weights)

        self.num_classes = num_classes 
        self.mask_num_classes = mask_num_classes
        self.cls_score = Linear(input_size, mask_num_classes + 1)
        self.bbox_pred = Linear(input_size, mask_num_bbox_reg_classes * box_dim)
        self.transfer_cls_score = Linear(input_size, num_classes + 1)
        self.transfer_bbox_pred = Linear(input_size, num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.normal_(self.transfer_cls_score.weight, std=0.01)
        nn.init.normal_(self.transfer_bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred, self.transfer_cls_score, self.transfer_bbox_pred]:
            nn.init.constant_(l.bias, 0)
        
        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image
        self.box_reg_loss_type = box_reg_loss_type
        if isinstance(loss_weight, float):
            loss_weight = {"loss_cls": loss_weight, "loss_box_reg": loss_weight}
        self.loss_weight = loss_weight

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            # fmt: off
            "num_classes"           : cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "mask_num_classes"      : cfg.MODEL.ROI_HEADS.MASK_NUM_CLASSES,
            "cls_agnostic_bbox_reg" : cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "smooth_l1_beta"        : cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh"     : cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh"       : cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image"   : cfg.TEST.DETECTIONS_PER_IMAGE,
            "box_reg_loss_type"     : cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE,
            "loss_weight"           : {"loss_box_reg": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT},
            # fmt: on
        }

    def forward(self, x, is_transfer=True):
        """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.
        Returns:
            (Tensor, Tensor):
            First tensor: shape (N,K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class.
            Second tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4),
            or (N,4) for class-agnostic regression.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        if not is_transfer:
            scores = self.cls_score(x)
            proposal_deltas = self.bbox_pred(x)
        else:
            scores = self.transfer_cls_score(x)
            proposal_deltas = self.transfer_bbox_pred(x)
        return scores, proposal_deltas
