import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from detectron2.config import configurable
from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from detectron2.structures import Instances
from detectron2.modeling.roi_heads.mask_head import MaskRCNNConvUpsampleHead, ROI_MASK_HEAD_REGISTRY
from detectron2.modeling.roi_heads.mask_head import mask_rcnn_inference, mask_rcnn_loss
from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from detectron2.utils.events import get_event_storage
from detectron2.structures import Boxes, Instances
from detectron2.modeling.roi_heads.roi_heads import select_foreground_proposals
import copy

@ROI_MASK_HEAD_REGISTRY.register()
class SceneGraphMaskHeadAllClasses(MaskRCNNConvUpsampleHead):
    def forward(self, x, pred_instances):
        x = self.layers(x)

        mask_probs_pred = x.sigmoid()
        num_boxes_per_image = [len(i) for i in pred_instances]
        mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)

        for prob, instances in zip(mask_probs_pred, pred_instances):
            instances.pred_masks = prob  # (1, Hmask, Wmask)

        return pred_instances

@ROI_MASK_HEAD_REGISTRY.register()
class SceneGraphMaskHead(MaskRCNNConvUpsampleHead):
    def forward(self, x, instances):
        x = self.layers(x)
        mask_rcnn_inference(x, instances)
        return instances


@ROI_MASK_HEAD_REGISTRY.register()
class SceneGraphMaskHeadEnd2End(MaskRCNNConvUpsampleHead):
    """
    SceneGraphMaskHeadEnd2End built using a mix from SceneGraphMaskHeadTransfer and its child-class SGSceneGraphMaskHead
    """

    # TODO-ROY: Verify and review this implementation (try to generalize it to non-class agnostic masks)

    @configurable
    def __init__(self, input_shape, *, num_classes, conv_dims, conv_norm="", **kwargs):
        self.use_only_fg_proposals = kwargs['use_only_fg_proposals']
        self.num_classes = num_classes
        del kwargs['use_only_fg_proposals']
        super(SceneGraphMaskHeadEnd2End, self).__init__(input_shape=input_shape, num_classes=num_classes,
                                                   conv_dims=conv_dims, conv_norm=conv_norm, **kwargs)
        # TODO-ROY: Do I need the following (i.e. overwrite init in MaskRCNNConvUpsampleHead)?
        #  ... (from SGSceneGraphMaskHead)
        # nn.init.constant_(self.predictor.weight, 0)
        # if self.predictor.bias is not None:
        #     nn.init.constant_(self.predictor.bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        if cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK:
            ret["num_classes"] = 1
        else:
            ret["num_classes"] = cfg.MODEL.ROI_HEADS.MASK_NUM_CLASSES
        ret['use_only_fg_proposals'] = cfg.MODEL.ROI_SCENEGRAPH_HEAD.USE_ONLY_FG_PROPOSALS
        return ret

    def forward(self, x, proposals, return_masks=False):
        """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`.
            proposals (list[Instances]): boxes & labels corresponding to the input features.

        Returns:
            predicted instances and loss-dict
        """

        # TODO: was (not self.use_only_fg_proposals), but seems wrong
        if self.use_only_fg_proposals and self.training:
            proposals, fg_indices = select_foreground_proposals(proposals, self.num_classes)
            fg_indices = torch.cat(fg_indices, 0)
            x = x[fg_indices]

        # predict mask-logits
        x = self.layers(x)
        if return_masks:
            return x

        if self.training:
            for proposal in proposals:
                if not proposal.has('proposal_boxes'):
                    proposal.proposal_boxes = Boxes(proposal.pred_boxes.tensor.detach().clone())
            loss = mask_rcnn_loss(x, proposals)
            if torch.any(torch.isnan(loss)):
                loss = torch.sum(x) * 0.0
            # TODO-Generalize: mask_rcnn_inference requires a predicted class,
            #  currently works with a class-agnostic mask
            # SIDENOTE: Maybe was caused by a bug in the box-predictor that did not pass along everything
            mask_rcnn_inference(x, proposals)

            # rename proposal_boxes to pred_boxes
            for proposal in proposals:
                if not proposal.has('pred_boxes'):
                    proposal.set('pred_boxes', Boxes(proposal.proposal_boxes.tensor.detach().clone()))
                    proposal.remove('proposal_boxes')

            return proposals, {"loss_mask_segmentation": loss}
        else:
            mask_rcnn_inference(x, proposals)

            # rename proposal_boxes to pred_boxes
            for proposal in proposals:
                if not proposal.has('pred_boxes'):
                    proposal.set('pred_boxes', Boxes(proposal.proposal_boxes.tensor.detach().clone()))
                    proposal.remove('proposal_boxes')

            return proposals, {}

@ROI_MASK_HEAD_REGISTRY.register()
class SceneGraphMaskHeadTransfer(MaskRCNNConvUpsampleHead):
    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        if cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK:
            ret["num_classes"] = 1
        else:
            ret["num_classes"] = cfg.MODEL.ROI_HEADS.MASK_NUM_CLASSES
        return ret
    
    def forward(self, x, pred_instances, similarity_matrix, base_class_indexer, novel_class_indexer, output_to_coco_indexer, segmentation_step=False, residual_masks=None):
        x = self.layers(x)
        if residual_masks is not None:
            x = x + residual_masks
        if not segmentation_step:
            #Get mask for output class
            base_class_mask = x.index_select(1, output_to_coco_indexer)
            novel_class_mask = torch.bmm(similarity_matrix.unsqueeze(0).expand(x.size(0),-1,-1), x.view(*x.size()[:2],-1)).view(x.size(0), -1, *x.size()[2:])
            output_class_mask = torch.zeros(x.size(0), base_class_mask.size(1) + novel_class_mask.size(1), *x.size()[2:]).to(x.device)
            output_class_mask = output_class_mask.index_copy(1, base_class_indexer, base_class_mask)
            output_class_mask = output_class_mask.index_copy(1, novel_class_indexer, novel_class_mask) 
            mask_probs_pred = output_class_mask.sigmoid()
        else:
            mask_probs_pred = x.sigmoid()
        num_boxes_per_image = [len(i) for i in pred_instances]
        mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)
        for prob, instances in zip(mask_probs_pred, pred_instances):
            instances.pred_masks = prob  # (1, Hmask, Wmask)
            if not self.training:
                instances.pred_masks_base = prob.detach().clone()

        # if not self.training:
        #     pred_mask_logits = pred_instances[0].pred_masks_base
        #     num_masks = pred_mask_logits.shape[0]
        #     class_pred = cat([i.pred_classes for i in pred_instances])
        #     indices = torch.arange(num_masks, device=class_pred.device)
        #     mask_probs_pred_copy = pred_mask_logits[indices, class_pred][:, None]
        #     num_boxes_per_image = [len(i) for i in pred_instances]
        #     mask_probs_pred_copy = mask_probs_pred_copy.split(num_boxes_per_image, dim=0)
        #     for prob, instances in zip(mask_probs_pred_copy, pred_instances):
        #         instances.pred_masks_base = prob  # (1, Hmask, Wmask)
        return pred_instances

@ROI_MASK_HEAD_REGISTRY.register()
class SceneGraphMaskHeadTransferSingleClass(MaskRCNNConvUpsampleHead):
    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        if cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK:
            ret["num_classes"] = 1
        else:
            ret["num_classes"] = cfg.MODEL.ROI_HEADS.MASK_NUM_CLASSES
        return ret
    
    def forward(self, x, pred_instances, similarity_matrix, base_class_indexer, novel_class_indexer, output_to_coco_indexer, segmentation_step=False, residual_masks=None):
        x = self.layers(x)
        if residual_masks is not None:
            x = x + residual_masks
        if not segmentation_step:
            #Get mask for output class
            base_class_mask = x.index_select(1, output_to_coco_indexer)
            novel_class_mask = torch.bmm(similarity_matrix.unsqueeze(0).expand(x.size(0),-1,-1), x.view(*x.size()[:2],-1)).view(x.size(0), -1, *x.size()[2:])
            output_class_mask = torch.zeros(x.size(0), base_class_mask.size(1) + novel_class_mask.size(1), *x.size()[2:]).to(x.device)
            output_class_mask = output_class_mask.index_copy(1, base_class_indexer, base_class_mask)
            output_class_mask = output_class_mask.index_copy(1, novel_class_indexer, novel_class_mask) 
            mask_rcnn_inference(output_class_mask, pred_instances)
        else:
            try:
                mask_rcnn_inference(x, pred_instances)
            except:
                if not self.training:
                    pred_instances[0].pred_masks = torch.zeros_like(x).narrow(1, 0, 1)
                else:
                    pass
        return pred_instances


@ROI_MASK_HEAD_REGISTRY.register()
class SGSceneGraphMaskHead(SceneGraphMaskHeadTransfer):
    @configurable
    def __init__(self, input_shape, *, num_classes, conv_dims, conv_norm="", **kwargs):
        self.use_only_fg_proposals = kwargs['use_only_fg_proposals']
        self.num_classes = num_classes
        del kwargs['use_only_fg_proposals']
        super(SGSceneGraphMaskHead, self).__init__(input_shape=input_shape, num_classes=num_classes, conv_dims=conv_dims, conv_norm=conv_norm, **kwargs)
        nn.init.constant_(self.predictor.weight, 0)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        if cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK:
            ret["num_classes"] = 1
        else:
            ret["num_classes"] = cfg.MODEL.ROI_HEADS.MASK_NUM_CLASSES
        ret['use_only_fg_proposals'] = cfg.MODEL.ROI_SCENEGRAPH_HEAD.USE_ONLY_FG_PROPOSALS
        return ret

    def forward(self, x, masks, proposals, eps=1e-8, return_masks=False):
        masks = torch.cat(masks)
        masks = -1 * torch.log((1.0 / (masks + eps)) - 1)
        if (not self.use_only_fg_proposals) and self.training:
            proposals, fg_indices = select_foreground_proposals(proposals, self.num_classes)
            fg_indices = torch.cat(fg_indices, 0)
            x = x[fg_indices]
            masks = masks[fg_indices]
        x = self.layers(x)
        if return_masks:
            return x
        
        combined_masks = x + masks
        if self.training:
            for proposal in proposals:
                if not proposal.has('proposal_boxes'):
                    proposal.proposal_boxes = Boxes(proposal.pred_boxes.tensor.detach().clone())
            loss = mask_rcnn_loss(combined_masks, proposals)
            if torch.any(torch.isnan(loss)):
                loss = torch.sum(x) * 0.0
            return {"refine_loss_mask_segmentation" : loss}, proposals
        else:
            mask_rcnn_inference(combined_masks, proposals)
            return {}, proposals



@ROI_MASK_HEAD_REGISTRY.register()
class MaskLabelRCNNHead(MaskRCNNConvUpsampleHead):
    def forward(self, x, instances, similarity_matrix, base_class_indexer, novel_class_indexer, output_to_coco_indexer):
        x = self.layers(x)
        #Get mask for output class
        base_class_mask = x.index_select(1, output_to_coco_indexer)
        novel_class_mask = torch.bmm(similarity_matrix.unsqueeze(0).expand(x.size(0),-1,-1), x.view(*x.size()[:2],-1)).view(x.size(0), -1, *x.size()[2:])
        output_class_mask = torch.zeros(x.size(0), base_class_mask.size(1) + novel_class_mask.size(1), *x.size()[2:]).to(x.device)
        output_class_mask = output_class_mask.index_copy(1, base_class_indexer, base_class_mask)
        output_class_mask = output_class_mask.index_copy(1, novel_class_indexer, novel_class_mask)

        if self.training:
            raise NotImplementedError
        else:
            mask_rcnn_inference(output_class_mask, instances)
            return instances

@ROI_MASK_HEAD_REGISTRY.register()
class PretrainObjectDetectionMaskHead(MaskRCNNConvUpsampleHead):
    def forward(self, x, instances):
        """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`.
            instances (list[Instances]): contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.
        Returns:
            A dict of losses in training. The predicted "instances" in inference.
        """
        x = self.layers(x)
        if self.training:
            return {"loss_mask": mask_rcnn_loss_with_empty_polygons(x, instances, self.vis_period)}
        else:
            mask_rcnn_inference(x, instances)
            return instances

@ROI_MASK_HEAD_REGISTRY.register()
class MaskRCNNConvUpsampleHeadwithCOCO(MaskRCNNConvUpsampleHead):
    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        conv_dim = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        num_conv = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        ret.update(
            conv_dims=[conv_dim] * (num_conv + 1),  # +1 for ConvTranspose
            conv_norm=cfg.MODEL.ROI_MASK_HEAD.NORM,
            input_shape=input_shape,
        )
        if cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK:
            ret["num_classes"] = 1
        else:
            ret["num_classes"] = cfg.MODEL.ROI_HEADS.MASK_NUM_CLASSES
        return ret

def mask_rcnn_loss_with_empty_polygons(pred_mask_logits, instances, vis_period=0):
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

    gt_classes = []
    gt_masks = []
    gt_masks_nonzero = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        current_gt_masks_nonzero = instances_per_image.gt_masks.nonempty()
        instances_per_image = instances_per_image[current_gt_masks_nonzero]
        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)
        gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len
        ).to(device=pred_mask_logits.device)
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_masks.append(gt_masks_per_image)
        gt_masks_nonzero.append(current_gt_masks_nonzero)

    if len(gt_masks) == 0:
        return pred_mask_logits.sum() * 0

    gt_masks_nonzero = cat(gt_masks_nonzero, dim=0)
    gt_masks = cat(gt_masks, dim=0)
    pred_mask_logits = pred_mask_logits[gt_masks_nonzero]
    total_num_masks = pred_mask_logits.size(0)
    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]

    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
        gt_masks_bool = gt_masks > 0.5
    gt_masks = gt_masks.to(dtype=torch.float32)

    # Log the training accuracy (using gt classes and 0.5 threshold)
    mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
    mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
    num_positive = gt_masks_bool.sum().item()
    false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
        gt_masks_bool.numel() - num_positive, 1.0
    )
    false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(num_positive, 1.0)

    storage = get_event_storage()
    storage.put_scalar("mask_rcnn/accuracy", mask_accuracy)
    storage.put_scalar("mask_rcnn/false_positive", false_positive)
    storage.put_scalar("mask_rcnn/false_negative", false_negative)
    if vis_period > 0 and storage.iter % vis_period == 0:
        pred_masks = pred_mask_logits.sigmoid()
        vis_masks = torch.cat([pred_masks, gt_masks], axis=2)
        name = "Left: mask prediction;   Right: mask GT"
        for idx, vis_mask in enumerate(vis_masks):
            vis_mask = torch.stack([vis_mask] * 3, axis=0)
            storage.put_image(name + f" ({idx})", vis_mask)

    mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_masks, reduction="mean")
    return mask_loss
