import sys
import os
import torch
import logging
import numpy as np
from detectron2.modeling import META_ARCH_REGISTRY, GeneralizedRCNN
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.config import configurable
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.layers import paste_masks_in_image
from detectron2.structures import Instances, Boxes
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.modeling.backbone import Backbone, build_backbone
from ..backbone import *
import cv2
from detectron2.utils.events import get_event_storage

@META_ARCH_REGISTRY.register()
class SceneGraphRCNN(GeneralizedRCNN):
    @configurable
    def __init__(self, *, backbone, proposal_generator, roi_heads, pixel_mean, pixel_std, input_format=None, vis_period=0, freeze_layers=[]):
        super(SceneGraphRCNN, self).__init__(backbone=backbone, proposal_generator=proposal_generator, roi_heads=roi_heads, pixel_mean=pixel_mean, pixel_std=pixel_std, input_format=input_format, vis_period=vis_period)
        self._freeze_layers(layers=freeze_layers)

    def _freeze_layers(self, layers):
        # Freeze layers
        for name, param in self.named_parameters():
            if any(layer == name.split(".")[0] for layer in layers):
                logging.getLogger('detectron2').log(logging.WARN, "Freezed Layer: {}".format(name))
                param.requires_grad = False

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret['freeze_layers'] = cfg.MODEL.FREEZE_LAYERS.META_ARCH
        return ret

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            gt_relations = [x["relations"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
            gt_relations = None

        features = self.backbone(images.tensor)

        if self.proposal_generator:
            proposals, _ = self.proposal_generator(images, features, gt_instances)
            proposal_losses = {}
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}
        
        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, gt_relations)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.
        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.
        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training
        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            gt_relations = [x["relations"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
            gt_relations = None

        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, gt_instances, gt_relations)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)
        if do_postprocess:
            return self._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    @staticmethod
    def _postprocess(instances, batched_inputs, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            # import pdb; pdb.set_trace()
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r, 
                                      "rel_pair_idxs": results_per_image._rel_pair_idxs,
                                      "pred_rel_scores": results_per_image._pred_rel_scores
                                      })
            if hasattr(results_per_image, '_viz_outputs'):
                processed_results[-1]['viz_outputs'] = SceneGraphRCNN.visualization_preprocess(results_per_image._viz_outputs, results_per_image.image_size, height, width)
        return processed_results

    @staticmethod
    def visualization_preprocess(results, image_size, output_height, output_width, mask_threshold=0.5):
        if isinstance(output_width, torch.Tensor):
            output_width_tmp = output_width.float()
        else:
            output_width_tmp = output_width

        if isinstance(output_height, torch.Tensor):
            output_height_tmp = output_height.float()
        else:
            output_height_tmp = output_height
        scale_x, scale_y = (
            output_width_tmp / image_size[1],
            output_height_tmp / image_size[0],
        )
        for key, value in results.items():
            if 'box' in key:
                results[key] = Boxes(value)
                results[key].scale(scale_x, scale_y)
                results[key].clip((output_height, output_width))
        for key, value in results.items():
            if 'mask' in key or 'attention' in key:
                if 'head_mask' == key:
                    results[key] = retry_if_cuda_oom(paste_masks_in_image)(value[:, 0, :, :], results['head_box'], (output_height, output_width), threshold=mask_threshold)
                elif 'tail_mask' == key:
                    results[key] = retry_if_cuda_oom(paste_masks_in_image)(value[:, 0, :, :], results['tail_box'], (output_height, output_width), threshold=mask_threshold)
                elif 'attention_object1' == key:
                    results[key] = retry_if_cuda_oom(paste_masks_in_image)(value[:, 0, :, :], results['head_box'], (output_height, output_width), threshold=-1)
                elif 'attention_object2' == key:
                    results[key] = retry_if_cuda_oom(paste_masks_in_image)(value[:, 0, :, :], results['tail_box'], (output_height, output_width), threshold=-1)
                else:
                    results[key] = retry_if_cuda_oom(paste_masks_in_image)(value[:, 0, :, :], results['union_box'], (output_height, output_width), threshold=-1)
        return results




@META_ARCH_REGISTRY.register()
class SceneGraphRCNNEnd2End(GeneralizedRCNN):
    @configurable
    def __init__(self, *, backbone, proposal_generator, roi_heads, pixel_mean, pixel_std, input_format=None,
                 vis_period=0, freeze_layers=[]):
        assert freeze_layers == []
        super(SceneGraphRCNNEnd2End, self).__init__(backbone=backbone, proposal_generator=proposal_generator,
                                                    roi_heads=roi_heads, pixel_mean=pixel_mean, pixel_std=pixel_std,
                                                    input_format=input_format, vis_period=vis_period)


    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            gt_relations = [x["relations"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
            gt_relations = None

        features = self.backbone(images.tensor)

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, gt_relations)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.
        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.
        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training
        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            gt_relations = [x["relations"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
            gt_relations = None

        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, gt_instances, gt_relations)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)
        if do_postprocess:
            return self._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results


    @staticmethod
    def _postprocess(instances, batched_inputs, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
                instances, batched_inputs, image_sizes
        ):

            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            # import pdb; pdb.set_trace()
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r,
                                      "rel_pair_idxs": results_per_image._rel_pair_idxs,
                                      "pred_rel_scores": results_per_image._pred_rel_scores
                                      })
            if hasattr(results_per_image, '_viz_outputs'):
                processed_results[-1]['viz_outputs'] = SceneGraphRCNN.visualization_preprocess(
                    results_per_image._viz_outputs, results_per_image.image_size, height, width)
        return processed_results

    @staticmethod
    def visualization_preprocess(results, image_size, output_height, output_width, mask_threshold=0.5):
        if isinstance(output_width, torch.Tensor):
            output_width_tmp = output_width.float()
        else:
            output_width_tmp = output_width

        if isinstance(output_height, torch.Tensor):
            output_height_tmp = output_height.float()
        else:
            output_height_tmp = output_height
        scale_x, scale_y = (
            output_width_tmp / image_size[1],
            output_height_tmp / image_size[0],
        )
        for key, value in results.items():
            if 'box' in key:
                results[key] = Boxes(value)
                results[key].scale(scale_x, scale_y)
                results[key].clip((output_height, output_width))
        for key, value in results.items():
            if 'mask' in key or 'attention' in key:
                if 'head_mask' == key:
                    results[key] = retry_if_cuda_oom(paste_masks_in_image)(value[:, 0, :, :], results['head_box'],
                                                                           (output_height, output_width),
                                                                           threshold=mask_threshold)
                elif 'tail_mask' == key:
                    results[key] = retry_if_cuda_oom(paste_masks_in_image)(value[:, 0, :, :], results['tail_box'],
                                                                           (output_height, output_width),
                                                                           threshold=mask_threshold)
                elif 'attention_object1' == key:
                    results[key] = retry_if_cuda_oom(paste_masks_in_image)(value[:, 0, :, :], results['head_box'],
                                                                           (output_height, output_width), threshold=-1)
                elif 'attention_object2' == key:
                    results[key] = retry_if_cuda_oom(paste_masks_in_image)(value[:, 0, :, :], results['tail_box'],
                                                                           (output_height, output_width), threshold=-1)
                else:
                    results[key] = retry_if_cuda_oom(paste_masks_in_image)(value[:, 0, :, :], results['union_box'],
                                                                           (output_height, output_width), threshold=-1)
        return results


@META_ARCH_REGISTRY.register()
class SceneGraphSegmentationRCNN(SceneGraphRCNN):
    def forward(self, batched_inputs, mask_batched_inputs=None, segmentation_step=False):
        if not self.training:
            return self.inference(batched_inputs, segmentation_step=segmentation_step)
    
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            gt_relations = [x["relations"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
            gt_relations = None

        features = self.backbone(images.tensor)

        if self.proposal_generator:
            proposals, _ = self.proposal_generator(images, features, gt_instances)
            proposal_losses = {}
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        if mask_batched_inputs is not None:
            mask_images = self.preprocess_image(mask_batched_inputs)
            if "instances" in mask_batched_inputs[0]:
                mask_gt_instances = [x["instances"].to(self.device) for x in mask_batched_inputs]
            else:
                mask_gt_instances = None

            mask_features = self.backbone(mask_images.tensor)

            if self.proposal_generator:
                mask_proposals, _ = self.proposal_generator(mask_images, mask_features, mask_gt_instances)
                mask_proposal_losses = {}
            else:
                assert "proposals" in mask_batched_inputs[0]
                mask_proposals = [x["proposals"].to(self.device) for x in mask_batched_inputs]
                mask_proposal_losses = {}

        _, mask_detector_losses = self.roi_heads(mask_images, mask_features, mask_proposals, mask_gt_instances, None, segmentation_step=True)
        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, gt_relations, segmentation_step=False)
    
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses.update(mask_detector_losses)
        return losses

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True, segmentation_step=False):
        """
        Run inference on the given inputs.
        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.
        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training
        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            if not segmentation_step:
                gt_relations = [x["relations"].to(self.device) for x in batched_inputs]
            else:
                gt_relations = None
        else:
            gt_instances = None
            gt_relations = None

        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            results, _ = self.roi_heads(images, features, proposals, gt_instances, gt_relations, segmentation_step=segmentation_step)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            return self._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.
        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.
        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training
        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            gt_relations = [x["relations"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
            gt_relations = None

        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, gt_instances, gt_relations)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)
        if do_postprocess:
            return self._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results


    @staticmethod
    def _postprocess(instances, batched_inputs, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess_modified(results_per_image, height, width)
            try:
                processed_results.append({"instances": r, 
                                      "rel_pair_idxs": results_per_image._rel_pair_idxs,
                                      "pred_rel_scores": results_per_image._pred_rel_scores
                                      })
            except:
                processed_results.append({"instances": r})
            if hasattr(results_per_image, '_viz_outputs'):
                processed_results[-1]['viz_outputs'] = SceneGraphRCNN.visualization_preprocess(results_per_image._viz_outputs, results_per_image.image_size, height, width)
        return processed_results

@META_ARCH_REGISTRY.register()
class MaskLabelRCNN(GeneralizedRCNN):
    @configurable
    def __init__(self, *, backbone, proposal_generator, roi_heads, pixel_mean, pixel_std, input_format=None, vis_period=0):
        super(MaskLabelRCNN, self).__init__(backbone=backbone, proposal_generator=proposal_generator, roi_heads=roi_heads, pixel_mean=pixel_mean, pixel_std=pixel_std, input_format=input_format, vis_period=vis_period)

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)
        raise NotImplementedError
    
    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        proposals = [x["instances"].to(self.device) for x in batched_inputs]
        results, _ = self.roi_heads(images, features, proposals, None)
        if do_postprocess:
            return MaskLabelRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

@META_ARCH_REGISTRY.register()
class GeneralizedRCNNWithCOCO(GeneralizedRCNN):
    @configurable
    def __init__(
        self,
        *,
        backbone,
        proposal_generator,
        roi_heads,
        pixel_mean,
        pixel_std,
        num_classes,
        num_mask_classes,
        input_format = None,
        vis_period = 0,
    ):
        super(GeneralizedRCNNWithCOCO, self).__init__(backbone=backbone, proposal_generator=proposal_generator, roi_heads=roi_heads, pixel_mean=pixel_mean, pixel_std=pixel_std, input_format=input_format, vis_period=vis_period)
        self.num_classes = num_classes
        self.num_mask_classes = num_mask_classes

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret['num_classes'] = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        ret['num_mask_classes'] = cfg.MODEL.ROI_HEADS.MASK_NUM_CLASSES
        return ret

    def forward(self, batched_inputs, mask_batched_inputs = None, mode='sg'):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs, mode=mode)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        features = self.backbone(images.tensor)
        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        if mask_batched_inputs is not None:
            mask_images = self.preprocess_image(mask_batched_inputs)
            if "instances" in mask_batched_inputs[0]:
                mask_gt_instances = [x["instances"].to(self.device) for x in mask_batched_inputs]
            else:
                mask_gt_instances = None
            mask_features = self.backbone(mask_images.tensor)
            if self.proposal_generator is not None:
                mask_proposals, mask_proposal_losses = self.proposal_generator(mask_images, mask_features, mask_gt_instances)
            else:
                assert "proposals" in mask_batched_inputs[0]
                mask_proposals = [x["proposals"].to(self.device) for x in mask_batched_inputs]
                mask_proposal_losses = {}
        self.roi_heads.mask_on = False
        self.roi_heads.num_classes = self.num_classes
        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, is_transfer=True)
        if mask_batched_inputs is not None:
            self.roi_heads.mask_on = True
            self.roi_heads.num_classes = self.num_mask_classes
            _, mask_detector_losses = self.roi_heads(mask_images, mask_features, mask_proposals, mask_gt_instances, is_transfer=False)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        if mask_batched_inputs is not None:
            for key in mask_detector_losses.keys():
                losses['mask_' + key] = mask_detector_losses[key]
            for key in mask_proposal_losses.keys():
                losses['mask_' + key] = mask_proposal_losses[key]
        return losses

    def inference(self, batched_inputs, mode='sg', detected_instances = None, do_postprocess = True):
        assert not self.training
        if mode == 'sg':
            self.roi_heads.mask_on = False
            self.roi_heads.num_classes = self.num_classes
            is_transfer = True
        else:
            self.roi_heads.mask_on = True
            self.roi_heads.num_classes = self.num_mask_classes
            is_transfer = False

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None, is_transfer=is_transfer)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)
        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

@META_ARCH_REGISTRY.register()
class TempRCNN(GeneralizedRCNN):
    def forward(self, batched_inputs):
        return {}
        from PIL import Image
        for idx, _ in enumerate(batched_inputs):
            img = cv2.imread(batched_inputs[idx]["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get('VG_train'), scale=1)
            vis = visualizer.draw_dataset_dict(batched_inputs[idx])
            im = Image.fromarray(vis.get_image())
            im.save('temp_{}.jpeg'.format(idx))
        a = 1

def detector_postprocess_modified(
    results, output_height, output_width, mask_threshold=0.5
):
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.
    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.
    Args:
        results (Instances): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.
    Returns:
        Instances: the resized output from the model, based on the output resolution
    """
    # Change to 'if is_tracing' after PT1.7
    if isinstance(output_height, torch.Tensor):
        # Converts integer tensors to float temporaries to ensure true
        # division is performed when computing scale_x and scale_y.
        output_width_tmp = output_width.float()
        output_height_tmp = output_height.float()
        new_size = torch.stack([output_height, output_width])
    else:
        new_size = (output_height, output_width)
        output_width_tmp = output_width
        output_height_tmp = output_height

    scale_x, scale_y = (
        output_width_tmp / results.image_size[1],
        output_height_tmp / results.image_size[0],
    )
    results = Instances(new_size, **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes
    else:
        output_boxes = None
    assert output_boxes is not None, "Predictions must contain boxes!"

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(results.image_size)

    results = results[output_boxes.nonempty()]

    if results.has("pred_masks"):
        results.pred_masks = retry_if_cuda_oom(paste_masks_in_image)(
            results.pred_masks[:, 0, :, :],  # N, 1, M, M
            results.pred_boxes,
            results.image_size,
            threshold=mask_threshold,
        )

    if results.has("pred_masks_base"):
        results.pred_masks_base = retry_if_cuda_oom(paste_masks_in_image)(
            results.pred_masks_base[:, 0, :, :],  # N, 1, M, M
            results.pred_boxes,
            results.image_size,
            threshold=mask_threshold,
        )

    if results.has("pred_keypoints"):
        results.pred_keypoints[:, :, 0] *= scale_x
        results.pred_keypoints[:, :, 1] *= scale_y

    return results