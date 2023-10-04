import torch
import copy
from torch import nn
from torch.nn import functional as F

from detectron2.utils.registry import Registry
from detectron2.modeling.poolers import ROIPooler
from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes, Instances

from ....structures import boxes_union, masks_union
from .box_feature_extractor import build_box_feature_extractor

ROI_RELATION_FEATURE_EXTRACTORS_REGISTRY = Registry("ROI_RELATION_FEATURE_EXTRACTORS_REGISTRY")

@ROI_RELATION_FEATURE_EXTRACTORS_REGISTRY.register()
class RelationFeatureExtractor(nn.Module):
    '''
    Class containg method to extract feature for edge states
    '''

    def __init__(self, cfg, input_shape):
        cfg.defrost()
        cfg.MODEL.ROI_BOX_FEATURE_EXTRACTORS.BOX_FEATURE_MASK = True
        cfg.MODEL.ROI_BOX_FEATURE_EXTRACTORS.NAME = 'BoxFeatureExtractor'
        cfg.freeze()
        super(RelationFeatureExtractor, self).__init__()
        
        #Feature Extractor(Pools the feature from diffreent scales and converts to matrix of shape num_ojbects x feature_dim)
        self.feature_extractor = build_box_feature_extractor(cfg, input_shape)
        self.out_channels = self.feature_extractor.out_channels

        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        in_channels = [input_shape[f].channels for f in in_features][0]
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES

        self.mask_on = cfg.MODEL.MASK_ON
        self.use_gt_box = cfg.MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_BOX
        self.in_features = in_features
        self.attention_type = cfg.MODEL.ROI_SCENEGRAPH_HEAD.MASK_ATTENTION_TYPE
        self.is_sigmoid = cfg.MODEL.ROI_SCENEGRAPH_HEAD.SIGMOID_ATTENTION
        self.rect_size = pooler_resolution * 4 -1 # size of union bounding box
        # self.rect_conv = nn.Sequential(*[
        #     nn.Conv2d(2, in_channels //2, kernel_size=7, stride=2, padding=3, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(in_channels//2, momentum=0.01),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #     nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(in_channels, momentum=0.01),
        #     ])
        self.use_mask_attention = cfg.MODEL.ROI_SCENEGRAPH_HEAD.USE_MASK_ATTENTION
        self.mask_combiner = cfg.MODEL.ROI_RELATION_FEATURE_EXTRACTORS.USE_MASK_COMBINER
        self.multiply_logits_with_masks = cfg.MODEL.ROI_RELATION_FEATURE_EXTRACTORS.MULTIPLY_LOGITS_WITH_MASKS
        if self.use_mask_attention:
            self.attention_dimension = cfg.MODEL.ROI_BOX_FEATURE_EXTRACTORS.POOLER_RESOLUTION
            if self.attention_type == 'Gaussian':
                self.kernel_size = 7
                self.variance_net = nn.Linear((self.num_classes + 1)*2, 6)
            elif self.attention_type == 'Weighted' or self.attention_type == 'Diff_Channels':
                self.kernel_size = 7
                self.variance_net = nn.Linear((self.num_classes + 1)*2, 6)
                if self.mask_combiner:
                    self.mask_class_combiner = nn.Conv2d(self.num_classes, 1, kernel_size=3, padding=1)
            elif self.attention_type == 'Zero':
                self.kernel_size = 7
            else:
                self.attention = nn.Linear(self.attention_dimension**2 + (self.num_classes + 1)*2, self.attention_dimension**2)
    
    def _init_gaussian_attention(self, variance, eps=1e-7):
        x_cord = torch.arange(-self.kernel_size//2+1, self.kernel_size//2+1)
        x_grid = x_cord.repeat(self.kernel_size).view(self.kernel_size, self.kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).to(variance.device)
    
        mean = 0
        sigma = torch.exp(0.5 * variance.view(-1, 2, variance.size(1)//2).narrow(-1, 0, 2)).clamp(min=eps)
        rho = torch.tanh(variance.view(-1, 2, variance.size(1)//2).narrow(-1, 2, 1))
        sigma = sigma.view(-1, 2)
        rho = rho.view(-1, 1)
        x_by_sigma = xy_grid.view(-1, xy_grid.size(2)).unsqueeze(0) / sigma.unsqueeze(1)
        x_by_sigma_squared = torch.pow(torch.sum(x_by_sigma, -1), 2.0)
        x_multiplied = 2 * torch.prod(x_by_sigma, -1) * (1.0 + rho)
        z = x_by_sigma_squared - x_multiplied
        one_minus_rho = (1.0 - torch.pow(rho, 2)).clamp(max=1.0)
        z_by_rho = z / (2 * one_minus_rho)
        gaussian_kernel = torch.exp(-1 * z_by_rho)
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel, -1, keepdim=True)
        gaussian_kernel = gaussian_kernel.view(variance.size(0), -1, self.kernel_size * self.kernel_size)
        gaussian_kernel = gaussian_kernel.view(gaussian_kernel.size(0), gaussian_kernel.size(1), self.kernel_size, self.kernel_size)
        return gaussian_kernel
        # self.attention.weight.data = gaussian_kernel[None, None, :]
        # self.attention.weight.requires_grad = False
  
    def forward(self, features, boxes, rel_pair_list=None, masks=None, proposals=None, return_seg_masks=False):
        device = boxes[0].device
        objects_per_image = torch.tensor([0] + [len(x) for x in proposals]).to(device)
        objects_per_image_sum = torch.cumsum(objects_per_image, dim=0)
        rel_pair_idx = []
        pred_scores = []
        num_rel_pair_idx = []
        for idx, (box, rel_idx) in enumerate(zip(boxes, rel_pair_list)):
            rel_pair_idx.append(rel_idx + objects_per_image_sum[idx])
            num_rel_pair_idx.append(len(rel_idx))
            pred_scores.append(proposals[idx].pred_scores)
            
        rel_pair_idx = torch.cat(rel_pair_idx, 0)
        num_rel_pair_idx = torch.tensor([0] + num_rel_pair_idx).to(device)
        num_rel_pair_idx_sum = torch.cumsum(num_rel_pair_idx, dim=0)
        boxes = Boxes.cat(boxes)
        head_box = boxes[rel_pair_idx[:, 0]]
        tail_box = boxes[rel_pair_idx[:, 1]]
        union_box = boxes_union(head_box, tail_box)
        if self.mask_on:
            masks = torch.cat(masks, 0)
            head_mask = masks[rel_pair_idx[:, 0]]
            tail_mask = masks[rel_pair_idx[:, 1]]
            if self.use_mask_attention:
                pred_scores = torch.cat(pred_scores, 0)
                head_score = pred_scores[rel_pair_idx[:, 0]]
                tail_score = pred_scores[rel_pair_idx[:, 1]]
                if self.attention_type == 'Gaussian' or self.attention_type == 'Weighted' or self.attention_type == 'Diff_Channels':
                    variance = self.variance_net(torch.cat([head_score, tail_score], -1))
                    gaussian_kernels = self._init_gaussian_attention(variance)
                    if head_mask.size(1) > 1:
                        if self.multiply_logits_with_masks:
                            head_mask = head_mask * head_score.narrow(1, 0, head_mask.size(1)).unsqueeze(2).unsqueeze(3)
                            tail_mask = tail_mask * tail_score.narrow(1, 0, tail_mask.size(1)).unsqueeze(2).unsqueeze(3)
                            print ("REL-NOPE")
                        if head_mask.size(0) > 500:
                            # Do it in chunks
                            union_attention = []
                            head_mask_chunks = torch.split(head_mask, 100, dim=0)
                            tail_mask_chunks = torch.split(tail_mask, 100, dim=0)
                            gaussian_kernel_chunks = torch.split(gaussian_kernels, 100, dim=0)
                            for idx, (head_mask_chunk, tail_mask_chunk, gaussian_kernel_chunk) in enumerate(zip(head_mask_chunks, tail_mask_chunks, gaussian_kernel_chunks)):
                                gaussian_kernel_chunk = gaussian_kernel_chunk.view(-1, 1, self.kernel_size, self.kernel_size)
                                cat_mask_chunk = torch.stack([head_mask_chunk, tail_mask_chunk], 1).view(-1, *head_mask_chunk.size()[1:]).transpose(0,1)
                                union_attention_conv_chunk = nn.functional.conv2d(cat_mask_chunk, gaussian_kernel_chunk, stride=1, padding=self.kernel_size//2, groups=cat_mask_chunk.size(1))
                                union_attention_chunk = torch.prod(union_attention_conv_chunk.transpose(0,1).contiguous().view(head_mask_chunk.size(0), -1, *head_mask_chunk.size()[1:]), 1)
                                union_attention.append(union_attention_chunk)
                            union_attention = torch.cat(union_attention, dim=0)
                        else:
                            gaussian_kernels = gaussian_kernels.view(-1, 1, self.kernel_size, self.kernel_size)
                            cat_mask = torch.stack([head_mask, tail_mask], 1).view(-1, *head_mask.size()[1:]).transpose(0,1)
                            union_attention_conv = nn.functional.conv2d(cat_mask, gaussian_kernels, stride=1, padding=self.kernel_size//2, groups=cat_mask.size(1))
                            union_attention = torch.prod(union_attention_conv.transpose(0,1).view(head_mask.size(0), -1, *head_mask.size()[1:]), 1)
                        if self.mask_combiner:
                            union_attention = self.mask_class_combiner(union_attention)
                    else:
                        gaussian_kernels = gaussian_kernels.view(-1, 1, self.kernel_size, self.kernel_size)
                        cat_mask = torch.stack([head_mask, tail_mask], 1).view(-1, *head_mask.size()[1:]).transpose(0,1)
                        union_attention_conv = nn.functional.conv2d(cat_mask, gaussian_kernels, stride=1, padding=self.kernel_size//2, groups=cat_mask.size(1))
                        union_attention = torch.prod(union_attention_conv.transpose(0,1).view(head_mask.size(0), -1, *head_mask.size()[1:]), 1)
                    if self.is_sigmoid:
                        union_attention = torch.sigmoid(union_attention)
                    union_mask = union_attention
                elif self.attention_type == 'Zero':
                    union_mask = torch.zeros_like(head_mask)
                else:
                    union_attention_features = torch.cat([head_score, tail_score, union_mask.view(union_mask.size(0), -1)], -1)
                    union_attention = torch.sigmoid(self.attention(union_attention_features))
                    union_mask = masks_union(head_mask, tail_mask)
                    union_mask = union_mask * union_attention.view(*union_mask.size())
            if return_seg_masks:
                viz_output = {}
                head_gt_classes = head_score.argmax(-1)
                tail_gt_classes = tail_score.argmax(-1)
                num_masks = union_mask.shape[0]
                indices = torch.arange(num_masks, device=union_mask.device)
                viz_output['head_gt_classes'] = head_gt_classes.clone().detach()
                viz_output['tail_gt_classes'] = tail_gt_classes.clone().detach()
                viz_output['head_mask'] = head_mask[indices, head_gt_classes][:, None]
                viz_output['tail_mask'] = tail_mask[indices, tail_gt_classes][:, None]
                viz_output['union_mask'] = union_mask
                import ipdb; ipdb.set_trace()
        union_boxes = []
        union_masks = [] if self.mask_on else None
        for i in range(num_rel_pair_idx_sum.size(0) - 1):
            union_boxes.append(union_box[num_rel_pair_idx_sum[i]:num_rel_pair_idx_sum[i+1]]) 
            if self.mask_on:
                union_masks.append(union_mask[num_rel_pair_idx_sum[i]:num_rel_pair_idx_sum[i+1]])
        union_features = self.feature_extractor(features, union_boxes, masks=union_masks)
        return union_features, None   

@ROI_RELATION_FEATURE_EXTRACTORS_REGISTRY.register()
class RelationFeatureExtractorNoMask(nn.Module):
    '''
    Class containg method to extract feature for edge states
    '''

    def __init__(self, cfg, input_shape):
        cfg.defrost()
        cfg.MODEL.ROI_BOX_FEATURE_EXTRACTORS.BOX_FEATURE_MASK = False
        cfg.MODEL.ROI_BOX_FEATURE_EXTRACTORS.NAME = 'BoxFeatureExtractor'
        cfg.freeze()
        super(RelationFeatureExtractorNoMask, self).__init__()
        
        #Feature Extractor(Pools the feature from diffreent scales and converts to matrix of shape num_ojbects x feature_dim)
        self.feature_extractor = build_box_feature_extractor(cfg, input_shape)
        self.out_channels = self.feature_extractor.out_channels

        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        in_channels = [input_shape[f].channels for f in in_features][0]
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES

        self.mask_on = cfg.MODEL.MASK_ON
        self.use_gt_box = cfg.MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_BOX
        self.in_features = in_features
        self.attention_type = cfg.MODEL.ROI_SCENEGRAPH_HEAD.MASK_ATTENTION_TYPE
        self.is_sigmoid = cfg.MODEL.ROI_SCENEGRAPH_HEAD.SIGMOID_ATTENTION
        self.rect_size = pooler_resolution * 4 -1 # size of union bounding box
  
    def forward(self, features, boxes, rel_pair_list=None, masks=None, proposals=None, return_seg_masks=False):
        device = boxes[0].device
        objects_per_image = torch.tensor([0] + [len(x) for x in proposals]).to(device)
        objects_per_image_sum = torch.cumsum(objects_per_image, dim=0)
        rel_pair_idx = []
        pred_scores = []
        num_rel_pair_idx = []
        for idx, (box, rel_idx) in enumerate(zip(boxes, rel_pair_list)):
            rel_pair_idx.append(rel_idx + objects_per_image_sum[idx])
            num_rel_pair_idx.append(len(rel_idx))
            pred_scores.append(proposals[idx].pred_scores)
            
        rel_pair_idx = torch.cat(rel_pair_idx, 0)
        num_rel_pair_idx = torch.tensor([0] + num_rel_pair_idx).to(device)
        num_rel_pair_idx_sum = torch.cumsum(num_rel_pair_idx, dim=0)
        boxes = Boxes.cat(boxes)
        head_box = boxes[rel_pair_idx[:, 0]]
        tail_box = boxes[rel_pair_idx[:, 1]]
        union_box = boxes_union(head_box, tail_box)
        union_boxes = []
        union_masks = [] if self.mask_on else None
        for i in range(num_rel_pair_idx_sum.size(0) - 1):
            union_boxes.append(union_box[num_rel_pair_idx_sum[i]:num_rel_pair_idx_sum[i+1]]) 
        union_features = self.feature_extractor(features, union_boxes, masks=None)
        return union_features, None                   

@ROI_RELATION_FEATURE_EXTRACTORS_REGISTRY.register()
class RelationFeatureExtractorAvgMask(nn.Module):
    def __init__(self, cfg, input_shape):
        cfg.defrost()
        cfg.MODEL.ROI_BOX_FEATURE_EXTRACTORS.BOX_FEATURE_MASK = True
        cfg.MODEL.ROI_BOX_FEATURE_EXTRACTORS.NAME = 'BoxFeatureExtractor'
        cfg.freeze()
        super(RelationFeatureExtractorAvgMask, self).__init__()
        
        #Feature Extractor(Pools the feature from diffreent scales and converts to matrix of shape num_ojbects x feature_dim)
        self.feature_extractor = build_box_feature_extractor(cfg, input_shape)
        self.out_channels = self.feature_extractor.out_channels

        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        in_channels = [input_shape[f].channels for f in in_features][0]
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES

        self.mask_on = cfg.MODEL.MASK_ON
        self.use_gt_box = cfg.MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_BOX
        self.in_features = in_features
        self.attention_type = cfg.MODEL.ROI_SCENEGRAPH_HEAD.MASK_ATTENTION_TYPE
        self.is_sigmoid = cfg.MODEL.ROI_SCENEGRAPH_HEAD.SIGMOID_ATTENTION
        self.rect_size = pooler_resolution * 4 -1 # size of union bounding box
        # self.rect_conv = nn.Sequential(*[
        #     nn.Conv2d(2, in_channels //2, kernel_size=7, stride=2, padding=3, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(in_channels//2, momentum=0.01),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #     nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(in_channels, momentum=0.01),
        #     ])
        self.use_mask_attention = cfg.MODEL.ROI_SCENEGRAPH_HEAD.USE_MASK_ATTENTION
        self.mask_combiner = cfg.MODEL.ROI_RELATION_FEATURE_EXTRACTORS.USE_MASK_COMBINER
        self.multiply_logits_with_masks = cfg.MODEL.ROI_RELATION_FEATURE_EXTRACTORS.MULTIPLY_LOGITS_WITH_MASKS   

    def forward(self, features, boxes, rel_pair_list=None, masks=None, proposals=None, return_seg_masks=False):
        device = boxes[0].device
        objects_per_image = torch.tensor([0] + [len(x) for x in proposals]).to(device)
        objects_per_image_sum = torch.cumsum(objects_per_image, dim=0)
        rel_pair_idx = []
        pred_scores = []
        num_rel_pair_idx = []
        for idx, (box, rel_idx) in enumerate(zip(boxes, rel_pair_list)):
            rel_pair_idx.append(rel_idx + objects_per_image_sum[idx])
            num_rel_pair_idx.append(len(rel_idx))
            pred_scores.append(proposals[idx].pred_scores)
            
        rel_pair_idx = torch.cat(rel_pair_idx, 0)
        num_rel_pair_idx = torch.tensor([0] + num_rel_pair_idx).to(device)
        num_rel_pair_idx_sum = torch.cumsum(num_rel_pair_idx, dim=0)
        boxes = Boxes.cat(boxes)
        head_box = boxes[rel_pair_idx[:, 0]]
        tail_box = boxes[rel_pair_idx[:, 1]]
        union_box = boxes_union(head_box, tail_box)
        if self.mask_on:
            masks = torch.cat(masks, 0)
            head_mask = masks[rel_pair_idx[:, 0]]
            tail_mask = masks[rel_pair_idx[:, 1]]
            if self.attention_type == 'Union':
                union_mask = masks_union(head_mask, tail_mask)
            elif self.attention_type == 'Avg':
                union_mask = (0.5 * head_mask) + (0.5 * tail_mask)
            else:
                raise Exception
        union_boxes = []
        union_masks = [] if self.mask_on else None
        for i in range(num_rel_pair_idx_sum.size(0) - 1):
            union_boxes.append(union_box[num_rel_pair_idx_sum[i]:num_rel_pair_idx_sum[i+1]]) 
            if self.mask_on:
                union_masks.append(union_mask[num_rel_pair_idx_sum[i]:num_rel_pair_idx_sum[i+1]])
        union_features = self.feature_extractor(features, union_boxes, masks=union_masks)
        return union_features, None   

def build_relation_feature_extractor(cfg, input_shape):
    name = cfg.MODEL.ROI_RELATION_FEATURE_EXTRACTORS.NAME
    new_cfg = cfg.clone()
    return ROI_RELATION_FEATURE_EXTRACTORS_REGISTRY.get(name)(new_cfg, input_shape)