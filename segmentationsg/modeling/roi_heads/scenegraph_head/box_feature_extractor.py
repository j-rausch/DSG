import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.utils.registry import Registry
from detectron2.modeling.poolers import ROIPooler
from detectron2.layers import ShapeSpec
from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
ROI_BOX_FEATURE_EXTRACTORS_REGISTRY = Registry("ROI_BOX_FEATURE_EXTRACTORS_REGISTRY")

@ROI_BOX_FEATURE_EXTRACTORS_REGISTRY.register()
class BoxFeatureExtractor(nn.Module):
    """
    Class to pool the the features from different scale and flatten them using some fully connected layers.
    These feature will be used as node states for the scene graph.
    """

    def __init__(self, cfg, input_shape):
        super(BoxFeatureExtractor, self).__init__()

        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_FEATURE_EXTRACTORS.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_FEATURE_EXTRACTORS.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_FEATURE_EXTRACTORS.POOLER_TYPE
        mask_on           = cfg.MODEL.MASK_ON
        use_mask_in_box_features = cfg.MODEL.ROI_BOX_FEATURE_EXTRACTORS.BOX_FEATURE_MASK
        pooler = ROIPooler(
                            output_size=pooler_resolution,
                            scales=pooler_scales,
                            sampling_ratio=sampling_ratio,
                            pooler_type=pooler_type
                            )
        in_channels = [input_shape[f].channels for f in in_features][0]
        
        input_size = in_channels * pooler_resolution ** 2
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.attention_type = cfg.MODEL.ROI_SCENEGRAPH_HEAD.MASK_ATTENTION_TYPE
        if mask_on and use_mask_in_box_features:
            # input_size = input_size * 2
            self.combined_mask_input = cfg.MODEL.ROI_RELATION_FEATURE_EXTRACTORS.USE_MASK_COMBINER
            if not self.combined_mask_input:
                if self.attention_type == 'Diff_Channels':
                    self.mask_combiner_box = nn.Conv2d(in_channels, in_channels - 10, kernel_size=3, padding=1)
                    self.mask_combiner_mask = nn.Conv2d(self.num_classes, 10, kernel_size=3, padding=1)
                else:
                    self.mask_combiner = nn.Conv2d(in_channels + self.num_classes, in_channels, kernel_size=3, padding=1)
            else:
                self.mask_feature_extractor = nn.Conv2d(1, in_channels, kernel_size=3, padding=1)
                self.mask_combiner = nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1)
        representation_size = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        
        self.mask_on = mask_on
        self.use_mask_in_box_features = use_mask_in_box_features
        self.in_features = in_features
        # self.input_shape = shape
        self.pooler = pooler
        self.fc6 = make_fc(input_size, representation_size)
        
        out_dim = representation_size
        
        self.fc7 = make_fc(representation_size, out_dim)
        self.resize_channels = input_size
        self.out_channels = out_dim

    def forward(self, features, boxes, masks=None, logits=None, segmentation_step=False):
        features = [features[f] for f in self.in_features]
        box_features = self.pooler(features, boxes)
        if self.mask_on and (masks is not None) and self.use_mask_in_box_features:
            masks = torch.cat(masks)
            if self.attention_type == 'Zero':
                masks = torch.zeros_like(masks)
            if logits is not None:
                logits = torch.cat(logits)
                logits = logits.narrow(1, 0, masks.size(1)).unsqueeze(2).unsqueeze(3)
                masks = masks * logits 
                print ("NOPE")
            if not self.combined_mask_input:
                if self.attention_type == 'Diff_Channels':
                    box_features = self.mask_combiner_box(box_features)
                    masks = self.mask_combiner_mask(masks)
                    box_features = torch.cat([box_features, masks], 1)
                else:
                    box_features = torch.cat([box_features, masks], 1)
                    if box_features.size(0) > 500:
                        # Do it in chunks
                        box_features_chunks = torch.split(box_features, 100, dim=0)
                        box_features_all = []
                        for idx, box_feature_chunk in enumerate(box_features_chunks):
                            box_features_all.append(self.mask_combiner(box_feature_chunk))
                        box_features = torch.cat(box_features_all, dim=0)
                    else:
                        box_features = self.mask_combiner(box_features)
            else:
                mask_features = self.mask_feature_extractor(masks)
                box_features = torch.cat([box_features, mask_features], 1)
                if box_features.size(0) > 500:
                    # Do it in chunks
                    box_features_chunks = torch.split(box_features, 100, dim=0)
                    box_features_all = []
                    for idx, box_feature_chunk in enumerate(box_features_chunks):
                        box_features_all.append(self.mask_combiner(box_feature_chunk))
                    box_features = torch.cat(box_features_all, dim=0)
                else:
                    box_features = self.mask_combiner(box_features)
        box_features = box_features.flatten(1)
        box_features = F.relu(self.fc6(box_features))
        box_features = F.relu(self.fc7(box_features))
        return box_features

    def forward_without_pool(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x

@ROI_BOX_FEATURE_EXTRACTORS_REGISTRY.register()
class BoxFeatureSegmentationExtractor(nn.Module):
    """
    Class to pool the the features from different scale and flatten them using some fully connected layers.
    These feature will be used as node states for the scene graph.
    """

    def __init__(self, cfg, input_shape):
        super(BoxFeatureSegmentationExtractor, self).__init__()

        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_FEATURE_EXTRACTORS.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_FEATURE_EXTRACTORS.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_FEATURE_EXTRACTORS.POOLER_TYPE
        mask_on           = cfg.MODEL.MASK_ON
        use_mask_in_box_features = cfg.MODEL.ROI_BOX_FEATURE_EXTRACTORS.BOX_FEATURE_MASK
        pooler = ROIPooler(
                            output_size=pooler_resolution,
                            scales=pooler_scales,
                            sampling_ratio=sampling_ratio,
                            pooler_type=pooler_type
                            )
        in_channels = [input_shape[f].channels for f in in_features][0]
        input_size = in_channels * pooler_resolution ** 2
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.mask_num_classes = cfg.MODEL.ROI_HEADS.MASK_NUM_CLASSES
        if mask_on and use_mask_in_box_features:
            # input_size = input_size * 2
            self.combined_mask_input = cfg.MODEL.ROI_RELATION_FEATURE_EXTRACTORS.USE_MASK_COMBINER
            if not self.combined_mask_input:
                self.mask_combiner = nn.Conv2d(in_channels + self.num_classes, in_channels, kernel_size=3, padding=1)
                self.mask_combiner_segmentation = nn.Conv2d(in_channels + self.mask_num_classes, in_channels, kernel_size=3, padding=1)
            else:
                self.mask_feature_extractor = nn.Conv2d(1, in_channels, kernel_size=3, padding=1)
                self.mask_combiner = nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1)
        representation_size = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        self.attention_type = cfg.MODEL.ROI_SCENEGRAPH_HEAD.MASK_ATTENTION_TYPE
        self.mask_on = mask_on
        self.use_mask_in_box_features = use_mask_in_box_features
        self.in_features = in_features
        # self.input_shape = shape
        self.pooler = pooler
        self.fc6 = make_fc(input_size, representation_size)
        
        out_dim = representation_size
        
        self.fc7 = make_fc(representation_size, out_dim)
        self.resize_channels = input_size
        self.out_channels = out_dim

    def forward(self, features, boxes, masks=None, logits=None, segmentation_step=False):
        features = [features[f] for f in self.in_features]
        box_features = self.pooler(features, boxes)
        if self.mask_on and (masks is not None) and self.use_mask_in_box_features:
            masks = torch.cat(masks)
            if self.attention_type == 'Zero':
                masks = torch.zeros_like(masks)
            if logits is not None:
                logits = torch.cat(logits)
                logits = logits.narrow(1, 0, masks.size(1)).unsqueeze(2).unsqueeze(3)
                masks = masks * logits 
                print ("NOPE")
            if not self.combined_mask_input:
                box_features = torch.cat([box_features, masks], 1)
                if box_features.size(0) > 500:
                    # Do it in chunks
                    box_features_chunks = torch.split(box_features, 100, dim=0)
                    box_features_all = []
                    for idx, box_feature_chunk in enumerate(box_features_chunks):
                        if not segmentation_step:
                            box_features_all.append(self.mask_combiner(box_feature_chunk))
                        else:
                            box_features_all.append(self.mask_combiner_segmentation(box_feature_chunk))
                    box_features = torch.cat(box_features_all, dim=0)
                else:
                    if not segmentation_step:
                        box_features = self.mask_combiner(box_features)
                    else:
                        box_features = self.mask_combiner_segmentation(box_features)
            else:
                mask_features = self.mask_feature_extractor(masks)
                box_features = torch.cat([box_features, mask_features], 1)
                if box_features.size(0) > 500:
                    # Do it in chunks
                    box_features_chunks = torch.split(box_features, 100, dim=0)
                    box_features_all = []
                    for idx, box_feature_chunk in enumerate(box_features_chunks):
                        box_features_all.append(self.mask_combiner(box_feature_chunk))
                    box_features = torch.cat(box_features_all, dim=0)
                else:
                    box_features = self.mask_combiner(box_features)
        box_features = box_features.flatten(1)
        box_features = F.relu(self.fc6(box_features))
        box_features = F.relu(self.fc7(box_features))
        return box_features

    def forward_without_pool(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x

def make_fc(dim_in, hidden_dim):
    '''
        Make Fully connected Layer with xavier initialization
    '''
    
    fc = nn.Linear(dim_in, hidden_dim)
    # nn.init.kaiming_uniform_(fc.weight, a=1)
    # nn.init.constant_(fc.bias, 0)
    weight_init.c2_xavier_fill(fc)
    return fc

def build_box_feature_extractor(cfg, in_channels):
    name = cfg.MODEL.ROI_BOX_FEATURE_EXTRACTORS.NAME
    return ROI_BOX_FEATURE_EXTRACTORS_REGISTRY.get(name)(cfg, in_channels)
