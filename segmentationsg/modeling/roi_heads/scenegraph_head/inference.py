import torch
import torch.nn.functional as F
from torch import nn

from detectron2.structures.instances import Instances
from detectron2.structures.boxes import Boxes
from .utils import obj_prediction_nms, fast_rcnn_inference_single_image
from segmentationsg.utils import grammar_postprocessing
from collections import defaultdict
from detectron2.data import MetadataCatalog

class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
        self,
        use_gt_box=False,
        later_nms_pred_thres=0.3,
        later_obj_pred_score_thres=None,
        argmax_over_all_logits_and_filter_afterwards=False,
        topk_per_image=50,
        use_advanced_grammar_postprocessing=False,
        metadata=None
    ):
        """
        Arguments:
        """
        super(PostProcessor, self).__init__()
        self.use_gt_box = use_gt_box
        self.later_nms_pred_thres = later_nms_pred_thres
        self.later_obj_pred_score_thres = later_obj_pred_score_thres
        self.topk_per_image = topk_per_image
        self.argmax_over_all_logits_and_filter_afterwards = argmax_over_all_logits_and_filter_afterwards
        self.use_advanced_grammar_postprocessing= use_advanced_grammar_postprocessing
        self.metadata = metadata

    def forward(self, x, rel_pair_idxs, boxes, img_sizes, segmentation_vis=False, proposals=None):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the relation logits
                and finetuned object logits from the relation model.
            rel_pair_idxs （list[tensor]): subject and object indice of each relation,
                the size of tensor is (num_rel, 2)
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image
        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        relation_logits, refine_logits = x
        finetune_obj_logits = refine_logits

        results = []
        for i, (rel_logit, obj_logit, rel_pair_idx, box, img_size) in enumerate(zip(
            relation_logits, finetune_obj_logits, rel_pair_idxs, boxes, img_sizes
        )):
            
            obj_class_prob = F.softmax(obj_logit, -1)
            obj_class_prob[:, -1] = 0  # set background score to 0
            num_obj_bbox = obj_class_prob.shape[0]
            num_obj_class = obj_class_prob.shape[1]

            if self.use_gt_box:
                obj_scores, obj_pred = obj_class_prob[:, :-1].max(dim=1)
                result = Instances(img_size)
                result.pred_boxes = box
                obj_class = obj_pred
                result.pred_classes = obj_class
                result.scores = obj_scores
                result.pred_class_prob = obj_class_prob
            else:
                if self.later_obj_pred_score_thres is None:
                    # apply late nms for object prediction
                    #NOTE: previously, the 'boxes' object actually contained the proposals, not the boxes!
                    obj_pred = obj_prediction_nms(proposals[i].boxes_per_cls, obj_logit, self.later_nms_pred_thres)
                    obj_score_ind = torch.arange(num_obj_bbox, device=obj_logit.device) * num_obj_class + obj_pred
                    obj_scores = obj_class_prob.view(-1)[obj_score_ind]
                    assert obj_scores.shape[0] == num_obj_bbox
                    result = Instances(img_size)
                    result.pred_boxes =proposals[i]
                    obj_class = obj_pred
                    result.pred_classes = obj_class
                    result.scores = obj_scores
                    result.pred_class_prob = obj_class_prob
                    device = obj_class.device
                    batch_size = obj_class.shape[0]
                    regressed_box_idxs = obj_class
                    result.pred_boxes = Boxes(proposals[i].boxes_per_cls[torch.arange(batch_size, device=device), regressed_box_idxs])
                elif self.later_obj_pred_score_thres is not None:
                    #obj_scores = obj_class_prob.view(-1)
                    box_tensor_per_cls = proposals[i].boxes_per_cls
                    box_tensor_per_cls = box_tensor_per_cls.view(num_obj_bbox,-1)
                    result, filter_idxs = fast_rcnn_inference_single_image(box_tensor_per_cls, obj_class_prob, img_size, score_thresh=self.later_obj_pred_score_thres, nms_thresh=self.later_nms_pred_thres, topk_per_image=self.topk_per_image)

                    #filter rel_pair_idx and rel_logits after
                    #new_id_to_old_id_mapping = dict()
                    #old_id_to_new_id_mapping = {old_obj_nr:-1 for old_obj_nr in range(num_obj_bbox)}
                    old_id_to_new_id_mapping = dict()
                    for new_id, old_id in enumerate(filter_idxs):
                        old_id_to_new_id_mapping[old_id.item()] = new_id

                    #obj_id_to_new_id_mapping = None
                    rel_pair_idx_filtered = []
                    #rel_logit_filtered = []
                    invalid_relations_filter = []
                    for rel_pair_i, rel_pair in enumerate(rel_pair_idx):
                        subj_id, obj_id = rel_pair
                        subj_id = subj_id.item()
                        obj_id = obj_id.item()
                        if subj_id not in old_id_to_new_id_mapping or obj_id not in old_id_to_new_id_mapping:
                            invalid_relations_filter.append(False)
                        else:
                            invalid_relations_filter.append(True)
                            remapped_rel_pair = [old_id_to_new_id_mapping[subj_id], old_id_to_new_id_mapping[obj_id]]
                            rel_pair_idx_filtered.append(remapped_rel_pair)
                    rel_pair_idx_filtered = torch.LongTensor(rel_pair_idx_filtered)
                    rel_logit_filtered = rel_logit[invalid_relations_filter]
                    rel_pair_idx = rel_pair_idx_filtered
                    rel_logit = rel_logit_filtered
                    obj_scores = result.scores




            #            obj_class = obj_pred

#            result = Instances(img_size)
#
#            if self.use_gt_box:
#                result.pred_boxes = box
#            else:
#                # mode==sgdet
#                # apply regression based on finetuned object class
#                #FIXME
#                device = obj_class.device
#                batch_size = obj_class.shape[0]
#                regressed_box_idxs = obj_class
#                result.pred_boxes = Boxes(box.boxes_per_cls[torch.arange(batch_size, device=device), regressed_box_idxs])


            # sorting triples according to score production
            obj_scores0 = obj_scores[rel_pair_idx[:, 0]]
            obj_scores1 = obj_scores[rel_pair_idx[:, 1]]
            rel_class_prob = F.softmax(rel_logit, -1)

            if self.argmax_over_all_logits_and_filter_afterwards == True:
                #NOTE: Try to do a more precision-focused prediction (don't try to get all possible relations, even if they have lower logits than background, since we don't optimize for recall)
                num_rel_class = rel_class_prob.shape[1]
                rel_scores, rel_class = rel_class_prob[:, :].max(dim=1)
            else:
                rel_scores, rel_class = rel_class_prob[:, :-1].max(dim=1)

            triple_scores = rel_scores * obj_scores0 * obj_scores1
            _, sorting_idx = torch.sort(triple_scores.view(-1), dim=0, descending=True)
            rel_pair_idx = rel_pair_idx[sorting_idx]
            rel_class_prob = rel_class_prob[sorting_idx]
            rel_labels = rel_class[sorting_idx]

#            if self.argmax_over_all_logits_and_filter_afterwards == True:
#                background_removal_mask = (rel_labels != num_rel_class - 1)
#                filtered_rel_pair_idx = rel_pair_idx[background_removal_mask,:]
#                filtered_rel_class_prob = rel_class_prob[background_removal_mask,:]
#                filtered_rel_labels = rel_labels[background_removal_mask]
#
#                result._rel_pair_idxs = filtered_rel_pair_idx # (#rel, 2)
#                result._pred_rel_scores = filtered_rel_class_prob # (#rel, #rel_class)
#                result._pred_rel_labels = filtered_rel_labels # (#rel, )
#            else:
            result._rel_pair_idxs = rel_pair_idx # (#rel, 2)
            result._pred_rel_scores = rel_class_prob # (#rel, #rel_class)
            result._pred_rel_labels = rel_labels # (#rel, )


            if segmentation_vis:
                result._sorting_idx = sorting_idx
            # should have fields : rel_pair_idxs, pred_rel_class_prob, pred_rel_labels, pred_labels, pred_scores


            if self.use_advanced_grammar_postprocessing is True:
                metadata = self.metadata
                if 'ADtgt' in metadata.name:
                    has_documentroot = False
                else:
                    has_documentroot = True
                thing_classes = self.metadata.thing_classes
                predicate_classes = self.metadata.predicate_classes
                obj_class_prob = F.softmax(finetune_obj_logits[i], -1)
                num_obj_bbox = obj_class_prob.shape[0]
                rel_logit_matrix_tailcentric = torch.zeros((num_obj_bbox, num_obj_bbox+1, 3)).cuda()
                rel_logit_matrix_tailcentric[rel_pair_idx[:,1],rel_pair_idx[:,0],:] = rel_logit
                prediction = {"instances": result,
                              "rel_pair_idxs": result._rel_pair_idxs,
                              "pred_rel_scores": result._pred_rel_scores,
                              "rel_logit_matrix_tailcentric":
                                  rel_logit_matrix_tailcentric
                              }
                prediction_postprocessed = grammar_postprocessing.postprocess_prediction_with_grammar(prediction, thing_classes, has_documentroot=has_documentroot)
                result = prediction_postprocessed['instances']
                result._rel_pair_idxs = prediction_postprocessed['rel_pair_idxs']
                result._pred_rel_scores = prediction_postprocessed['pred_rel_scores']

            results.append(result)
        return results


def build_roi_scenegraph_post_processor(cfg):

    use_gt_box = cfg.MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_BOX
    later_nms_pred_thres = cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES
    later_obj_pred_score_thres = cfg.TEST.RELATION.LATER_OBJECT_SCORE_THRES
    argmax_over_all_logits_and_filter_afterwards = cfg.MODEL.ROI_SCENEGRAPH_HEAD.APPLY_ARGMAX_OVER_ALL_LOGITS_AND_FILTER_FG
    topk_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
    use_advanced_grammar_postprocessing = cfg.TEST.RELATION.USE_ADVANCED_GRAMMAR_POSTPROCESSING
    test_dataset_names = cfg.DATASETS.TEST
    test_dataset_name = test_dataset_names[0]
    metadata = MetadataCatalog.get(test_dataset_name)

    postprocessor = PostProcessor(
        use_gt_box,
        later_nms_pred_thres,
        later_obj_pred_score_thres,
        argmax_over_all_logits_and_filter_afterwards,
        topk_per_image,
        use_advanced_grammar_postprocessing,
        metadata
    )
    return postprocessor


class PostProcessorWithGrammar(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
            self,
            use_gt_box=False,
            later_nms_pred_thres=0.3,
            later_obj_pred_score_thres=None,
            use_mask_for_inference_softmax = True,
            argmax_over_all_logits_and_filter_afterwards=False,
            use_advanced_grammar_postprocessing = False,
            metadata=None
    ):
        """
        Arguments:
        """
        super(PostProcessorWithGrammar, self).__init__()
        self.use_gt_box = use_gt_box
        self.later_nms_pred_thres = later_nms_pred_thres
        self.later_obj_pred_score_thres = later_obj_pred_score_thres
        self.use_mask_for_inference_softmax = use_mask_for_inference_softmax
        self.argmax_over_all_logits_and_filter_afterwards = argmax_over_all_logits_and_filter_afterwards
        self.use_advanced_grammar_postprocessing = use_advanced_grammar_postprocessing
        self.metadata = metadata

    def forward(self, x, rel_pair_idxs, boxes, img_sizes, segmentation_vis=False, grammar_outputs=None):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the relation logits
                and finetuned object logits from the relation model.
            rel_pair_idxs （list[tensor]): subject and object indice of each relation,
                the size of tensor is (num_rel, 2)
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image
        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        relation_logits, refine_logits = x
        finetune_obj_logits = refine_logits
        assert grammar_outputs is not None

        results = []

        for i, (rel_logit, obj_logit, rel_pair_idx, box, img_size) in enumerate(zip(
                relation_logits, finetune_obj_logits, rel_pair_idxs, boxes, img_sizes
        )):

            obj_class_prob = F.softmax(obj_logit, -1)
            obj_class_prob[:, -1] = 0  # set background score to 0
            num_obj_bbox = obj_class_prob.shape[0]
            num_obj_class = obj_class_prob.shape[1]

            if self.use_gt_box:
                obj_scores, obj_pred = obj_class_prob[:, :-1].max(dim=1)
            else:
                # apply late nms for object prediction
                obj_pred = obj_prediction_nms(box.boxes_per_cls, obj_logit, self.later_nms_pred_thres)
                obj_score_ind = torch.arange(num_obj_bbox, device=obj_logit.device) * num_obj_class + obj_pred
                obj_scores = obj_class_prob.view(-1)[obj_score_ind]

            assert obj_scores.shape[0] == num_obj_bbox
            obj_class = obj_pred

            result = Instances(img_size)

            if self.use_gt_box:
                result.pred_boxes = box
            else:
                # mode==sgdet
                # apply regression based on finetuned object class
                device = obj_class.device
                batch_size = obj_class.shape[0]
                regressed_box_idxs = obj_class
                result.pred_boxes = Boxes(
                    box.boxes_per_cls[torch.arange(batch_size, device=device), regressed_box_idxs])

            result.pred_classes = obj_class
            result.scores = obj_scores
            result.pred_class_prob = obj_class_prob

            # sorting triples according to score production
            obj_scores0 = obj_scores[rel_pair_idx[:, 0]]
            obj_scores1 = obj_scores[rel_pair_idx[:, 1]]

            rel_class_prob = F.softmax(rel_logit, -1)
            if self.argmax_over_all_logits_and_filter_afterwards == True:
                # NOTE: Try to do a more precision-focused prediction (don't try to get all possible relations, even if they have lower logits than background, since we don't optimize for recall)
                num_rel_class = rel_class_prob.shape[1]
                rel_scores, rel_class = rel_class_prob[:, :].max(dim=1)
            else:
                rel_scores, rel_class = rel_class_prob[:, :-1].max(dim=1)


            triple_scores = rel_scores * obj_scores0 * obj_scores1
            _, sorting_idx = torch.sort(triple_scores.view(-1), dim=0, descending=True)
            rel_pair_idx = rel_pair_idx[sorting_idx]
            rel_class_prob = rel_class_prob[sorting_idx]
            rel_labels = rel_class[sorting_idx]

            if self.argmax_over_all_logits_and_filter_afterwards == True:
                background_removal_mask = (rel_labels != num_rel_class - 1)
                filtered_rel_pair_idx = rel_pair_idx[background_removal_mask, :]
                filtered_rel_class_prob = rel_class_prob[background_removal_mask, :]
                filtered_rel_labels = rel_labels[background_removal_mask]

                result._rel_pair_idxs = filtered_rel_pair_idx  # (#rel, 2)
                result._pred_rel_scores = filtered_rel_class_prob  # (#rel, #rel_class)
                result._pred_rel_labels = filtered_rel_labels  # (#rel, )
            else:
                result._rel_pair_idxs = rel_pair_idx  # (#rel, 2)
                result._pred_rel_scores = rel_class_prob  # (#rel, #rel_class)
                result._pred_rel_labels = rel_labels  # (#rel, )

            if self.use_advanced_grammar_postprocessing is True:
                metadata = self.metadata
                if 'ADtgt' in metadata.name:
                    has_documentroot = False
                else:
                    has_documentroot = True
                thing_classes = self.metadata.thing_classes
                predicate_classes = self.metadata.predicate_classes
                prediction = {"instances": result,
                                      "rel_pair_idxs": result._rel_pair_idxs,
                                      "pred_rel_scores": result._pred_rel_scores,
                                      "rel_logit_matrix_tailcentric": grammar_rel_dists_tailcentric_for_all_imgs[i]
                                      }
                prediction_postprocessed = grammar_postprocessing.postprocess_prediction_with_grammar(prediction, thing_classes, has_documentroot=has_documentroot)
                result = prediction_postprocessed['instances']
                result._rel_pair_idxs = prediction_postprocessed['rel_pair_idxs']
                result._pred_rel_scores = prediction_postprocessed['pred_rel_scores']

            # should have fields : rel_pair_idxs, pred_rel_class_prob, pred_rel_labels, pred_labels, pred_scores
            results.append(result)
        return results


def build_roi_scenegraph_post_processor_with_grammar(cfg):

    use_gt_box = cfg.MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_BOX
    later_nms_pred_thres = cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES
    later_obj_pred_score_thres = cfg.TEST.RELATION.LATER_OBJECT_SCORE_THRES
    argmax_over_all_logits_and_filter_afterwards = cfg.MODEL.ROI_SCENEGRAPH_HEAD.APPLY_ARGMAX_OVER_ALL_LOGITS_AND_FILTER_FG

    use_advanced_grammar_postprocessing = cfg.TEST.RELATION.USE_ADVANCED_GRAMMAR_POSTPROCESSING

    test_dataset_names = cfg.DATASETS.TEST
    test_dataset_name = test_dataset_names[0]
    metadata = MetadataCatalog.get(test_dataset_name)

    postprocessor = PostProcessorWithGrammar(
        use_gt_box,
        later_nms_pred_thres,
        argmax_over_all_logits_and_filter_afterwards,
        use_advanced_grammar_postprocessing,
        metadata
    )
    return postprocessor
