#Modified from https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/maskrcnn_benchmark/modeling/roi_heads/relation_head/sampling.py

import torch
from torch.nn import functional as F
import numpy as np
import numpy.random as npr

from detectron2.structures.boxes import pairwise_iou

class RelationSampling(object):
    #sample relation pair proposals from given sets of bounding boxes
    def __init__(
        self,
        fg_thres,
        require_overlap,
        num_sample_per_gt_rel,
        batch_size_per_image,
        positive_fraction,
        use_gt_box,
        num_rel_classes,
        test_overlap,
    ):

        self.fg_thres = fg_thres
        self.require_overlap = require_overlap
        self.num_sample_per_gt_rel = num_sample_per_gt_rel
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.use_gt_box = use_gt_box
        self.num_rel_classes = num_rel_classes
        self.test_overlap = test_overlap
        

    def prepare_test_pairs(self, device, proposals):
        # prepare object pairs for relation prediction
        rel_pair_idxs = []
        for p in proposals:
            n = len(p)
            cand_matrix = torch.ones((n, n), device=device) - torch.eye(n, device=device)
            # mode==sgdet and require_overlap
            if (not self.use_gt_box) and self.test_overlap:
                cand_matrix = cand_matrix.byte() & pairwise_iou(p.pred_boxes, p.pred_boxes).gt(0).byte()
            
            idxs = torch.nonzero(cand_matrix, as_tuple=False).view(-1,2)
            del cand_matrix
            if len(idxs) > 0:
                rel_pair_idxs.append(idxs)
            else:
                # if there is no candidate pairs, give a placeholder of [[0, 0]]
                rel_pair_idxs.append(torch.zeros((1, 2), dtype=torch.int64, device=device))
        return rel_pair_idxs
    
    def gtbox_relsample(self, boxes, targets, relations):
        assert self.use_gt_box
        num_pos_per_img = int(self.batch_size_per_image * self.positive_fraction)
        rel_idx_pairs = []
        rel_labels = []
        rel_sym_binarys = []
        
        for img_id, (box, target, relation) in enumerate(zip(boxes, targets, relations)):
            device = box.device
            num_prp = box.tensor.shape[0]

            assert box.tensor.shape[0] == target.gt_boxes.tensor.shape[0]
            tgt_relations = relation # [tgt, tgt]
            # tgt_pair_idxs = torch.nonzero(tgt_rel_matrix > 0)
            tgt_pair_idxs = tgt_relations[:, :2]
            assert tgt_pair_idxs.shape[1] == 2
            tgt_head_idxs = tgt_pair_idxs[:, 0].contiguous().view(-1)
            tgt_tail_idxs = tgt_pair_idxs[:, 1].contiguous().view(-1)
            tgt_rel_labs = tgt_relations[:,2].contiguous().view(-1)

            # sym_binary_rels
            binary_rel = torch.zeros((num_prp, num_prp), device=device).long()
            binary_rel[tgt_head_idxs, tgt_tail_idxs] = 1
            binary_rel[tgt_tail_idxs, tgt_head_idxs] = 1
            rel_sym_binarys.append(binary_rel)
            
            rel_possibility = torch.ones((num_prp, num_prp), device=device).long() - torch.eye(num_prp, device=device).long()
            rel_possibility[tgt_head_idxs, tgt_tail_idxs] = 0
            tgt_bg_idxs = torch.nonzero(rel_possibility > 0, as_tuple=False)

            # generate fg bg rel_pairs
            if tgt_pair_idxs.shape[0] > num_pos_per_img:
                perm = torch.randperm(tgt_pair_idxs.shape[0], device=device)[:num_pos_per_img]
                tgt_pair_idxs = tgt_pair_idxs[perm]
                tgt_rel_labs = tgt_rel_labs[perm]
            num_fg = min(tgt_pair_idxs.shape[0], num_pos_per_img)

            num_bg = self.batch_size_per_image - num_fg
            perm = torch.randperm(tgt_bg_idxs.shape[0], device=device)[:num_bg]
            tgt_bg_idxs = tgt_bg_idxs[perm]

            img_rel_idxs = torch.cat((tgt_pair_idxs, tgt_bg_idxs), dim=0)
            img_rel_labels = torch.cat((tgt_rel_labs.long(), torch.full((tgt_bg_idxs.shape[0],), fill_value=self.num_rel_classes, device=device, dtype=torch.long)), dim=0).contiguous().view(-1)

            rel_idx_pairs.append(img_rel_idxs)
            rel_labels.append(img_rel_labels)

        return boxes, rel_labels, rel_idx_pairs, rel_sym_binarys
    
    def detect_relsample(self, proposals, targets, relations):
        # corresponding to rel_assignments function in neural-motifs
        """
        The input proposals are already processed by subsample function of box_head,
        in this function, we should only care about fg box, and sample corresponding fg/bg relations
        Note: this function keeps a state.
        Arguments:
            boxes (list[BoxList])  contain fields: labels, predict_logits
            targets (list[BoxList]) contain fields: labels
        """
        self.num_pos_per_img = int(self.batch_size_per_image * self.positive_fraction)
        rel_idx_pairs = []
        rel_labels = []
        rel_sym_binarys = []
        for img_id, (proposal, target, relation) in enumerate(zip(proposals, targets, relations)):
            device = proposal.pred_boxes.device
            prp_box = proposal.pred_boxes
            prp_lab = proposal.pred_classes.long()
            tgt_box = target.gt_boxes
            tgt_lab = target.gt_classes.long()
            tgt_rel_matrix = torch.zeros(tgt_lab.shape[0],tgt_lab.shape[0]).long().to(device) # [tgt, tgt]
            #NOTE: this here was initialized with ZEROS. 'followedby' is also marked with a zero in our GT!
            tgt_rel_matrix[relation[:,0], relation[:,1]] = relation[:,2] + 1
            # IoU matching
            #TODO: how is possible that many proposals are not matched to targets even if the proposals are essentially GT boxes?
            ious = pairwise_iou(tgt_box, prp_box)  # [tgt, prp]

            is_match = (tgt_lab[:,None] == prp_lab[None]) & (ious > self.fg_thres) # [tgt, prp]
            # Proposal self IoU to filter non-overlap
            prp_self_iou = pairwise_iou(prp_box, prp_box)  # [prp, prp]
            if self.require_overlap and (not self.use_gt_box):
                rel_possibility = (prp_self_iou > 0) & (prp_self_iou < 1)  # not self & intersect
            else:
                num_prp = len(prp_box)
                rel_possibility = torch.ones((num_prp, num_prp), device=device).long() - torch.eye(num_prp, device=device).long()
            # only select relations between fg proposals
            #Fix for background class
#            print('prp lab: {}'.format(prp_lab))
#            print('num rel classes: {}'.format(self.num_rel_classes))
#            print('rel possibility: {}'.format(rel_possibility))
            #NOTE: this seems to be wrong. how does the label of a predicted object == number of relations have any significance?
#            rel_possibility[prp_lab == self.num_rel_classes] = 0
#            rel_possibility[:, prp_lab == self.num_rel_classes] = 0

            img_rel_triplets, binary_rel = self.motif_rel_fg_bg_sampling(device, tgt_rel_matrix, ious, is_match, rel_possibility)
            rel_idx_pairs.append(img_rel_triplets[:, :2]) # (num_rel, 2),  (sub_idx, obj_idx)
            rel_labels.append(img_rel_triplets[:, 2]) # (num_rel, )
            rel_sym_binarys.append(binary_rel)

        return proposals, rel_labels, rel_idx_pairs, rel_sym_binarys
    
    def motif_rel_fg_bg_sampling(self, device, tgt_rel_matrix, ious, is_match, rel_possibility):
        """
        prepare to sample fg relation triplet and bg relation triplet
        tgt_rel_matrix: # [number_target, number_target]
        ious:           # [number_target, num_proposal]
        is_match:       # [number_target, num_proposal]
        rel_possibility:# [num_proposal, num_proposal]
        """
        tgt_pair_idxs = torch.nonzero(tgt_rel_matrix > 0)
        assert tgt_pair_idxs.shape[1] == 2
        tgt_head_idxs = tgt_pair_idxs[:, 0].contiguous().view(-1)
        tgt_tail_idxs = tgt_pair_idxs[:, 1].contiguous().view(-1)
        tgt_rel_labs = tgt_rel_matrix[tgt_head_idxs, tgt_tail_idxs].contiguous().view(-1)
        #NOTE: we incremented tgt_rel_matrix values by one, so we can filter by non-zero. we adjust for this by subtracting one from the labels afterwards
        tgt_rel_labs = tgt_rel_labs - 1



        num_tgt_rels = tgt_rel_labs.shape[0]
        # generate binary prp mask
        num_prp = is_match.shape[-1]
        binary_prp_head = is_match[tgt_head_idxs] # num_tgt_rel, num_prp (matched prp head)
        binary_prp_tail = is_match[tgt_tail_idxs] # num_tgt_rel, num_prp (matched prp head)
        binary_rel = torch.zeros((num_prp, num_prp), device=device).long()

        fg_rel_triplets = []
        for i in range(num_tgt_rels):
            # generate binary prp mask
            bi_match_head = torch.nonzero(binary_prp_head[i] > 0)
            bi_match_tail = torch.nonzero(binary_prp_tail[i] > 0)

            num_bi_head = bi_match_head.shape[0]
            num_bi_tail = bi_match_tail.shape[0]
            if num_bi_head > 0 and num_bi_tail > 0:
                bi_match_head = bi_match_head.view(1, num_bi_head).expand(num_bi_tail, num_bi_head).contiguous()
                bi_match_tail = bi_match_tail.view(num_bi_tail, 1).expand(num_bi_tail, num_bi_head).contiguous()
                # binary rel only consider related or not, so its symmetric
                binary_rel[bi_match_head.view(-1), bi_match_tail.view(-1)] = 1
                binary_rel[bi_match_tail.view(-1), bi_match_head.view(-1)] = 1

            tgt_head_idx = int(tgt_head_idxs[i])
            tgt_tail_idx = int(tgt_tail_idxs[i])
            tgt_rel_lab = int(tgt_rel_labs[i])
            # find matching pair in proposals (might be more than one)
            prp_head_idxs = torch.nonzero(is_match[tgt_head_idx]).squeeze(1)
            prp_tail_idxs = torch.nonzero(is_match[tgt_tail_idx]).squeeze(1)
            num_match_head = prp_head_idxs.shape[0]
            num_match_tail = prp_tail_idxs.shape[0]
            if num_match_head <= 0 or num_match_tail <= 0:
                continue
            # all combination pairs
            prp_head_idxs = prp_head_idxs.view(-1,1).expand(num_match_head,num_match_tail).contiguous().view(-1)
            prp_tail_idxs = prp_tail_idxs.view(1,-1).expand(num_match_head,num_match_tail).contiguous().view(-1)
            valid_pair = prp_head_idxs != prp_tail_idxs
            if valid_pair.sum().item() <= 0:
                continue
            # remove self-pair
            # remove selected pair from rel_possibility
            prp_head_idxs = prp_head_idxs[valid_pair]
            prp_tail_idxs = prp_tail_idxs[valid_pair]
            rel_possibility[prp_head_idxs, prp_tail_idxs] = 0
            # construct corresponding proposal triplets corresponding to i_th gt relation
            fg_labels = torch.tensor([tgt_rel_lab]*prp_tail_idxs.shape[0], dtype=torch.int64, device=device).view(-1,1)
            fg_rel_i = cat((prp_head_idxs.view(-1,1), prp_tail_idxs.view(-1,1), fg_labels), dim=-1).to(torch.int64)
            # select if too many corresponding proposal pairs to one pair of gt relationship triplet
            # NOTE that in original motif, the selection is based on a ious_score score 
            if fg_rel_i.shape[0] > self.num_sample_per_gt_rel:
                ious_score = (ious[tgt_head_idx, prp_head_idxs] * ious[tgt_tail_idx, prp_tail_idxs]).view(-1).detach().cpu().numpy()
                ious_score = ious_score / ious_score.sum()
                perm = npr.choice(ious_score.shape[0], p=ious_score, size=self.num_sample_per_gt_rel, replace=False)
                fg_rel_i = fg_rel_i[perm]
            if fg_rel_i.shape[0] > 0:
                fg_rel_triplets.append(fg_rel_i)
        
        # select fg relations
        if len(fg_rel_triplets) == 0:
            fg_rel_triplets = torch.zeros((0, 3), dtype=torch.int64, device=device)
        else:
            fg_rel_triplets = cat(fg_rel_triplets, dim=0).to(torch.int64)
            if fg_rel_triplets.shape[0] > self.num_pos_per_img:
                perm = torch.randperm(fg_rel_triplets.shape[0], device=device)[:self.num_pos_per_img]
                fg_rel_triplets = fg_rel_triplets[perm]

        # select bg relations
        bg_rel_inds = torch.nonzero(rel_possibility>0).view(-1,2)
        bg_rel_labs = torch.full((bg_rel_inds.shape[0],), fill_value=self.num_rel_classes, dtype=torch.int64, device=device)
        bg_rel_triplets = cat((bg_rel_inds, bg_rel_labs.view(-1,1)), dim=-1).to(torch.int64)

        num_neg_per_img = min(self.batch_size_per_image - fg_rel_triplets.shape[0], bg_rel_triplets.shape[0])
        if bg_rel_triplets.shape[0] > 0:
            perm = torch.randperm(bg_rel_triplets.shape[0], device=device)[:num_neg_per_img]
            bg_rel_triplets = bg_rel_triplets[perm]
        else:
            bg_rel_triplets = torch.zeros((0, 3), dtype=torch.int64, device=device)

        # if both fg and bg is none
        if fg_rel_triplets.shape[0] == 0 and bg_rel_triplets.shape[0] == 0:
            bg_rel_triplets = torch.zeros((1, 3), dtype=torch.int64, device=device)

        return cat((fg_rel_triplets, bg_rel_triplets), dim=0), binary_rel

def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)

def build_roi_scenegraph_samp_processor(cfg):

    samp_processor = RelationSampling(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_SCENEGRAPH_HEAD.REQUIRE_BOX_OVERLAP,
        cfg.MODEL.ROI_SCENEGRAPH_HEAD.NUM_SAMPLE_PER_GT_REL,
        cfg.MODEL.ROI_SCENEGRAPH_HEAD.BATCH_SIZE_PER_IMAGE, 
        cfg.MODEL.ROI_SCENEGRAPH_HEAD.POSITIVE_FRACTION,
        cfg.MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_BOX,
        cfg.MODEL.ROI_SCENEGRAPH_HEAD.NUM_CLASSES,
        cfg.TEST.RELATION.REQUIRE_OVERLAP,
    )

    return samp_processor
