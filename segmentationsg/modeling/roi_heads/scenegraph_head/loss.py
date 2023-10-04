import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import numpy.random as npr

from fvcore.nn import smooth_l1_loss 
from .motif import cat

class RelationLossComputation(object):
    """
    Computes the loss for relation triplet.
    Also supports FPN
    """

    def __init__(
        self,
        use_label_smoothing,
        predicate_proportion,
        use_doc_relation_weights=None
    ):
        """
        Arguments:
            bbox_proposal_matcher (Matcher)
            rel_fg_bg_sampler (RelationPositiveNegativeSampler)
        """
        
        self.use_label_smoothing = use_label_smoothing
        self.pred_weight = (1.0 / torch.FloatTensor([0.5,] + predicate_proportion)).cuda()
        self.use_doc_relation_weights = use_doc_relation_weights

        if self.use_doc_relation_weights:
            doc_relation_weights = torch.tensor([20, 10, 1]).float().cuda() #20 for followedby, 10 for parentof, 1 for background
            assert not self.use_label_smoothing
            self.criterion_loss_relation = nn.CrossEntropyLoss(weight=doc_relation_weights)
        else:
            self.criterion_loss_relation = nn.CrossEntropyLoss()

        if self.use_label_smoothing:
            self.criterion_loss = Label_Smoothing_Regression(e=0.01)
        else:
            self.criterion_loss = nn.CrossEntropyLoss()


    def __call__(self, proposals, rel_labels, relation_logits, refine_logits):
        """
        Computes the loss for relation triplet.
        This requires that the subsample method has been called beforehand.
        Arguments:
            relation_logits (list[Tensor])
            refine_obj_logits (list[Tensor])
        Returns:
            predicate_loss (Tensor)
            finetune_obj_loss (Tensor)
        """
        refine_obj_logits = refine_logits
        relation_logits = cat(relation_logits, dim=0)
        refine_obj_logits = cat(refine_obj_logits, dim=0)

        fg_labels = cat([proposal.gt_classes for proposal in proposals], dim=0)
        rel_labels = cat(rel_labels, dim=0)

        #loss_relation = self.criterion_loss(relation_logits, rel_labels.long())
#        debug_pred_classes = torch.argmax(refine_obj_logits, dim=1)
#        debug_rel_classes = torch.argmax(relation_logits, dim=1)
        loss_relation = self.criterion_loss_relation(relation_logits, rel_labels.long())
        loss_refine_obj = self.criterion_loss(refine_obj_logits, fg_labels.long())

        
        return loss_relation, loss_refine_obj


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        target = target.view(-1)

        logpt = F.log_softmax(input)
        logpt = logpt.index_select(-1, target).diag()
        logpt = logpt.view(-1)
        pt = logpt.exp()

        logpt = logpt * self.alpha * (target > 0).float() + logpt * (1 - self.alpha) * (target <= 0).float()

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

import torch
import torch.nn as nn


class Label_Smoothing_Regression(nn.Module):

    def __init__(self, e=0.01, reduction='mean'):
        super().__init__()

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.e = e
        self.reduction = reduction
    
    def _one_hot(self, labels, classes, value=1):
        """
            Convert labels to one hot vectors
        
        Args:
            labels: torch tensor in format [label1, label2, label3, ...]
            classes: int, number of classes
            value: label value in one hot vector, default to 1
        
        Returns:
            return one hot format labels in shape [batchsize, classes]
        """

        one_hot = torch.zeros(labels.size(0), classes)

        #labels and value_added  size must match
        labels = labels.view(labels.size(0), -1)
        value_added = torch.Tensor(labels.size(0), 1).fill_(value)

        value_added = value_added.to(labels.device)
        one_hot = one_hot.to(labels.device)

        one_hot.scatter_add_(1, labels, value_added)

        return one_hot

    def _smooth_label(self, target, length, smooth_factor):
        """convert targets to one-hot format, and smooth
        them.
        Args:
            target: target in form with [label1, label2, label_batchsize]
            length: length of one-hot format(number of classes)
            smooth_factor: smooth factor for label smooth
        
        Returns:
            smoothed labels in one hot format
        """
        one_hot = self._one_hot(target, length, value=1 - smooth_factor)
        one_hot += smooth_factor / length

        return one_hot.to(target.device)

    def forward(self, x, target):

        if x.size(0) != target.size(0):
            raise ValueError('Expected input batchsize ({}) to match target batch_size({})'
                    .format(x.size(0), target.size(0)))

        if x.dim() < 2:
            raise ValueError('Expected input tensor to have least 2 dimensions(got {})'
                    .format(x.size(0)))

        if x.dim() != 2:
            raise ValueError('Only 2 dimension tensor are implemented, (got {})'
                    .format(x.size()))


        smoothed_target = self._smooth_label(target, x.size(1), self.e)
        x = self.log_softmax(x)
        loss = torch.sum(- x * smoothed_target, dim=1)

        if self.reduction == 'none':
            return loss
        
        elif self.reduction == 'sum':
            return torch.sum(loss)
        
        elif self.reduction == 'mean':
            return torch.mean(loss)
        
        else:
            raise ValueError('unrecognized option, expect reduction to be one of none, mean, sum')

def build_roi_scenegraph_loss_evaluator(cfg):

    loss_evaluator = RelationLossComputation(
        cfg.MODEL.ROI_SCENEGRAPH_HEAD.LABEL_SMOOTHING_LOSS,
        cfg.MODEL.ROI_SCENEGRAPH_HEAD.REL_PROP,
        use_doc_relation_weights=cfg.MODEL.ROI_SCENEGRAPH_HEAD.USE_FG_DOC_RELATION_WEIGHTS
    )

    return loss_evaluator


class RelationLossComputationWithGrammar(object):
    """
    Computes the loss for relation triplet.
    Also supports FPN
    """

    def __init__(
            self,
            use_label_smoothing,
            predicate_proportion,
            num_obj_cls,
            num_rel_cls,
            proposals_batch_size,
            use_doc_relation_weights=None
    ):
        """
        Arguments:
            bbox_proposal_matcher (Matcher)
            rel_fg_bg_sampler (RelationPositiveNegativeSampler)
        """

        self.use_label_smoothing = use_label_smoothing
        self.pred_weight = (1.0 / torch.FloatTensor([0.5, ] + predicate_proportion)).cuda()
        self.use_doc_relation_weights = use_doc_relation_weights

        self.num_obj_cls = num_obj_cls
        self.num_rel_cls = num_rel_cls

        if self.use_doc_relation_weights:
            doc_relation_weights = torch.tensor([20, 10, 1]).float().cuda() #20 for followedby, 10 for parentof, 1 for background
            assert not self.use_label_smoothing
            self.criterion_loss_relation = nn.CrossEntropyLoss(weight=doc_relation_weights)
        else:
            self.criterion_loss_relation = nn.CrossEntropyLoss()

        self.proposals_batch_size = proposals_batch_size
        
        if self.use_label_smoothing:
            self.criterion_loss = Label_Smoothing_Regression(e=0.01)
        else:
            self.criterion_loss = nn.CrossEntropyLoss()

    def __call__(self, proposals, rel_labels, relation_logits, refine_logits, grammar_outputs):
        """
        Computes the loss for relation triplet.
        This requires that the subsample method has been called beforehand.
        Arguments:
            relation_logits (list[Tensor])
            refine_obj_logits (list[Tensor])
        Returns:
            predicate_loss (Tensor)
            finetune_obj_loss (Tensor)
        """
        refine_obj_logits = refine_logits
        refine_obj_logits = cat(refine_obj_logits, dim=0)

        fg_labels = cat([proposal.gt_classes for proposal in proposals], dim=0)
        loss_refine_obj = self.criterion_loss(refine_obj_logits, fg_labels.long())



        relation_losses = dict()
        relation_logits = cat(relation_logits, dim=0)
        rel_labels = cat(rel_labels, dim=0)
        #loss_relation = self.criterion_loss(relation_logits, rel_labels.long())
        loss_relation = self.criterion_loss_relation(relation_logits, rel_labels.long())
        relation_losses['loss_relation'] = loss_relation

        return relation_losses, loss_refine_obj



def build_roi_scenegraph_loss_evaluator_with_grammar(cfg):
    loss_evaluator = RelationLossComputationWithGrammar(
        cfg.MODEL.ROI_SCENEGRAPH_HEAD.LABEL_SMOOTHING_LOSS,
        cfg.MODEL.ROI_SCENEGRAPH_HEAD.REL_PROP,
        proposals_batch_size=cfg.MODEL.ROI_SCENEGRAPH_HEAD.BATCH_SIZE_PER_IMAGE,
        num_obj_cls = cfg.MODEL.ROI_HEADS.NUM_CLASSES,
        num_rel_cls = cfg.MODEL.ROI_SCENEGRAPH_HEAD.NUM_CLASSES,
        use_doc_relation_weights=cfg.MODEL.ROI_SCENEGRAPH_HEAD.USE_FG_DOC_RELATION_WEIGHTS
    )

    return loss_evaluator
