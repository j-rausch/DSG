import json
import logging
import logging.config
import os
from collections import defaultdict
from shutil import copyfile
import re
import torch

from segmentationsg.modeling.roi_heads.scenegraph_head.doc_heuristics.objdetmetrics_lib import Evaluator
from segmentationsg.modeling.roi_heads.scenegraph_head.doc_heuristics.objdetmetrics_lib.BoundingBox import BoundingBox
from segmentationsg.modeling.roi_heads.scenegraph_head.doc_heuristics.objdetmetrics_lib.BoundingBoxes import getBoundingBoxesForFile, getBoundingBoxesFromDetectron2
from segmentationsg.modeling.roi_heads.scenegraph_head.doc_heuristics.objdetmetrics_lib.utils import BBFormat

from detectron2.utils.logger import setup_logger
from detectron2.structures.instances import Instances
import torch.nn.functional as F
from detectron2.structures.boxes import Boxes

logger = setup_logger(name=__name__)
#logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

from segmentationsg.modeling.roi_heads.scenegraph_head.doc_heuristics.utils.postprocess_utils import get_detections_from_file
from segmentationsg.modeling.roi_heads.scenegraph_head.doc_heuristics.utils.postprocess_table_structure import process_all_table_structure_annotations
from segmentationsg.modeling.roi_heads.scenegraph_head.doc_heuristics.utils.structure_utils import create_flat_annotation_list, merge_annotation_lists_util

LEGACY_CLASSES = [
    'abstract',
    'affiliation',
    'author',
    'bibblock',
    'contentblock',
    'date',
    'document',
    'equation',
    'figure',
    'figurecaption',
    'figuregraphic',
    'foot',
    'head',
    'heading',
    'item',
    'itemize',
    'meta',
    'pagenr',
    'subject',
    'table',
    'tablecaption',
    'tabular',
]



#LEGACY_CLASSES_EP =  ['article', 'author', 'backgroundfigure', 'col', 'contentblock',
#                              'documentroot', 'figure', 'figurecaption', 'figuregraphic', 'foot',
#                              'footnote', 'head', 'header', 'introduction', 'item', 'itemize', 'logo',
#                              'meta', 'pagenr', 'row', 'table', 'tableofcontent', 'tabular', 'unk']
LEGACY_CLASSES_EP =  ['article', 'author', 'backgroundfigure', 'col', 'contentblock', 'documentroot', 'figure', 'figurecaption', 'figuregraphic', 'foot', 'footnote', 'head', 'header', 'item', 'itemize', 'meta', 'orderedgroup', 'pagenr', 'row', 'table', 'tableofcontent', 'tabular', 'unorderedgroup']
LEGACY_REMAP_EP = {'article': 'document'}


def create_dir_if_not_exists(dir_path):
    if not os.path.isdir(dir_path):
        logger.debug('creating directory: {}'.format(dir_path))
        try:
            os.makedirs(dir_path)
        except FileExistsError:
            logger.error('paralleliziation error')

class StructureParser(object):

    def __init__(self, classes=None, dataset_type='ADtgt'):

        class_list = []
        if classes is not None:
            for cl in classes:
                #if cl not in DocsDataset.ALL_CLASSES:
                #if cl not in LEGACY_CLASSES:
                if cl not in LEGACY_CLASSES:
                        raise AttributeError(
                        "Selected class {} not in LEGACY_CLASSES list: {}".format(cl,
                                                                               LEGACY_CLASSES))
                else:
                    class_list.append(cl)
            self.classes = list(set(class_list))
        else:
            if dataset_type == 'ADtgt':
                self.classes = LEGACY_CLASSES
            elif dataset_type == 'EP':
                self.classes = LEGACY_CLASSES_EP
                self.remap_classes = LEGACY_REMAP_EP
                self.classes = [self.remap_classes.get(cl, cl) for cl in self.classes]
            else:
                raise NotImplementedError("dataset_type {} not implemented".format(dataset_mode))

        class_mapping = {i:v for i,v in enumerate(self.classes)}
        reverse_class_mapping = {v:i for i,v in enumerate(self.classes)}
        self.class_mapping = class_mapping
        self.reverse_class_mapping = reverse_class_mapping
        self.current_added_imgs = 0

    def forward_document_heuristics_from_detectron2(self, features,instances,targets,relations, table_mode=False, do_postprocessing=False, *, rel_pair_idxs):

#        boxes = [x.pred_boxes for x in instances]
#        rel_labels, rel_binarys = None, None
#        rel_pair_idxs = self.samp_processor.prepare_test_pairs(boxes[0].device, instances)

        if table_mode:
            raise NotImplementedError("table structure parsing not ported from legacy")

        #TODO: either make temporary file with bounding boxes from detectron2, or pass directly
#        allBoundingBoxes, _ = getBoundingBoxesForFile(detection_result_file, isGT=False,
#                                                              bbFormat=BBFormat.XYX2Y2,
#                                                              coordType='abs', header=True)
        first_img_id = self.current_added_imgs
        #class_mapping = {i:v for i,v in enumerate(self.classes)}
        class_mapping = self.class_mapping
        allBoundingBoxes, _ = getBoundingBoxesFromDetectron2(instances, isGT=False,
                                                              coordType='abs', first_img_id=first_img_id, class_mapping=class_mapping)
        self.current_added_imgs += len(instances)
        img_names = sorted(allBoundingBoxes.getBoundingBoxImageNames())
        try:
            img_name = img_names[0]
        except IndexError as e:
            logger.warning(
                "No image names exist for detection file. \nReturning empty structure, as no objects were detected")
            #img_relations_dict = {'relations': [], 'flat_annotations': None, 'all_bboxes': []}
            #TODO better handling if no objects are detected; basic operation: add 'article' and 'meta' nodes
            results = instances
#            for result in results:
#                rel_pair_idxs_list = []
#                pred_rel_scores_list = []
#                new_pred_rel_scores = torch.tensor(pred_rel_scores_list)
#                new_rel_pair_idxs = torch.tensor(rel_pair_idxs_list)
#                rel_labels = torch.tensor([])
##                rel_scores, rel_class = new_pred_rel_scores[:, :].max(dim=1)
##                rel_labels = rel_class
#                result._rel_pair_idxs = new_rel_pair_idxs.cuda()  # (#rel, 2)
#                result._pred_rel_scores = new_pred_rel_scores.cuda()  # (#rel, #rel_class)
#                result._pred_rel_labels = rel_labels.cuda()  # (#rel, )
            
            return results 
        assert len(img_names) == 1
        all_bboxes_for_img = allBoundingBoxes.getBoundingBoxesByImageName(img_name)
        all_ids_for_img = [x.getBboxID() for x in all_bboxes_for_img]
        logger.debug('creating structure for current img: {}'.format(img_name))
        max_loop_count = 30
        loop_count = 0
      
      
        
        number_of_boxes_changed = False 
        if do_postprocessing:
            updated_bboxes = True
            while (updated_bboxes is True):
                loop_count += 1

                is_parent_relations, sequence_relations, meta_bboxes_for_img, invalid_toplevel_bboxes = generate_relations_for_image(
                    all_bboxes_for_img,
                    enforce_hierarchy=False)  # also allow bad parent-child relations to be generated such that they can be potentially fixed

                if loop_count > max_loop_count:
                    logger.debug("Exited postprocessing, because maximum loop count reached")
                    break
                updated_bboxes, all_bboxes_for_img = self.align_parents_and_children(
                    all_bboxes_for_img,
                    is_parent_relations)
                if updated_bboxes:
                    continue

                len_before = len(all_bboxes_for_img)
                updated_bboxes, all_bboxes_for_img = self.merged_direct_nesting_of_same_category(
                    all_bboxes_for_img,
                    is_parent_relations)
                len_after = len(all_bboxes_for_img)
                if len_before != len_after:
                    number_of_boxes_changed = True
                    print('number of entities was changed from {} to {}'.format(len_before, len_after))
                if updated_bboxes:
                    continue

                len_before = len(all_bboxes_for_img)
                updated_bboxes, all_bboxes_for_img = self.wrap_invalid_children(
                    all_bboxes_for_img,
                    is_parent_relations,
                    all_ids_for_img)
                if len_before != len_after:
                    number_of_boxes_changed = True
                    print('(wrapped invalid children - number of entities was changed from {} to {}'.format(len_before, len_after))
                if updated_bboxes:
                    continue
                updated_bboxes, all_bboxes_for_img = self.wrap_invalid_toplevel_bboxes(
                    all_bboxes_for_img,
                    invalid_toplevel_bboxes,
                    sequence_relations,
                    is_parent_relations)
                if updated_bboxes:
                    number_of_boxes_changed = True
                    continue
                
            is_parent_relations, sequence_relations, meta_bboxes_for_img, invalid_toplevel_bboxes = generate_relations_for_image(
                all_bboxes_for_img)
            flat_annotations = None

        else:
            is_parent_relations, sequence_relations, meta_bboxes_for_img, invalid_toplevel_bboxes = generate_relations_for_image(
                all_bboxes_for_img)
            flat_annotations = None
#            flat_annotations = create_flat_annotation_list(all_bboxes_for_img, meta_bboxes_for_img,
#                                                           is_parent_relations,
#                                                           sequence_relations)
#            img_relations_dict = {'relations': is_parent_relations + sequence_relations,
#                                  'flat_annotations': flat_annotations,
#                                  'all_bboxes': all_bboxes_for_img}


        #TODO: generate relation logits from 'relations' and take over refine logtis from instances

        relation_logits = None
        refine_logits = None
        relation_logits = []
        relation_outputs = dict()
        for is_parent_relation in is_parent_relations:
            (head, tail, _) = is_parent_relation
            relation_outputs[(head,tail)] = 1
        for followedby_relation in sequence_relations:
            (head, tail, _) = followedby_relation
            if (head,tail) in relation_outputs:
                raise AssertionError
            relation_outputs[(head,tail)] = 0

        results = []
        for batch_nr, rel_pair_idxs_for_img in enumerate(rel_pair_idxs):
            
            #TODO: extend rel_pair_idxs_for_img, if new entities were added by postprocessing
            entity_id_mapping = {i:i for i in range(len(all_bboxes_for_img))}
            if number_of_boxes_changed is True:
                remaining_box_ids = [x.getBboxID() for x in all_bboxes_for_img]
                removed_box_indices = [x for x in all_ids_for_img if x not in remaining_box_ids]
                for removed_box_index in removed_box_indices:
                    entity_id_mapping[removed_box_index] = None
                instances_for_batch = instances[batch_nr]
                orig_nr_of_instances = len(instances_for_batch)
                new_max_nr_of_instances = max(remaining_box_ids) + 1
                all_newly_created_instances = []
                instance_index_mapping = {x:x for x in remaining_box_ids}
                if new_max_nr_of_instances > orig_nr_of_instances:
                    #TODO: extend instances_for_batch
                    missing_bboxes = [x for x in all_bboxes_for_img if x.getBboxID() > orig_nr_of_instances-1]
                    new_box_index = orig_nr_of_instances
                    for missing_bbox in missing_bboxes:
                        instance_index_mapping[missing_bbox.getBboxID()] = new_box_index
                        new_instance = self.create_new_instance_from_bboxes(missing_bbox, instances_for_batch)
                        all_newly_created_instances.append(new_instance)
                        new_box_index += 1
                    merged_newly_created_instances = Instances.cat([instances_for_batch] + all_newly_created_instances)
                else:
                    merged_newly_created_instances = instances_for_batch
                print('remaining box ids: {}'.format(remaining_box_ids))
                instances_to_preserve = [instance_index_mapping[x] for x in remaining_box_ids]
                new_instances = merged_newly_created_instances[instances_to_preserve]
                for new_idx, remaining_box_id in enumerate(remaining_box_ids):
                    entity_id_mapping[remaining_box_id] = new_idx 
                pass
                #TODO: relation mapping needs to be made now
            else:
                instances_for_batch = instances[batch_nr]
                new_instances = instances_for_batch
            
            remapped_relation_outputs = dict()
            for (head, tail) in relation_outputs:
                remapped_head = entity_id_mapping[head]
                remapped_tail = entity_id_mapping[tail]
                remapped_relation_outputs[(remapped_head,remapped_tail)] = relation_outputs[(head,tail)]

#            relation_logits_for_img = torch.zeros((rel_pair_idxs_for_img.size()[0], 3))
#            relation_logits_for_img[:] = torch.nn.functional.one_hot(torch.tensor(2),num_classes=3) #by default, all relations are background
            
            rel_pair_idxs_list = []
            pred_rel_scores_list = []
            for i in range(len(new_instances)):
                for j in range(len(new_instances)):
                    rel_pair_idxs_list.append([i,j])
                    if (i,j) in remapped_relation_outputs:
                        if remapped_relation_outputs[(i,j)] == 1:
                            pred_rel_scores_list.append([0,1,0])
                        elif remapped_relation_outputs[(i,j)] == 0:
                            pred_rel_scores_list.append([1,0,0])
                    else:
                        pred_rel_scores_list.append([0,0,1])
                        
            assert(len(rel_pair_idxs_list) == len(pred_rel_scores_list))
            new_pred_rel_scores = torch.tensor(pred_rel_scores_list)
            new_rel_pair_idxs = torch.tensor(rel_pair_idxs_list)
            
            
#            obj_combinations = rel_pair_idxs_for_img.cpu().numpy().tolist()
#           
#            #TODO: updated rel_pair_idxs to respect new/deleted entities 
#            obj_combinations = [tuple(x) for x in obj_combinations]
#            
#            
#            for j, obj_combination in enumerate(obj_combinations):
#                if tuple(obj_combination) in relation_outputs:
#                    corrected_obj_combination = tuple([entity_id_mapping[x] for x in obj_combination])
#                    relation_logits_for_img[j,:] = torch.nn.functional.one_hot(torch.tensor(relation_outputs[corrected_obj_combination]),num_classes=3)

            #rel_class_prob = F.softmax(rel_pair_idxs_for_img, -1)
            #rel_class_prob = relation_logits_for_img #is already one_hot
#            rel_scores, rel_class = rel_class_prob[:, :].max(dim=1)
#            rel_labels = rel_class
#            result = new_instances
#            result._rel_pair_idxs = rel_pair_idxs_for_img  # (#rel, 2)
#            result._pred_rel_scores = rel_class_prob  # (#rel, #rel_class)
#            result._pred_rel_labels = rel_labels  # (#rel, )
#            results.append(result)

            rel_scores, rel_class = new_pred_rel_scores[:, :].max(dim=1)
            rel_labels = rel_class
            result = new_instances
            result._rel_pair_idxs = new_rel_pair_idxs.cuda()  # (#rel, 2)
            result._pred_rel_scores = new_pred_rel_scores.cuda()  # (#rel, #rel_class)
            result._pred_rel_labels = rel_labels.cuda()  # (#rel, )
            results.append(result)



        return results

#            result = self.post_processor((relation_logits, refine_logits), rel_pair_idxs, boxes,
#                                         img_sizes,
#                                         segmentation_vis=self.cfg.MODEL.ROI_SCENEGRAPH_HEAD.RETURN_SEG_MASKS)
#            return None, (relation_logits, refine_logits, rel_pair_idxs, instances), None
#
#            #return img_relations_dict

    def create_new_instance_from_bboxes(self, missing_bbox, instances_for_batch):
        # adding the instance for documentroot
        new_instance = Instances(instances_for_batch.image_size)
        bbox_size = missing_bbox.getAbsoluteBoundingBox(format=BBFormat.XYX2Y2)
        box_tensor = torch.tensor([bbox_size[0],bbox_size[1],bbox_size[2],bbox_size[3]]).unsqueeze(0)
        #metabox = torch.tensor([0,0,raw_tensor["instances"].image_size[1],raw_tensor["instances"].image_size[0]]).unsqueeze(0)
        new_instance.pred_boxes = Boxes(box_tensor)
        missing_box_category = missing_bbox.getClassId()
        category_index = self.reverse_class_mapping[missing_box_category]
        new_instance.pred_classes = torch.tensor([category_index])
        
        #add score = -1 to show that this is an instance not created by the docparser
        new_instance.scores = torch.tensor([1])
        
        pred_class_prob = torch.nn.functional.one_hot(torch.tensor(category_index),num_classes=len(self.classes)+1).unsqueeze(0) #add one for background
        #boxes_per_cls = torch.tensor(1,box_tensor)
        boxes_per_cls = box_tensor.repeat((1,len(self.classes),1))
        new_instance.boxes_per_cls = boxes_per_cls
        new_instance.pred_class_prob = pred_class_prob 
        new_instance.pred_scores = pred_class_prob
        
        # now concatenate the old instances with the new instances
        
        used_device = instances_for_batch.pred_classes.get_device()
        new_instance = new_instance.to(used_device)  
        return new_instance

    def create_flat_annotation_list_for_table(self, table_structure_annotations, default_page_nr=0):
        anns = table_structure_annotations
        all_ids = set()
        for ann in anns:
            if 'id' in ann:
                # TODO: Fix missing 'id' earlier in postprocessing
                assert ann['id'] not in all_ids
                all_ids.add(ann['id'])
            if ann['category'] == 'box' and 'page' not in ann:
                ann['page'] = 0
        if len(all_ids) == 0:
            return []
        max_id = max(all_ids) + 1
        new_anns = []
        for ann in anns:
            if 'id' not in ann:
                ann['id'] = max_id
                max_id += 1
            # Move bounding box into simple nested entity
            current_ann_id = ann['id']
            new_box_id = max_id
            max_id += 1
            bbox = ann.get('bbox')
            page = ann.get('page', default_page_nr)
            new_ann = {"id": new_box_id, "category": "box", "parent": current_ann_id, "bbox": bbox,
                       'page': page}
            new_anns.append(new_ann)

        # Create parent structure entities for all annotations

        document_root_id = max_id + 1
        meta_root_id = max_id + 2
        other_root_id = max_id + 3
        max_id = max_id + 4
        parent_table_id = max_id + 5
        all_annotations = [
            {"id": other_root_id, "category": "unk", "parent": None},
            {"id": meta_root_id, "category": "meta", "parent": None},
            {"id": document_root_id, "category": "document", "parent": None},
            {"id": parent_table_id, "category": "table", "parent": document_root_id}
        ]
        for ann in anns:
            ann['parent'] = parent_table_id
        all_annotations += anns
        all_annotations += new_anns
        return all_annotations

    def get_table_structure_annotations(self, detection_result_file):
        all_annotations, img_size = get_detections_from_file(detection_result_file)
        table_structure_annotations = process_all_table_structure_annotations(all_annotations)
        flat_annotations = self.create_flat_annotation_list_for_table(table_structure_annotations)
        img_relations_dict = {'relations': None, 'flat_annotations': flat_annotations,
                              'table_structure_annotations': table_structure_annotations}
        return img_relations_dict

    def add_offset_to_annotations(self, all_anns, x0, y0, max_id,
                                  selection_categories=['table_cell', 'table_col', 'table_row']):
        table_anns = []
        ann_by_id = dict()
        ann_by_parent = defaultdict(list)

        for ann in all_anns:
            if ann['parent'] is not None:
                ann['parent'] = ann['parent'] + max_id + 1
            try:
                ann['id'] = ann['id'] + max_id + 1
            except TypeError as e:
                print(e)

        for ann in all_anns:
            ann_by_id[ann['id']] = ann
            if ann['parent'] is not None:
                ann_by_parent[ann['parent']].append(ann)
            if ann['category'] in selection_categories:
                table_anns.append(ann)

        bbox_anns = []
        for ann in table_anns:
            children = ann_by_parent[ann['id']]
            for child_ann in children:
                if 'bbox' in child_ann:
                    [x, y, w, h] = child_ann['bbox']
                    x_new = x + x0
                    y_new = y + y0
                    child_ann['bbox'] = [x_new, y_new, w, h]
                bbox_anns += [child_ann]

        return table_anns, bbox_anns

    def wrap_invalid_toplevel_bboxes(self, all_bboxes_for_img, invalid_toplevel_bboxes,
                                     sequence_relations,
                                     is_parent_relations):
        updated_bboxes = False

        if len(invalid_toplevel_bboxes) > 0:
            for invalid_toplevel_bbox in invalid_toplevel_bboxes:
                invalid_category = invalid_toplevel_bbox.getClassId()
                possible_parents = [parent for (parent, child) in
                                    HighlevelGrammar.get_allowed_hierarchies() if
                                    child == invalid_category]
                logger.debug(
                    'invalid toplevel class: {}. possible parents: {}'.format(invalid_category,
                                                                              possible_parents))
                # candidate bboxes: boxes that appear before or after in sequence
                invalid_box_id = invalid_toplevel_bbox.getBboxID()
                before_and_after_ids = set()
                for id_a, id_b, _ in sequence_relations:
                    if id_a == invalid_box_id:
                        before_and_after_ids.add(id_b)
                    elif id_b == invalid_box_id:
                        before_and_after_ids.add(id_a)
                # logger.debug('remaining canidate parents: {}'.format(before_and_after_ids))
                candidate_bboxes = [x.getBboxID() for x in all_bboxes_for_img if
                                    x.getBboxID() in before_and_after_ids and Evaluator.Evaluator.boxesIntersect(
                                        x.getAbsoluteBoundingBox(format=BBFormat.XYX2Y2),
                                        invalid_toplevel_bbox.getAbsoluteBoundingBox(
                                            format=BBFormat.XYX2Y2))]
                if len(candidate_bboxes) == 1:
                    new_parent = \
                    [x for x in all_bboxes_for_img if x.getBboxID() == candidate_bboxes[0]][0]
                    new_parent_bbox_xywh_updated = new_parent.get_union_bbox_xywh(
                        invalid_toplevel_bbox)
                    new_parent.setAbsoluteBoundingBox(*new_parent_bbox_xywh_updated)
                    updated_bboxes = True
                logger.debug(
                    "Found exactly one candidate parent, expanded new parent bbox to union")
                # logger.debug('remaining canidate parents: {}'.format(candidate_bboxes))

        return updated_bboxes, all_bboxes_for_img
        # return all_bboxes_for_img


    def merged_direct_nesting_of_same_category(self, all_bboxes_for_img, is_parent_relations):
        updated_bboxes = False
        children_by_parent = defaultdict(set)
        same_category_nestings = []
        id_to_bbox_mapping = {bbox.getBboxID(): bbox for bbox in all_bboxes_for_img}
        id_to_class_mapping = {bbox.getBboxID(): bbox.getClassId() for bbox in all_bboxes_for_img}
        for (subj, obj, rel) in is_parent_relations:  # + sequence_relations:
            children_by_parent[subj].add(obj)
            if id_to_class_mapping[subj] == id_to_class_mapping[obj]:
                same_category_nestings.append((subj, obj))

        # TODO: consider to also merge if parent has more than one child
        for parent_id, child_id in same_category_nestings:
            if len([x for x in children_by_parent[parent_id] if
                    id_to_class_mapping[x] == id_to_class_mapping[
                        parent_id]]) == 1:  # or  and id_to_class_mapping[child_id] not in HighlevelGrammar.allowed_to_be_nested():
                child_bbox = id_to_bbox_mapping[child_id]
                parent_bbox = id_to_bbox_mapping[parent_id]
                new_parent_bbox_xywh = parent_bbox.get_union_bbox_xywh(child_bbox)
                parent_bbox.setAbsoluteBoundingBox(*new_parent_bbox_xywh)
                logger.debug(
                    "merging two nested bounding boxes ({}/{}) of same category: {}".format(
                        parent_id, child_id,
                        id_to_class_mapping[
                            parent_id]))
                all_bboxes_for_img = [x for x in all_bboxes_for_img if x.getBboxID() != child_id]
                updated_bboxes = True
        return updated_bboxes, all_bboxes_for_img

#    def create_structure_for_doc(self, detection_result_file, table_mode=False,
#                                 do_postprocessing=False, max_loop_count=30):
#
#        if table_mode:
#            if do_postprocessing is False:
#                raise NotImplementedError("table structure parsing is coupled with postprocessing")
#            img_relations_dict = self.get_table_structure_annotations(detection_result_file)
#            return img_relations_dict
#        else:
#
#            allBoundingBoxes, _ = getBoundingBoxesForFile(detection_result_file, isGT=False,
#                                                          bbFormat=BBFormat.XYX2Y2,
#                                                          coordType='abs', header=True)
#            img_names = sorted(allBoundingBoxes.getBoundingBoxImageNames())
#            try:
#                img_name = img_names[0]
#            except IndexError as e:
#                logger.warning(
#                    "No image names exist for detection file: {} \nReturning empty structure, as no objects were detected".format(
#                        detection_result_file))
#                img_relations_dict = {'relations': [], 'flat_annotations': None, 'all_bboxes': []}
#                return img_relations_dict
#            assert len(img_names) == 1
#            all_bboxes_for_img = allBoundingBoxes.getBoundingBoxesByImageName(img_name)
#            all_ids_for_img = [x.getBboxID() for x in all_bboxes_for_img]
#            logger.debug('creating structure for current img: {}'.format(img_name))
#            # max_loop_count = 30
#            loop_count = 0
#            if do_postprocessing:
#                updated_bboxes = True
#                while (updated_bboxes is True):
#                    loop_count += 1
#
#                    # TODO refactor to avoid repeated fetching of currently detected relations
#
#                    # try:
#                    is_parent_relations, sequence_relations, meta_bboxes_for_img, invalid_toplevel_bboxes = generate_relations_for_image(
#                        all_bboxes_for_img,
#                        enforce_hierarchy=False)  # also allow bad parent-child relations to be generated such that they can be potentially fixed
#                    #                    except NotImplementedError: #sometimes, cycles for
#                    #                        is_parent_relations, sequence_relations, meta_bboxes_for_img, invalid_toplevel_bboxes = generate_relations_for_image(
#                    #                            all_bboxes_for_img, enforce_hierarchy=True) #also allow bad parent-child relations to be generated such that they can be potentially fixed
#
#                    if loop_count > max_loop_count:
#                        logger.debug("Exited postprocessing, because maximum loop count reached")
#                        break
#                    updated_bboxes, all_bboxes_for_img = self.align_parents_and_children(
#                        all_bboxes_for_img,
#                        is_parent_relations)
#                    if updated_bboxes:
#                        continue
#
#                    updated_bboxes, all_bboxes_for_img = self.merged_direct_nesting_of_same_category(
#                        all_bboxes_for_img,
#                        is_parent_relations)
#                    if updated_bboxes:
#                        continue
#
#                    updated_bboxes, all_bboxes_for_img = self.wrap_invalid_children(
#                        all_bboxes_for_img,
#                        is_parent_relations,
#                        all_ids_for_img)
#                    if updated_bboxes:
#                        continue
#                    updated_bboxes, all_bboxes_for_img = self.wrap_invalid_toplevel_bboxes(
#                        all_bboxes_for_img,
#                        invalid_toplevel_bboxes,
#                        sequence_relations,
#                        is_parent_relations)
#                    if updated_bboxes:
#                        continue
#
#                    # TODO, optional: after grouping columns, align/expand all grouped annotations horizontally
#                    # TODO, optional: Merge consecutive, overlapping content boxes in same column (per definition, they should only be one bbox)
#            is_parent_relations, sequence_relations, meta_bboxes_for_img, invalid_toplevel_bboxes = generate_relations_for_image(
#                all_bboxes_for_img)
#
#            flat_annotations = create_flat_annotation_list(all_bboxes_for_img, meta_bboxes_for_img,
#                                                           is_parent_relations,
#                                                           sequence_relations)
#            img_relations_dict = {'relations': is_parent_relations + sequence_relations,
#                                  'flat_annotations': flat_annotations,
#                                  'all_bboxes': all_bboxes_for_img}
#            return img_relations_dict

    def align_parents_and_children(self, all_bboxes_for_img, is_parent_relations):
        children_by_parent = defaultdict(set)
        id_to_bbox_mapping = {bbox.getBboxID(): bbox for bbox in all_bboxes_for_img}

        for (subj, obj, rel) in is_parent_relations:  # + sequence_relations:
            children_by_parent[subj].add(obj)

        new_changes = True
        updated_bboxes = False
        while (new_changes):
            new_changes = False
            for parent_id, children_ids in children_by_parent.items():
                parent_bbox = id_to_bbox_mapping[parent_id]

                for child_id in children_ids:
                    child_bbox = id_to_bbox_mapping[child_id]
                    orig_parent_bbox = list(
                        parent_bbox.getAbsoluteBoundingBox(format=BBFormat.XYWH))
                    new_parent_bbox_xywh = parent_bbox.get_union_bbox_xywh(child_bbox)
                    if new_parent_bbox_xywh != orig_parent_bbox:  # if a bbox gets updated
                        parent_bbox.setAbsoluteBoundingBox(*new_parent_bbox_xywh)
                        new_changes = True
                        updated_bboxes = True
        return updated_bboxes, all_bboxes_for_img

    def wrap_invalid_children(self, all_bboxes_for_img, is_parent_relations, all_ids_for_img):
        children_by_parent = defaultdict(set)
        children_by_parent_classes = defaultdict(set)
        id_to_bbox_mapping = {bbox.getBboxID(): bbox for bbox in all_bboxes_for_img}
        id_to_class_mapping = {bbox.getBboxID(): bbox.getClassId() for bbox in all_bboxes_for_img}

        allowed_siblings = HighlevelGrammar().get_allowed_siblings()

        updated_bboxes = False
        for (subj, obj, rel) in is_parent_relations:  # + sequence_relations:
            children_by_parent[subj].add(obj)
            children_by_parent_classes[id_to_class_mapping[subj]].add(id_to_class_mapping[obj])

        logger.debug("looking for children to wrap: {}".format(children_by_parent_classes))
        for parent_id, children_ids in children_by_parent.items():
            parent_bbox = id_to_bbox_mapping[parent_id]
            parent_category = parent_bbox.getClassId()
            if parent_category in allowed_siblings.keys():
                grammar_rule = allowed_siblings[parent_category]
                (cat1, rule, cat2, resolve) = grammar_rule
            else:
                continue
            children_categories = []
            for child_id in children_ids:
                child_bbox = id_to_bbox_mapping[child_id]
                child_category = child_bbox.getClassId()
                children_categories.append(child_category)
            if rule == "SINGLE_OR_WITH" and cat1 in children_categories:
                if any(x != cat1 and x != cat2 for x in
                       children_categories) or children_categories.count(
                        cat1) > 1:  # first condition: "are there any other children that are not of the allowed secondary category
                    if resolve == "RESOLVE_WRAP_GRAPHIC":
                        logger.debug(
                            'wrapping child bboxes of category {} (all children categories for current parent: {})'.format(
                                cat1, children_categories))
                        for child_id in children_ids:
                            child_bbox = id_to_bbox_mapping[child_id]
                            child_category = child_bbox.getClassId()
                            if child_category == cat1:
                                updated_bboxes = True
                                # create new parent bbox to wrap invalid bbox
                                # x,y,w,h = child_bbox.getAbsoluteBoundingBox(format=BBFormat.XYWH)
                                figure_bbox = BoundingBox.clone(child_bbox)
                                figure_bbox.setClassId('figure')
                                #                                #make parent bbox slightly bigger
                                #                                [x,y,w,h] = figure_bbox.getAbsoluteBoundingBox(format=BBFormat.XYWH)
                                #                                figure_bbox.setAbsoluteBoundingBox(x-1,y-1,w+2,h+2)
                                max_id = max(all_ids_for_img)
                                new_figure_id = max_id + 1
                                figure_bbox.setBboxID(new_figure_id)
                                all_bboxes_for_img.append(figure_bbox)
                                all_ids_for_img.append(new_figure_id)
                    else:
                        raise NotImplementedError
        return updated_bboxes, all_bboxes_for_img
        # return all_bboxes_for_img

    def create_gui_doc_entry(self, debug_gui_folder, flat_annotations, img_path,
                             out_tag='debugstruct-ws', page=0, total_pages=1):
        img_name = os.path.basename(img_path)

        dest_doc_id = re.sub('-\d+.png', '', img_name)
        # dest_doc_id = img_name.replace('-{}.png'.format(page), '')
        debug_doc_folder = os.path.join(debug_gui_folder, dest_doc_id)
        create_dir_if_not_exists(debug_doc_folder)
        logger.debug("Creating debug files for document {}".format(dest_doc_id))
        dest_meta_ann_name = os.path.join(dest_doc_id + '.json'.format(out_tag))
        dest_meta_ann_path = os.path.join(debug_doc_folder, dest_meta_ann_name)
        new_meta_contents = {'id': dest_doc_id, 'title': dest_doc_id, 'pages': 1}
        dest_img_path = os.path.join(debug_doc_folder, img_name)

        dest_ann_name = os.path.join(dest_doc_id + '-{}.json'.format(out_tag))
        dest_ann_path = os.path.join(debug_doc_folder, dest_ann_name)

        logger.debug('create meta file at {}'.format(dest_meta_ann_path))
        with open(dest_meta_ann_path, 'w') as out_file:
            json.dump(new_meta_contents, out_file, indent=1, sort_keys=True)

        logger.debug("Copying img from {} to {}".format(img_path, dest_img_path))
        copyfile(img_path, dest_img_path)
        logger.debug('create debug annotations at {}'.format(dest_ann_path))
        with open(dest_ann_path, 'w') as out_file:
            json.dump(flat_annotations, out_file, indent=1, sort_keys=True)


class HighlevelGrammar(object):
    AT_LEAST_ONE = 0
    AT_MOST_ONE = 1
    ANY_NUMBER = 2
    NONE = 3
    AT_LEAST_TWO = 4
    allowed_hierarchy_relations = {('figure', 'figurecaption'): ANY_NUMBER,
                                   ('figure', 'figuregraphic'): AT_MOST_ONE,
                                   ('table', 'tabular'): AT_LEAST_ONE,
                                   ('table', 'tablecaption'): ANY_NUMBER,
                                   ('itemize', 'item'): AT_LEAST_ONE,
                                   ('item', 'equation'): ANY_NUMBER,
                                   ('abstract', 'heading'): AT_MOST_ONE,
                                   ('figure', 'figure'): ANY_NUMBER}
    allowed_siblings = {
        'figure': ('figuregraphic', "SINGLE_OR_WITH", 'figurecaption', "RESOLVE_WRAP_GRAPHIC")}
    optional_hierarchies = [('item', 'equation'), ('abstract', 'heading')]
    allowed_highlevel_classes = set(
        ['figure', 'contentblock', 'bibblock', 'itemize', 'equation', 'heading', 'abstract',
         'table', 'author',
         'affiliation', 'date'])
    float_types = set(['figure', 'table'])
    float_to_caption_mapping = {'figure': 'figurecaption', 'table': 'tablecaption'}
    float_to_main_item_mapping = {'figure': 'figuregraphic', 'table': 'tabular'}
    meta_types = {'foot', 'head', 'subject', 'pagenr'}
    root_types = {'document', 'meta'}
    enforce_merge_of_nested_categories = {'contentblock'}

    special_text_content_classes = set(['bibblock', 'figurecaption', 'tablecaption', 'abstract'])

    @staticmethod
    def allowed_to_be_nested():
        all_same_category_hierarchies = set(
            [x for x in HighlevelGrammar.get_allowed_hierarchies() if x[0] == x[1]])
        return all_same_category_hierarchies

    @staticmethod
    def get_float_main_items():
        return HighlevelGrammar.float_to_main_item_mapping

    @staticmethod
    def get_special_text_content_classes():
        return HighlevelGrammar.special_text_content_classes

    @staticmethod
    def get_allowed_hierarchies():
        return set(HighlevelGrammar.allowed_hierarchy_relations.keys())

    @staticmethod
    def get_allowed_siblings():
        return HighlevelGrammar.allowed_siblings

    @staticmethod
    def get_allowed_highlevel_classes():
        return HighlevelGrammar.allowed_highlevel_classes

    @staticmethod
    def get_float_types():
        return HighlevelGrammar.float_types

    @staticmethod
    def get_float_caption_mapping():
        return HighlevelGrammar.float_to_caption_mapping

    @staticmethod
    def get_meta_types():
        return HighlevelGrammar.meta_types

    @staticmethod
    def get_root_types():
        return HighlevelGrammar.root_types

def is_in_region(bbox_x0, bbox_x1, left_region, right_region):
    bbox_x_range = bbox_x1 - bbox_x0
    if bbox_x1 < left_region[1]:
        return 0
    elif bbox_x0 > right_region[0]:
        return 2

    left_region_overlap_x0 = max(left_region[0], bbox_x0)
    left_region_overlap_x1 = min(left_region[1], bbox_x1)
    left_region_overlap_length = left_region_overlap_x1 - left_region_overlap_x0

    right_region_overlap_x0 = max(right_region[0], bbox_x0)
    right_region_overlap_x1 = min(right_region[1], bbox_x1)
    right_region_overlap_length = right_region_overlap_x1 - right_region_overlap_x0

    if (left_region_overlap_length / bbox_x_range) > 0.7:
        return 0
    elif (right_region_overlap_length / bbox_x_range) > 0.7:
        return 2
    else:
        return 1


def value_is_in_range(value, value_range):
    if value >= value_range[0] and value <= value_range[1]:
        return True


def get_y_ranges_of_bboxes(center_bboxes):
    center_y_ranges = []
    for center_bounding_box in center_bboxes:
        x0, y0, w, h = center_bounding_box.getAbsoluteBoundingBox(format=BBFormat.XYWH)
        y1 = y0 + h
        found_match = False
        for other_center_y_range in center_y_ranges:
            if value_is_in_range(y0, other_center_y_range) and value_is_in_range(y1,
                                                                                 other_center_y_range):
                found_match = True
                break
            elif value_is_in_range(y0, other_center_y_range):
                other_center_y_range[1] = y1  # extend y-range
                found_match = True
                break
            elif value_is_in_range(y1, other_center_y_range):
                other_center_y_range[0] = y0  # extend y-range
                found_match = True
                break
        if found_match is False:
            new_y_range = [y0, y1]
            center_y_ranges.append(new_y_range)

    return sorted(center_y_ranges)


def find_best_y_overlap(bbox, y_ranges):
    best_overlap_length = -1
    x0, y0, x1, y1 = bbox.getAbsoluteBoundingBox(format=BBFormat.XYX2Y2)
    best_y_range = None
    for y_range in y_ranges:
        if y0 > y_range[1]:
            continue
        elif y1 < y_range[0]:
            continue
        else:
            overlap_y0 = max(y0, y_range[0])
            overlap_y1 = min(y1, y_range[1])
            overlap_length = overlap_y1 - overlap_y0
            if best_y_range is None or overlap_length > best_overlap_length:
                best_y_range = tuple(y_range)
                best_overlap_length = overlap_length
    if best_y_range is None:
        logger.debug(
            'could not find a fitting range for bbox {} and ranges {}'.format([x0, y0, x1, y1],
                                                                              y_ranges))
    return best_y_range


def assign_center_bboxes_to_y_ranges(center_bboxes, sorted_center_y_ranges):
    sorted_center_bboxes = sorted(center_bboxes, key=lambda x: (
        x.getAbsoluteBoundingBox(format=BBFormat.XYWH)[1],
        x.getAbsoluteBoundingBox(format=BBFormat.XYWH)[0]))

    center_bboxes_by_range = dict()
    for center_bbox in sorted_center_bboxes:
        best_fit_range = find_best_y_overlap(center_bbox, sorted_center_y_ranges)
        if best_fit_range is None:
            raise NotImplementedError
        if best_fit_range not in center_bboxes_by_range:
            center_bboxes_by_range[best_fit_range] = {'center': []}
        center_bboxes_by_range[best_fit_range]['center'].append(center_bbox)
    return center_bboxes_by_range


def assign_leftright_bboxes_to_y_ranges(left_bboxes, right_bboxes, sorted_leftright_y_ranges):
    # sort by y and then by x center coordinates (in case bboxes have identical y coordinate)
    sorted_left_bboxes = sorted(left_bboxes, key=lambda x: (
        x.getAbsoluteBoundingBox(format=BBFormat.XYWH)[1],
        x.getAbsoluteBoundingBox(format=BBFormat.XYWH)[0]))
    sorted_right_bboxes = sorted(right_bboxes, key=lambda x: (
        x.getAbsoluteBoundingBox(format=BBFormat.XYWH)[1],
        x.getAbsoluteBoundingBox(format=BBFormat.XYWH)[0]))

    leftright_bboxes_by_range = dict()
    for left_bbox in sorted_left_bboxes:
        best_fit_range = find_best_y_overlap(left_bbox, sorted_leftright_y_ranges)
        if best_fit_range is None:
            raise NotImplementedError
        if best_fit_range not in leftright_bboxes_by_range:
            leftright_bboxes_by_range[best_fit_range] = {'left': [], 'right': [], 'center': []}
        leftright_bboxes_by_range[best_fit_range]['left'].append(left_bbox)

    for right_bbox in sorted_right_bboxes:
        best_fit_range = find_best_y_overlap(right_bbox, sorted_leftright_y_ranges)
        if best_fit_range is None:
            continue
        if best_fit_range not in leftright_bboxes_by_range:
            leftright_bboxes_by_range[best_fit_range] = {'left': [], 'right': [], 'center': []}
        leftright_bboxes_by_range[best_fit_range]['right'].append(right_bbox)

    return leftright_bboxes_by_range


def get_remaining_y_ranges_based_on_page_height(sorted_center_y_ranges, height):
    all_remaining_vertical_page_ranges = []
    if len(sorted_center_y_ranges) > 0:
        prev_y_end = 0
        for center_y_range in sorted_center_y_ranges:
            new_vertical_range = [prev_y_end, center_y_range[0]]
            prev_y_end = center_y_range[1]
            all_remaining_vertical_page_ranges.append(new_vertical_range)
        new_vertical_range = [prev_y_end, height]
        all_remaining_vertical_page_ranges.append(new_vertical_range)
    else:
        all_remaining_vertical_page_ranges = [[0, height]]
    return all_remaining_vertical_page_ranges


def sanity_check_grouped_bboxes(leftright_bboxes_by_y_range, center_bboxes_by_y_range, left_bboxes,
                                right_bboxes,
                                center_bboxes):
    all_left_grouped_bbox_ids = set()
    for y_range, side_group in leftright_bboxes_by_y_range.items():
        for side, group_bboxes in side_group.items():
            all_left_grouped_bbox_ids.update(x.getBboxID() for x in group_bboxes)
    assert all_left_grouped_bbox_ids == set(x.getBboxID() for x in left_bboxes + right_bboxes)

    all_center_grouped_bbox_ids = set()
    for y_range, side_group in center_bboxes_by_y_range.items():
        for side, group_bboxes in side_group.items():
            all_center_grouped_bbox_ids.update(x.getBboxID() for x in group_bboxes)
    assert all_center_grouped_bbox_ids == set(x.getBboxID() for x in center_bboxes)


def group_by_column(unsorted_bboxes, width, height):
    left_region = [0, width / 2]
    right_region = [width / 2, width]

    # Group all bounding boxes to left/center/right regions
    left_bboxes, right_bboxes, center_bboxes = [], [], []
    for bbox in unsorted_bboxes:
        x0, y0, w, h = bbox.getAbsoluteBoundingBox(format=BBFormat.XYWH)
        bboxes_by_left_center_right_region = is_in_region(x0, x0 + w, left_region, right_region)
        if bboxes_by_left_center_right_region == 0:
            left_bboxes.append(bbox)
        elif bboxes_by_left_center_right_region == 1:
            center_bboxes.append(bbox)
        elif bboxes_by_left_center_right_region == 2:
            right_bboxes.append(bbox)
        else:
            raise NotImplementedError

    # Segment the page into vertical regions that are sorted individually.
    # Segmenting based on boxes that are centered and span a large fraction of the page width
    sorted_center_y_ranges = get_y_ranges_of_bboxes(center_bboxes)
    all_remaining_vertical_page_ranges = get_remaining_y_ranges_based_on_page_height(
        sorted_center_y_ranges, height)
    all_sorted_vertical_y_ranges = sorted(
        sorted_center_y_ranges + all_remaining_vertical_page_ranges)

    leftright_bboxes_by_y_range = assign_leftright_bboxes_to_y_ranges(left_bboxes, right_bboxes,
                                                                      all_sorted_vertical_y_ranges)
    center_bboxes_by_y_range = assign_center_bboxes_to_y_ranges(center_bboxes,
                                                                all_sorted_vertical_y_ranges)
    sanity_check_grouped_bboxes(leftright_bboxes_by_y_range, center_bboxes_by_y_range, left_bboxes,
                                right_bboxes,
                                center_bboxes)

    merged_bbox_groups_by_y_range = dict()
    merged_bbox_groups_by_y_range.update(leftright_bboxes_by_y_range)
    for y_range in set(center_bboxes_by_y_range.keys()).union(
            set(merged_bbox_groups_by_y_range.keys())):
        if y_range in center_bboxes_by_y_range:
            if y_range in merged_bbox_groups_by_y_range:
                merged_bbox_groups_by_y_range[y_range]['center'] = \
                center_bboxes_by_y_range[y_range]['center']
            else:
                merged_bbox_groups_by_y_range[y_range] = center_bboxes_by_y_range[y_range]

    assert set(unsorted_bboxes) == set(left_bboxes + right_bboxes + center_bboxes)

    # sanity checks
    all_grouped_bbox_ids = set()
    for y_range, side_group in merged_bbox_groups_by_y_range.items():
        for side, group_bboxes in side_group.items():
            all_grouped_bbox_ids.update(x.getBboxID() for x in group_bboxes)
    if not all_grouped_bbox_ids == set(x.getBboxID() for x in unsorted_bboxes):
        logger.error(
            "Not all bboxes were returned in the grouped dictionary!\n original ids: {}, grouped ids: {}".format(
                set(x.getBboxID() for x in unsorted_bboxes), all_grouped_bbox_ids))
        raise AssertionError

    return merged_bbox_groups_by_y_range


def get_sequence_relations_from_sorted_bboxes_per_column(bboxes_by_column):
    y_ranges_sorted = sorted(list(bboxes_by_column.keys()))
    bboxes_ordered_in_sequence = []
    for y_range in y_ranges_sorted:
        bbox_dict_for_range = bboxes_by_column[y_range]
        if 'left' in bbox_dict_for_range:
            left_bboxes = bbox_dict_for_range['left']
            for bbox in left_bboxes:
                bbox.setColumn('left')
            bboxes_ordered_in_sequence += left_bboxes
        if 'center' in bbox_dict_for_range:
            center_bboxes = bbox_dict_for_range['center']
            for bbox in center_bboxes:
                bbox.setColumn('center')
            bboxes_ordered_in_sequence += center_bboxes
        if 'right' in bbox_dict_for_range:
            right_bboxes = bbox_dict_for_range['right']
            for bbox in right_bboxes:
                bbox.setColumn('right')
            bboxes_ordered_in_sequence += right_bboxes

    comes_before_relations = []
    for i, bbox in enumerate(bboxes_ordered_in_sequence):
        if i < len(bboxes_ordered_in_sequence) - 1:
            next_bbox = bboxes_ordered_in_sequence[i + 1]
            comes_before_relation = (bbox.getBboxID(), next_bbox.getBboxID(), "COMES_BEFORE")
            comes_before_relations.append(comes_before_relation)

    return comes_before_relations


def get_sibling_sequence_relations(sibling_bboxes_by_parent, width, height, id_to_bbox_mapping):
    all_sequence_relations = []
    all_collected_sibling_relations = []
    for parent_id, sibling_ids in sibling_bboxes_by_parent.items():
        parent_bbox = id_to_bbox_mapping[parent_id]
        parent_category = parent_bbox.getClassId()
        sibling_bboxes = [id_to_bbox_mapping[x] for x in sibling_ids]
        if parent_category in HighlevelGrammar.get_float_types():
            # TODO: This could be made more robust, perhaps also using some kind of grouping/using centroids
            sorted_siblings = sorted(sibling_bboxes, key=lambda x: (
                x.getAbsoluteBoundingBox(format=BBFormat.XYWH)[1],
                x.getAbsoluteBoundingBox(format=BBFormat.XYWH)[0]))
            new_sequence_relations = []
            for i, bbox in enumerate(sorted_siblings):
                if i < len(sorted_siblings) - 1:
                    next_bbox = sorted_siblings[i + 1]
                    all_collected_sibling_relations.append(
                        (bbox.getClassId(), next_bbox.getClassId()))
                    new_sequence_relations.append(
                        (bbox.getBboxID(), next_bbox.getBboxID(), "COMES_BEFORE"))
        else:
            bboxes_by_column = group_by_column(sibling_bboxes, width, height)
            new_sequence_relations = get_sequence_relations_from_sorted_bboxes_per_column(
                bboxes_by_column)

        # logger.debug("sibling sequence relations for {}: {}".format(sibling_ids, new_sequence_relations))
        all_sequence_relations += new_sequence_relations
    return all_sequence_relations


def get_correct_predictions(epoch_detection_results, iou, img_name):
    matches_bounding_boxes_iou = epoch_detection_results['iou'][iou]['matches_per_img']
    if img_name in matches_bounding_boxes_iou:
        correct_predictions = matches_bounding_boxes_iou[img_name]
    else:
        correct_predictions = dict()

    detection_to_gt_mapping = dict()
    correct_detection_ids = set()
    for gt_id, detection_id in correct_predictions.items():
        correct_detection_ids.add(int(detection_id))
        detection_to_gt_mapping[int(detection_id)] = int(gt_id)

    return correct_detection_ids, detection_to_gt_mapping


def determine_relation_matches(all_detected_relations, all_gt_relations, detection_to_gt_mapping,
                               correct_detection_ids):
    all_invalid_relations = set(
        [x for x in all_detected_relations if
         x[0] not in correct_detection_ids or x[1] not in correct_detection_ids])
    all_relation_candidates = all_detected_relations - all_invalid_relations

    all_relation_candidates_with_gt_ids = set()
    for relation_cand in all_relation_candidates:
        relation_with_gt_ids = (
            detection_to_gt_mapping[relation_cand[0]], detection_to_gt_mapping[relation_cand[1]],
            relation_cand[2])
        all_relation_candidates_with_gt_ids.add(relation_with_gt_ids)

    true_positives = all_gt_relations.intersection(all_relation_candidates_with_gt_ids)
    false_negatives = all_gt_relations - true_positives
    false_positives = all_relation_candidates_with_gt_ids - true_positives
    false_positives = false_positives.union(all_invalid_relations)
    return list(true_positives), list(false_positives), list(false_negatives)


def get_all_nested_bboxes(all_bboxes_for_img):
    child_bboxes_by_parent = defaultdict(list)
    for i, bbox in enumerate(all_bboxes_for_img):
        all_other_bboxes = all_bboxes_for_img[:i] + all_bboxes_for_img[i + 1:]
        for other_bbox in all_other_bboxes:
            if bbox.getAbsoluteBoundingBox(
                    format=BBFormat.XYX2Y2) == other_bbox.getAbsoluteBoundingBox(
                    format=BBFormat.XYX2Y2):
                logger.debug(
                    "Bboxes are identical for ID {} and ID {}, handle cycles in resulting graph!".format(
                        bbox.getBboxID(), other_bbox.getBboxID()))
                child_bboxes_by_parent[bbox.getBboxID()].append(other_bbox.getBboxID())
            elif Evaluator.Evaluator.firstBoxContainsSecondBox(bbox, other_bbox):
                child_bboxes_by_parent[bbox.getBboxID()].append(other_bbox.getBboxID())
            elif Evaluator.Evaluator.firstBoxContainsSecondBoxNoisy(bbox, other_bbox):
                child_bboxes_by_parent[bbox.getBboxID()].append(other_bbox.getBboxID())

    return child_bboxes_by_parent


def collect_all_circular_relations(child_bboxes_by_parent):
    all_rels = set()
    circular_rels = set()
    for parent_id, child_ids in child_bboxes_by_parent.items():
        for child_id in child_ids:
            rel = frozenset([parent_id, child_id])
            if rel in all_rels:
                logger.debug('adding rel to circular rels: {}'.format(rel))
                circular_rels.add(rel)
            else:
                all_rels.add(rel)
    return all_rels, circular_rels


def unique_relation_in_loop_by_hierarchy(circular_rel_bboxes):
    definite_rel = None
    allowed_hierarchy_relations = HighlevelGrammar.get_allowed_hierarchies()
    relation_option1 = (circular_rel_bboxes[0], circular_rel_bboxes[1])
    relation_option2 = (circular_rel_bboxes[1], circular_rel_bboxes[0])
    logger.debug("allowed hierarchies: {}".format(allowed_hierarchy_relations))
    logger.debug(
        "current hierarchy: {}".format(tuple(map(BoundingBox.getClassId, relation_option1))))
    if tuple(map(BoundingBox.getClassId, relation_option1)) in allowed_hierarchy_relations:
        definite_rel = (relation_option1[0].getBboxID(), relation_option1[1].getBboxID())
    if tuple(map(BoundingBox.getClassId, relation_option2)) in allowed_hierarchy_relations:
        if definite_rel is None:
            definite_rel = (relation_option2[0].getBboxID(), relation_option2[1].getBboxID())
        else:
            definite_rel = None  # no definite parent if both relations are allowed
    return definite_rel


def unique_relation_in_loop_by_confidence(circular_rel_bboxes, eps=0.05):
    definite_rel = None
    allowed_hierarchy_relations = HighlevelGrammar.get_allowed_hierarchies()
    relation_option1 = (circular_rel_bboxes[0], circular_rel_bboxes[1])
    relation_option2 = (circular_rel_bboxes[1], circular_rel_bboxes[0])
    if relation_option1[0].getConfidence() - relation_option1[1].getConfidence() > eps:
        definite_rel = (relation_option1.getBboxID(), relation_option2.getBboxID())
    elif relation_option2[0].getConfidence() - relation_option2[1].getConfidence() > eps:
        definite_rel = (relation_option2.getBboxID(), relation_option1.getBboxID())
    if definite_rel is not None:
        logger.debug(
            "Selected definite parent-child relation in circular relation based on confidence")
    return definite_rel


def remove_relation_from_hierarchy_dict(child_bboxes_by_parent, bad_relation):
    (bad_parent, bad_child) = bad_relation
    children_for_bad_parent = child_bboxes_by_parent[bad_parent]

    assert len(children_for_bad_parent) > 0
    if len(children_for_bad_parent) == 1:
        assert children_for_bad_parent[0] == bad_child
        child_bboxes_by_parent.pop(bad_parent, None)
    elif len(children_for_bad_parent) > 1:
        child_bboxes_by_parent[bad_parent].remove(bad_child)

    return child_bboxes_by_parent


def resolve_circular_relations(child_bboxes_by_parent, id_to_bbox_mapping):
    all_rels, circular_rels = collect_all_circular_relations(child_bboxes_by_parent)
    while (len(circular_rels) > 0):
        logger.debug("circular hierarchy relations found: {}".format(circular_rels))
        for circular_rel in circular_rels:
            logger.debug("circular rel: {}".format(list(circular_rel)))
            circular_rel_bboxes = list(map(id_to_bbox_mapping.get, list(circular_rel)))
            [rel0, rel1] = map(BoundingBox.getClassId, circular_rel_bboxes)
            [rel0_id, rel1_id] = map(BoundingBox.getBboxID, circular_rel_bboxes)
            logger.debug(
                "circular bboxes: {} with classes {} and {}".format(circular_rel_bboxes, rel0,
                                                                    rel1))
            if unique_relation_in_loop_by_hierarchy(circular_rel_bboxes) is not None:
                (allowed_parent_id, allowed_child_id) = unique_relation_in_loop_by_hierarchy(
                    circular_rel_bboxes)
                bad_relation = (allowed_child_id, allowed_parent_id)
                logger.debug("Found unique relation based on hierarchy for cycle!")
                child_bboxes_by_parent = remove_relation_from_hierarchy_dict(child_bboxes_by_parent,
                                                                             bad_relation)
            # TODO: fallback: if two annotations are identical and circular, let annotation ID decide
            elif rel0 == rel1:
                if rel0_id < rel1_id:
                    # remove the tuple where rel1 is parent
                    bad_relation = (rel1_id, rel0_id)
                else:
                    bad_relation = (rel0_id, rel1_id)
                child_bboxes_by_parent = remove_relation_from_hierarchy_dict(child_bboxes_by_parent,
                                                                             bad_relation)
                logger.debug("resolve circular relation via fallback (annotation id)!")

            else:
                logger.debug(
                    "Could not resolve unique relation based on hierarchy for cycle! Removing both relations")
                bad_relation1 = (rel0_id, rel1_id)
                bad_relation2 = (rel1_id, rel0_id)
                child_bboxes_by_parent = remove_relation_from_hierarchy_dict(child_bboxes_by_parent,
                                                                             bad_relation1)
                child_bboxes_by_parent = remove_relation_from_hierarchy_dict(child_bboxes_by_parent,
                                                                             bad_relation2)
                # remove both relations
                # raise NotImplementedError
        all_rels, circular_rels = collect_all_circular_relations(child_bboxes_by_parent)
    return child_bboxes_by_parent


def split_up_nested_hierarchies(child_bboxes_by_parent):
    cleaned_sibling_ids_by_parent = dict()
    for parent_id, child_ids in child_bboxes_by_parent.items():
        nested_child_ids = set()
        for other_parent_id, other_child_ids in child_bboxes_by_parent.items():
            if other_parent_id == parent_id:
                continue
            if any(other_parent_id == x for x in child_ids):
                for other_child_id in other_child_ids:
                    for child_id in child_ids:
                        if other_child_id == child_id:
                            nested_child_ids.add(other_child_id)
        next_level_child_ids = list(set(child_ids) - nested_child_ids)
        cleaned_sibling_ids_by_parent[parent_id] = next_level_child_ids
    return cleaned_sibling_ids_by_parent


def find_toplevel_bboxes(all_bboxes_for_img, cleaned_sibling_ids_by_parent):
    # toplevel bboxes: boxes that are not child to any other bbox
    all_child_bbox_ids = set(sum(cleaned_sibling_ids_by_parent.values(), []))
    all_toplevel_bboxes = [bbox for bbox in all_bboxes_for_img if
                           bbox.getBboxID() not in all_child_bbox_ids]
    allowed_highlevel_classes = HighlevelGrammar.get_allowed_highlevel_classes()
    valid_toplevel_bboxes = [bbox for bbox in all_toplevel_bboxes if
                             bbox.getClassId() in allowed_highlevel_classes]
    invalid_toplevel_bboxes = [bbox for bbox in all_toplevel_bboxes if
                               bbox.getClassId() not in allowed_highlevel_classes]

    all_toplevel_bbox_ids = set(
        x.getBboxID() for x in valid_toplevel_bboxes + invalid_toplevel_bboxes)
    all_toplevel_and_child_ids = all_toplevel_bbox_ids | all_child_bbox_ids
    all_bbox_ids = set(x.getBboxID() for x in all_bboxes_for_img)
    assert all_toplevel_and_child_ids == all_bbox_ids

    return valid_toplevel_bboxes, invalid_toplevel_bboxes


def ensure_unique_parents(child_bboxes_by_parent, id_to_bbox_mapping):
    cleaned_sibling_ids_by_parent = dict()
    cleaned_sibling_ids_by_parent.update(child_bboxes_by_parent)
    conflict_children = set()
    occurred_children = set()
    for parent_id, child_ids in child_bboxes_by_parent.items():
        for child_id in child_ids:
            if child_id in occurred_children:
                conflict_children.add(child_id)
            else:
                occurred_children.add(child_id)

    if len(conflict_children) > 0:
        logger.debug('conflict children: {}'.format(conflict_children))
        for conflict_child in conflict_children:
            child_bbox = id_to_bbox_mapping[conflict_child]
            shared_parents = [k for k, v in child_bboxes_by_parent.items() if
                              conflict_child in set(v)]
            parent_categories = [id_to_bbox_mapping[x].getClassId() for x in shared_parents]
            logger.debug(
                "shared parents: {} (categories: {})".format(shared_parents, parent_categories))

            relative_overlaps = [Evaluator.Evaluator.iou(
                id_to_bbox_mapping[x].getAbsoluteBoundingBox(BBFormat.XYX2Y2),
                child_bbox.getAbsoluteBoundingBox(BBFormat.XYX2Y2))
                                 for x in shared_parents]
            parent_confidences = [id_to_bbox_mapping[x].getConfidence() for x in shared_parents]
            parent_sizes = [id_to_bbox_mapping[x].getArea() for x in shared_parents]
            maximum_overlap_indices = [i for i, x in enumerate(relative_overlaps) if
                                       x == max(relative_overlaps) and max(relative_overlaps) > 0]

            maximum_overlap_and_confidence_indices = []
            max_confidence_in_subset = max(
                x for i, x in enumerate(parent_confidences) if i in maximum_overlap_indices)
            for i in maximum_overlap_indices:
                parent_confidence = parent_confidences[i]
                if parent_confidence == max_confidence_in_subset and parent_confidence > 0:
                    maximum_overlap_and_confidence_indices.append(i)

            maximum_overlap_and_confidence_and_size_indices = []
            max_size_in_subset = max(
                x for i, x in enumerate(parent_sizes) if
                i in maximum_overlap_and_confidence_indices)
            for i in maximum_overlap_and_confidence_indices:
                parent_size = parent_sizes[i]
                if parent_size == max_size_in_subset and parent_size > 0:
                    maximum_overlap_and_confidence_and_size_indices.append(i)

            if len(maximum_overlap_indices) == 1:
                best_parent_index = maximum_overlap_indices[0]
                parents_to_remove = [x for i, x in enumerate(shared_parents) if
                                     i != best_parent_index]
                logger.debug('found best parent based on IoU: {}'.format(relative_overlaps))
            elif len(maximum_overlap_and_confidence_indices) == 1:
                best_parent_index = maximum_overlap_and_confidence_indices[0]
                parents_to_remove = [x for i, x in enumerate(shared_parents) if
                                     i != best_parent_index]
                logger.debug(
                    'found best parent based on overlap and confidence: {}; overlaps:{}, confidences: {}'.format(
                        best_parent_index, relative_overlaps, parent_confidences))
            elif len(maximum_overlap_and_confidence_and_size_indices) == 1:
                best_parent_index = maximum_overlap_and_confidence_and_size_indices[0]
                parents_to_remove = [x for i, x in enumerate(shared_parents) if
                                     i != best_parent_index]
                logger.debug(
                    'found best parent based on overlap and confidence and parent_sizes: {}; overlaps:{}, confidences: {}, sizes: {}'.format(
                        best_parent_index, parent_confidences, relative_overlaps, parent_sizes))
            else:
                logger.error(
                    "Could not find parent for overlaps: {}, confidences: {}, sizes: {}".format(
                        relative_overlaps,
                        parent_confidences,
                        parent_sizes))
                raise NotImplementedError
            for k, v in cleaned_sibling_ids_by_parent.items():
                if k in set(parents_to_remove):
                    logger.debug('remove bad parent {} for child {}'.format(k, conflict_child))
                    cleaned_sibling_ids_by_parent[k].remove(conflict_child)

    # sanity check
    occurred_children = set()
    for parent_id, child_ids in cleaned_sibling_ids_by_parent.items():
        for child_id in child_ids:
            if child_id in occurred_children:
                raise AssertionError("Children groups overlap!")
            else:
                occurred_children.add(child_id)
    return cleaned_sibling_ids_by_parent


def remove_parents_based_on_hierarchy_grammar(child_bboxes_by_parent, id_to_bbox_mapping):
    valid_sibling_ids_by_parent = dict()
    allowed_hierarchy_relations = HighlevelGrammar.get_allowed_hierarchies()

    for parent_id, child_ids in child_bboxes_by_parent.items():
        parent_category = id_to_bbox_mapping[parent_id].getClassId()
        for child_id in child_ids:
            child_bbox = id_to_bbox_mapping[child_id]
            child_category = child_bbox.getClassId()
            class_relation = (parent_category, child_category)
            if class_relation in allowed_hierarchy_relations:
                if parent_id not in valid_sibling_ids_by_parent:
                    valid_sibling_ids_by_parent[parent_id] = []
                valid_sibling_ids_by_parent[parent_id].append(child_id)
            else:
                logger.debug("Invalid child removed for parent {} ({}): {}".format(parent_id,
                                                                                   class_relation[
                                                                                       0],
                                                                                   class_relation[
                                                                                       1]))

    return valid_sibling_ids_by_parent


def get_cleaned_sibling_by_parent_bboxes(all_bboxes_for_img, enforce_hierarchy=False):
    id_to_bbox_mapping = {bbox.getBboxID(): bbox for bbox in all_bboxes_for_img}
    id_to_class_mapping = {bbox.getBboxID(): bbox.getClassId() for bbox in all_bboxes_for_img}
    child_bboxes_by_parent = get_all_nested_bboxes(all_bboxes_for_img)
    # TODO: hierarchy exclusion removed for now
    if enforce_hierarchy:
        child_bboxes_by_parent = remove_parents_based_on_hierarchy_grammar(child_bboxes_by_parent,
                                                                           id_to_bbox_mapping)

    child_bboxes_by_parent = resolve_circular_relations(child_bboxes_by_parent, id_to_bbox_mapping)
    cleaned_sibling_ids_by_parent = split_up_nested_hierarchies(child_bboxes_by_parent)

    cleaned_sibling_ids_by_parent_with_unique_parents = ensure_unique_parents(
        cleaned_sibling_ids_by_parent,
        id_to_bbox_mapping)
    return cleaned_sibling_ids_by_parent_with_unique_parents


def get_gt_relations_and_filepath_mapping(eval_folder):
    gt_relations_output_json = os.path.join(eval_folder, 'groundtruths_origimg_relations.json')
    with open(gt_relations_output_json, 'r') as in_file:
        gt_relations_by_image = json.load(in_file)

    gt_img_infos_json = os.path.join(eval_folder, 'gt_img_infos.json')
    with open(gt_img_infos_json, 'r') as in_file:
        gt_img_infos = json.load(in_file)

    img_filepath_by_filename = {os.path.basename(gt_img_info['path']): gt_img_info['path'] for
                                gt_img_info in
                                gt_img_infos.values()}

    return gt_relations_by_image, img_filepath_by_filename


def all_bboxes_covered_in_relations(all_bboxes, relation_list):
    # make sure there exist relations for all bboxes
    all_bbox_ids = set(x.getBboxID() for x in all_bboxes)
    all_relation_ids = set()
    for subj, obj, _ in relation_list:
        all_relation_ids.add(subj)
        all_relation_ids.add(obj)

    if not all_relation_ids == all_bbox_ids:
        logger.debug('all relation ids: {}'.format(all_relation_ids))
        logger.debug('all bbox ids: {}'.format(all_bbox_ids))
        return False
    return True


def generate_relations_for_image(all_bboxes_for_img_including_meta, enforce_hierarchy=True):
    assert len(all_bboxes_for_img_including_meta) > 0

    (width, height) = all_bboxes_for_img_including_meta[0].getImageSize()
    width = int(width)
    height = int(height)
    meta_types = HighlevelGrammar.get_meta_types()
    root_types = HighlevelGrammar.get_root_types()
    all_bboxes_for_img = [x for x in all_bboxes_for_img_including_meta if
                          x.getClassId() not in meta_types and x.getClassId() not in root_types]
    meta_bboxes_for_img = [x for x in all_bboxes_for_img_including_meta if
                           x.getClassId() in meta_types and x.getClassId() not in root_types]
    root_bboxes_for_img = [x for x in all_bboxes_for_img_including_meta if
                           x.getClassId() in root_types]
    #TODO: at the end, stitch together relations with root annotations

    logger.debug('excluded {} annotation bboxes of meta types'.format(
        len(all_bboxes_for_img_including_meta) - len(all_bboxes_for_img)))

    id_to_bbox_mapping = {bbox.getBboxID(): bbox for bbox in all_bboxes_for_img}
    cleaned_sibling_ids_by_parent = get_cleaned_sibling_by_parent_bboxes(all_bboxes_for_img,
                                                                         enforce_hierarchy=enforce_hierarchy)

    document_root_bboxes = [x for x in root_bboxes_for_img if x.getClassId() == 'document']
    meta_root_bboxes = [x for x in root_bboxes_for_img if x.getClassId() == 'meta']
    if len(document_root_bboxes) != 1:
        logger.warning("Bad number of document root boxes: {}".format(len(document_root_bboxes)))
    if len(meta_root_bboxes) != 1:
        logger.warning("Bad number of meta root boxes: {}".format(len(meta_root_bboxes)))
    #assert len(document_root_bboxes) <= 1 and len(meta_root_bboxes) <= 1
    if len(document_root_bboxes) == 0:
        document_root_bbox = None
    else:
        confidences = [x.getConfidence() for x in document_root_bboxes]
        max_conf_idx = confidences.index(max(confidences))
        document_root_bbox = document_root_bboxes[max_conf_idx]
    if len(meta_root_bboxes) == 0:
        meta_root_bbox = None
    else:
        confidences = [x.getConfidence() for x in meta_root_bboxes]
        max_conf_idx = confidences.index(max(confidences))
        meta_root_bbox = meta_root_bboxes[max_conf_idx]

    is_parent_relations = []
    for parent_id, child_ids in cleaned_sibling_ids_by_parent.items():
        parent_bbox = id_to_bbox_mapping[parent_id]
        for child_id in child_ids:
            child_bbox = id_to_bbox_mapping[child_id]

            ids_relation = (parent_bbox.getBboxID(), child_bbox.getBboxID(), "IS_PARENT_OF")
            is_parent_relations.append(ids_relation)



    valid_toplevel_bboxes, invalid_toplevel_bboxes = find_toplevel_bboxes(all_bboxes_for_img,
                                                                          cleaned_sibling_ids_by_parent)
    all_toplevel_bboxes = valid_toplevel_bboxes + invalid_toplevel_bboxes


    #NOTE: add parent-of relations from document root to toplevel/meta
    for toplevel_bbox in all_toplevel_bboxes:
        parent_bbox = document_root_bbox
        if parent_bbox is None:
            continue
        ids_relation = (parent_bbox.getBboxID(), toplevel_bbox.getBboxID(), "IS_PARENT_OF")
        is_parent_relations.append(ids_relation)
    for meta_bbox in meta_bboxes_for_img:
        parent_bbox = meta_root_bbox
        if parent_bbox is None:
            continue
        ids_relation = (parent_bbox.getBboxID(), meta_bbox.getBboxID(), "IS_PARENT_OF")
        is_parent_relations.append(ids_relation)


    bboxes_by_column = group_by_column(all_toplevel_bboxes, width, height)
    toplevel_sequence_relations = get_sequence_relations_from_sorted_bboxes_per_column(
        bboxes_by_column)

    if len(all_toplevel_bboxes) <= 1:
        assert len(toplevel_sequence_relations) == 0
    else:
        assert all_bboxes_covered_in_relations(all_toplevel_bboxes, toplevel_sequence_relations)

    sequence_relations = toplevel_sequence_relations
    nested_sequence_relations = get_sibling_sequence_relations(cleaned_sibling_ids_by_parent, width,
                                                               height,
                                                               id_to_bbox_mapping)
    sequence_relations += nested_sequence_relations

    #NOTE: Add relation between 'document' and 'meta'
    if document_root_bbox is None or meta_root_bbox is None:
        pass
    else:
        ids_relation = (meta_root_bbox.getBboxID(), document_root_bbox.getBboxID(), "COMES_BEFORE")
        sequence_relations.append(ids_relation)

    # make sure there exist relations for all bboxes
    #all_bbox_ids = set(x.getBboxID() for x in all_bboxes_for_img)
    #NOTE: we changed the heuristics to have relations between toplevel bboxes and root annotations as well
    all_bbox_ids = set(x.getBboxID() for x in all_bboxes_for_img_including_meta)
    all_relation_ids = set()

    for subj, obj, _ in sequence_relations + is_parent_relations:
        all_relation_ids.add(subj)
        all_relation_ids.add(obj)
    if len(all_bbox_ids) > 1:
        if not all_relation_ids == all_bbox_ids:
            missing_boxes =  all_relation_ids.symmetric_difference(all_bbox_ids)
            #if missing_boxes.issubset(set([x.getBboxID() for x in root_bboxes_for_img])):
                #pass #if there is
            logger.warning(
                "Some bboxes have been left unannotated!: relation ids: {}, \nall bbox ids: {}, \nmissing: {}".format(
                    all_relation_ids,
                    all_bbox_ids,
                    missing_boxes))
            #raise AssertionError


    return is_parent_relations, sequence_relations, meta_bboxes_for_img, invalid_toplevel_bboxes


def merge_annotation_lists(base_list, new_list, new_list_page_nr):
    merged_anns_list = merge_annotation_lists_util(base_list, new_list, new_list_page_nr)
    return merged_anns_list