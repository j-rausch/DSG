import os
import json
from collections import defaultdict
from docparser.utils.data_utils import create_dir_if_not_exists


# https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
def get_iou(bbox1_tuple, bbox2_tuple):
    bbox1 = {'x1': bbox1_tuple[0], 'y1': bbox1_tuple[1], 'x2': bbox1_tuple[0] + bbox1_tuple[2],
             'y2': bbox1_tuple[1] + bbox1_tuple[3]}
    bbox2 = {'x1': bbox2_tuple[0], 'y1': bbox2_tuple[1], 'x2': bbox2_tuple[0] + bbox2_tuple[2],
             'y2': bbox2_tuple[1] + bbox2_tuple[3]}

    if bbox1['x1'] >= bbox1['x2'] or bbox1['y1'] >= bbox1['y2'] or bbox2['x1'] >= bbox2['x2'] or \
            bbox2['y1'] >= bbox2['y2']:
        return 0

    # determine the coordinates of the intersection rectangle
    x_left = max(bbox1['x1'], bbox2['x1'])
    y_top = max(bbox1['y1'], bbox2['y1'])
    x_right = min(bbox1['x2'], bbox2['x2'])
    y_bottom = min(bbox1['y2'], bbox2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bbox1['x2'] - bbox1['x1']) * (bbox1['y2'] - bbox1['y1'])
    bb2_area = (bbox2['x2'] - bbox2['x1']) * (bbox2['y2'] - bbox2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def merge_all_pdfminer_and_structure(dataset_file_id_path, pdfm_source_dir, docparser_source_dir,
                                     target_dir, struct_version='gtclean', out_tag='GTpdfm'):
    # create_dir_if_not_exists(target_dir)
    with open(dataset_file_id_path, 'r') as in_file:
        file_ids = [x.strip() for x in in_file.readlines()]
    for file_id in file_ids:
        print(file_id)
        page_nr = file_id.rsplit('_')[-1]
        print('page: {}'.format(page_nr))
        pdfminer_json = os.path.join(pdfm_source_dir, file_id, file_id + '-pdfm.json')
        struct_json = os.path.join(docparser_source_dir, file_id,
                                   file_id + '-{}.json'.format(struct_version))
        print(pdfminer_json)
        print(struct_json)
        merged_anns = merge_pdfminer_and_structure(pdfminer_json, struct_json)
        out_struct_json = os.path.join(target_dir, file_id, file_id + '-{}.json'.format(out_tag))

        create_dir_if_not_exists(os.path.join(target_dir, file_id))
        print('saving to {}'.format(out_struct_json))
        with open(out_struct_json, 'w') as out_file:
            json.dump(merged_anns, out_file)


def merge_pdfminer_and_structure(pdfminer_json, docparser_json):
    with open(pdfminer_json, 'r') as in_file:
        pdfminer_anns = json.load(in_file)
    with open(docparser_json, 'r') as in_file:
        docparser_anns = json.load(in_file)

    # all_content_lines = []
    pdfminer_parent_by_child = dict()
    pdfminer_ann_by_id = dict()
    pdfminer_bbox_to_ann_mapping = dict()
    pdfminer_children_by_parent = defaultdict(list)

    max_id_docparser = max([ann['id'] for ann in docparser_anns])
    docparser_document_root_ann = [x for x in docparser_anns if x['category'] == 'document'][0]
    docparser_document_root_ann_id = docparser_document_root_ann['id']

    # increment all pdfminer ids

    # NOTE: bboxes contain the text currently
    for ann in pdfminer_anns:
        ann['id'] = ann['id'] + max_id_docparser + 1
        pdfminer_ann_by_id[ann['id']] = ann
        # pdfminer_parent_by_child[ann['parent']] = ann
        #        if ann['category'] == 'content_line':
        #            all_content_lines.append(ann)
        if 'bbox' in ann:
            pdfminer_bbox_to_ann_mapping[tuple(ann['bbox'])] = ann
        if ann['parent'] is not None:
            ann['parent'] = ann['parent'] + max_id_docparser + 1
            pdfminer_children_by_parent[ann['parent']].append(ann)

    docparser_ann_by_id = dict()
    docparser_bbox_to_ann_mapping = dict()
    for ann in docparser_anns:
        docparser_ann_by_id[ann['id']] = ann
        if 'bbox' in ann:
            docparser_bbox_to_ann_mapping[tuple(ann['bbox'])] = ann

    pdfminer_bboxes = list(pdfminer_bbox_to_ann_mapping.keys())
    docparser_bboxes = list(docparser_bbox_to_ann_mapping.keys())
    pdfminer_bboxes_matched = []
    for pdfminer_bbox in pdfminer_bboxes:
        ious = [get_iou(pdfminer_bbox, x) for x in docparser_bboxes]
        max_iou = max(ious)
        if max_iou > 0:
            pdfminer_bboxes_matched.append(ious.index(max_iou))
        else:
            pdfminer_bboxes_matched.append(None)

    # docparser

    all_pdfminer_bboxes_to_keep = []
    for i, pdfminer_bbox in enumerate(pdfminer_bboxes):
        pdfminer_bbox_ann = pdfminer_bbox_to_ann_mapping[pdfminer_bbox]
        # pdfminer_bbox_parent_ann = pdfminer_ann_by_id[matched_bbox_ann['parent']]

        matched_docparser_bbox = pdfminer_bboxes_matched[i]
        if matched_docparser_bbox is None:
            new_parent_id = docparser_document_root_ann_id
        else:
            new_parent_id = docparser_bbox_to_ann_mapping[docparser_bboxes[matched_docparser_bbox]][
                'id']
        pdfminer_bbox_ann['parent'] = new_parent_id
        all_pdfminer_bboxes_to_keep.append(pdfminer_bbox_ann)

    print('total of {} pdfminer bboxes added'.format(len(all_pdfminer_bboxes_to_keep)))
    # input(all_pdfminer_bboxes_to_keep)
    merged_anns = docparser_anns + all_pdfminer_bboxes_to_keep
    return merged_anns

    # NOTE: make sure IDs don't overlap


def generate_structured_docs():
    for split in ['train', 'dev']:
        dataset_file_id_path = '../datasets/arxiv_mixed_full/manual3/splits/by_page/{}_ids.txt'.format(
            split)
        source_dir = '../datasets/arxiv_mixed_full/manual3/splits/by_page/{}/'.format(split)
        # target_dir = '../datasets/arxiv_mixed_full/manual3/splits/by_page_merged/{}/'.format(split)
        # merge_all_pdfminer_and_structure(dataset_file_id_path, source_dir, source_dir, source_dir)

        pdfm_source_dir = '../datasets/arxiv_mixed_full/manual3/splits/by_page/{}/'.format(split)
        docparser_source_dir = '../datasets/gui/arxiv_debug_fullpage_{}/'.format(split)

        # target_dir = '../datasets/gui/arxiv_debug_fullpage_merged_{}'.format(split)
        target_dir = '../datasets/gui/arxiv_debug_fullpage_{}/'.format(split)
        merge_all_pdfminer_and_structure(dataset_file_id_path, pdfm_source_dir,
                                         docparser_source_dir, target_dir,
                                         struct_version='wsft-fulldoc', out_tag='wsft-pdfm')


#if __name__ == '__main__':
#    generate_structured_docs()