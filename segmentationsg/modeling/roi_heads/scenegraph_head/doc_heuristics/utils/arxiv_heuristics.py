
import networkx as nx
from PIL import Image
import copy
from shutil import copyfile
from collections import defaultdict, OrderedDict
from TexSoup import TexSoup
import numpy as np
import os
import json
import re
#from pdfminer_utils import create_xml_for_page
import pdfminer

def second_bbox_contained_in_first_bbox(b1, b2, tolerance=0):
    return b1[0] - tolerance <= b2[0] and b1[1] - tolerance <= b2[1] and b1[0] + b1[2] + tolerance >= b2[0] + b2[2] and b1[1] + b1[3] + tolerance >= b2[1] + b2[3]



#
def second_bbox_is_smaller_within_tolerance(b1, b2, tolerance=0):
    return b2[2] + tolerance < b1[2] and b2[3] + tolerance < b1[3]
#
#
def bboxes_have_same_width_and_height(b1, b2):
    return b1[0] == b2[0] or b1[1] == b2[1]


def has_same_x_range_with_tolerance(b1, b2, tolerance=0):
    if (abs(b1[0] - b2[0]) <= tolerance)  and (abs((b1[0]+b1[2]) - (b2[0]+b2[2])) <= tolerance):
        return True
    else:
        return False

def has_same_y_range_with_tolerance(b1, b2, tolerance=0):
    if (abs(b1[1] - b2[1]) <= tolerance)  and (abs((b1[1]+b1[3]) - (b2[1]+b2[3])) <= tolerance):
        return True
    else:
        return False

def get_all_children_ids_with_child_dictionary(parent_id, ann_children_ids):
    final_new_children_ids = []
    for new_child_id in ann_children_ids[parent_id]:
        final_new_children_ids.append(new_child_id)
        final_new_children_ids += get_all_children_ids_with_child_dictionary(new_child_id, ann_children_ids)
    return final_new_children_ids


def move_all_bbox_annotations_by_offset(annotations, crop_x0, crop_y0):
    for ann in annotations:
        if ann['category'] == 'box':# or ann['category'] == 'bbox':
            bbox = ann['bbox']
            ann['bbox'] = [bbox[0] - crop_x0, bbox[1] - crop_y0, bbox[2], bbox[3]]
    return annotations 



def  get_all_bbox_anns_for_current_id(parent_ann_id, ann_children_ids, ann_by_id):
    all_children_ids =  get_all_children_ids_with_child_dictionary(parent_ann_id, ann_children_ids)
    all_bbox_anns = [ann_by_id[ann_id] for ann_id in all_children_ids if ann_by_id[ann_id]['category'] == 'box']
    return all_bbox_anns




def get_all_children_recursive(annotation_list, all_parent_ids):
    new_ids = []
    for ann in annotation_list:
        if ann['parent'] in all_parent_ids:
            new_ids.append(ann['id'])
    if len(new_ids) > 0:
        new_children_of_children = get_all_children_recursive(annotation_list, set(new_ids))
    else:
        new_children_of_children = []
    return list(new_ids) + new_children_of_children


#https://stackoverflow.com/questions/15800895/finding-clusters-of-numbers-in-a-list
def grouper(iterable, pixels_distance=5):
    prev = None
    group = []
    for item in iterable:
        if not prev or item - prev <= pixels_distance:
            group.append(item)
        else:
            yield group
            group = [item]
        prev = item
    if group:
        yield group


def union(bboxes):
    x = min([b[0] for b in bboxes])
    y = min([b[1] for b in bboxes])
    width = max([b[0] + b[2] for b in bboxes]) - x
    height = max([b[1] + b[3] for b in bboxes]) - y
    return [x, y, width, height]




def get_merged_bbox_from_list_of_bbox_anns(bbox_anns):
    xmin, ymin, xmax, ymax = 10000, 10000, -1, -1
    for ann in bbox_anns:
        try:
            [x0,y0,w,h] = ann['bbox']
        except KeyError as e:
            print('KeyError when trying to get bbox from ann: {}'.format(ann))
            raise
        x1 = x0+w
        y1 = y0+h
        if x0 < xmin:
            xmin = x0
        if y0 < ymin:
            ymin = y0
        if x1 > xmax:
            xmax = x1
        if y1 > ymax:
            ymax = y1
    result_w = xmax - xmin
    result_h = ymax - ymin
    return [xmin, ymin, result_w, result_h]


def remove_table_cells_contained_by_other_table_cells(annotation_list):
    if annotation_list is None:
        return

    ann_by_id = dict()
    anns_by_cat = defaultdict(list)
    ann_children_ids = defaultdict(list)
    for ann in annotation_list:
        ann_by_id[ann['id']] =  ann 
        anns_by_cat[ann['category']].append(ann)
        ann_children_ids[ann['parent']].append(ann['id'])

    if len(anns_by_cat['tabular']) > 50:
        return None
    for table_ann in anns_by_cat['table']:
        all_table_children = [ann_by_id[c] for c in ann_children_ids[table_ann['id']]]
        tabular_anns, caption_anns = [], []
        for table_child in all_table_children:
            if table_child['category'] == 'tabular':
                tabular_anns.append(table_child)

        all_tabular_bbox_anns = []
        for tabular_ann in tabular_anns:
            tabular_id = tabular_ann['id']
            tabular_child_ids = ann_children_ids[tabular_id]
            #all_tabular_children = [ann_by_id[chil_id] for child_id in tabular_child_ids]
            tabular_cell_anns = [ann_by_id[child_id] for child_id in tabular_child_ids if ann_by_id[child_id]['category'] == 'table_cell']
            tabular_cell_ids = set(x['id'] for x in tabular_cell_anns)

            #print('found {} tabular child ids and {} cell ids'.format(len(tabular_child_ids), len(tabular_cell_ids)))
            tabular_bbox_anns = []
            for tabular_cell_id in tabular_cell_ids:
                for cell_child_id in ann_children_ids[tabular_cell_id]:

                    cell_child_ann = ann_by_id[cell_child_id]
                    if cell_child_ann['category'] == 'box':
                        tabular_bbox_anns.append(cell_child_ann)
    
            all_tabular_bbox_anns += tabular_bbox_anns
            current_merged_tabular_bbox = get_merged_bbox_from_list_of_bbox_anns(tabular_bbox_anns)
            tabular_bbox_ids = set(ann['id'] for ann in tabular_bbox_anns)


            for cell_ann in tabular_cell_anns:
                child_ids_of_current_cell = ann_children_ids[cell_ann['id']]
                for cell_child_id in child_ids_of_current_cell:
                    cell_bbox_ann = ann_by_id[cell_child_id]


                    all_other_cell_bbox_ids = tabular_bbox_ids - set([cell_bbox_ann['id']])
                    all_other_cell_bbox_anns = [ann_by_id[x] for x in all_other_cell_bbox_ids]

                    if cell_bbox_ann['category'] != 'box':
                        raise AttributeError("expected 'box' as child of 'table_cell': {}".format(cell_bbox_ann))

                    #Delete cell bboxes that are full-table sized (note that this wouldnt fix a second, 'almost'-full table sized cell)
                    cell_bbox = cell_bbox_ann['bbox']
                    if any([second_bbox_contained_in_first_bbox(other_cell_bbox_ann['bbox'], cell_bbox, tolerance=5) for other_cell_bbox_ann in all_other_cell_bbox_anns]):
                        cell_ann['delete'] = True
                        cell_bbox_ann['delete'] = True

    annotation_list = [ann for ann in annotation_list if ann.get('delete', False) != True]
    return annotation_list
    
def fix_captions_in_float_annotations(annotation_list, float_type='figure', float_child_type='figure_graphic', caption_type='figure_caption'):


    ann_by_id = dict()
    anns_by_cat = defaultdict(list)
    ann_children_ids = defaultdict(list)

    for ann in annotation_list:
        ann_by_id[ann['id']] =  ann 
        anns_by_cat[ann['category']].append(ann)
        ann_children_ids[ann['parent']].append(ann['id'])

    for float_ann in anns_by_cat[float_type]:
        all_float_children = [ann_by_id[c] for c in ann_children_ids[float_ann['id']]]
        float_child_anns, caption_anns = [], []
        for float_child in all_float_children:
            if float_child['category'] == float_child_type:
                float_child_anns.append(float_child)
            elif float_child['category'] == caption_type:
                caption_anns.append(float_child)

        merged_float_child_bboxes = []
        for float_child_ann in float_child_anns:
            float_child_id = float_child_ann['id']
            float_child_children_ids =  get_all_children_ids_with_child_dictionary(float_child_id, ann_children_ids)


            float_child_bbox_anns = [ann_by_id[ann_id] for ann_id in float_child_children_ids if ann_by_id[ann_id]['category'] == 'box']
                
            current_merged_float_child_bbox = get_merged_bbox_from_list_of_bbox_anns(float_child_bbox_anns)
            merged_float_child_bboxes.append(current_merged_float_child_bbox)

        for caption_ann in caption_anns:
            caption_id = caption_ann['id']
            all_caption_child_ids =  get_all_children_ids_with_child_dictionary(caption_id, ann_children_ids)
            caption_bbox_anns = [ann_by_id[ann_id] for ann_id in all_caption_child_ids if ann_by_id[ann_id]['category'] == 'box' and 'bbox' in ann_by_id[ann_id]]
            for bbox_ann in caption_bbox_anns:
                if bbox_ann['bbox'][2] <= 2 or bbox_ann['bbox'][3] <= 2:
                    bbox_ann['delete'] = True
                elif any(second_bbox_contained_in_first_bbox(bbox_ann['bbox'], merged_float_child_bbox, tolerance=2.5) for merged_float_child_bbox in merged_float_child_bboxes):
                    bbox_ann['delete'] = True

    annotation_list = [ann for ann in annotation_list if ann.get('delete', False) != True]
    return annotation_list



def clean_up_table_annotations_and_convert_rows_or_columns(annotation_list):

    ann_by_id = dict()
    anns_by_cat = defaultdict(list)
    remove_indeces = []
    for i, ann in enumerate(annotation_list):
        if ann['category'] == 'box' and 'bbox' not in ann:
            remove_indeces.append(i)
            continue
        ann_by_id[ann['id']] =  ann 
        anns_by_cat[ann['category']].append(ann)
    for remove_index in reversed(remove_indeces):
        annotation_list.pop(remove_index)

    ann_children_ids = defaultdict(list)
    #max_id = 0
    for ann in annotation_list:
        ann_children_ids[ann['parent']].append(ann['id'])

    count_bad_cells = 0
    if len(anns_by_cat['tabular']) > 50:
        return None
    #print('total {} tables and {} tabulars in doc {}'.format(len(anns_by_cat['table']), len(anns_by_cat['tabular']), doc))
    for table_ann in anns_by_cat['table']:
        all_table_children = [ann_by_id[c] for c in ann_children_ids[table_ann['id']]]
        tabular_anns, caption_anns = [], []
        for table_child in all_table_children:
            if table_child['category'] == 'tabular':
                tabular_anns.append(table_child)
#            elif table_child['category'] == 'table_caption':
#                caption_anns.append(table_child)

        all_tabular_bbox_anns = []
        #merged_tabular_bboxes = []
        for tabular_ann in tabular_anns:
            tabular_id = tabular_ann['id']
            tabular_child_ids = ann_children_ids[tabular_id]
            #all_tabular_children = [ann_by_id[chil_id] for child_id in tabular_child_ids]
            tabular_cell_anns = [ann_by_id[child_id] for child_id in tabular_child_ids if ann_by_id[child_id]['category'] == 'table_cell']
            tabular_cell_ids = set(x['id'] for x in tabular_cell_anns)

            tabular_cell_bbox_anns = []
            for tabular_cell_id in tabular_cell_ids:
                for cell_child_id in ann_children_ids[tabular_cell_id]:
                    cell_child_ann = ann_by_id[cell_child_id]
                    if cell_child_ann['category'] == 'box' and 'bbox' in cell_child_ann: #there is a latex command '\box{}' which would lead to a 'box' category annotation unrelated to bbox
                        tabular_cell_bbox_anns.append(cell_child_ann)
    
                
            tabular_cell_bbox_ids = set(ann['id'] for ann in tabular_cell_bbox_anns)
            current_merged_tabular_bbox = get_merged_bbox_from_list_of_bbox_anns(tabular_cell_bbox_anns)
            #print('total nr of tabular bboxes: {}'.format(len(tabular_cell_bbox_ids)))
            if len(tabular_cell_bbox_ids) > 10000:
                return None

            for cell_ann in tabular_cell_anns:
                child_ids_of_current_cell = ann_children_ids[cell_ann['id']]
                for cell_child_id in child_ids_of_current_cell:
                    cell_bbox_ann = ann_by_id[cell_child_id]

                

                    all_other_cell_bbox_ids = tabular_cell_bbox_ids - set([cell_bbox_ann['id']])
                    all_other_cell_bbox_anns = [ann_by_id[x] for x in all_other_cell_bbox_ids]

                    if cell_bbox_ann['category'] != 'box' or 'bbox' not in cell_bbox_ann:
                        continue
                        raise AttributeError("expected 'box' as child of 'table_cell'")

                    #Delete cell bboxes that are full-table sized (note that this wouldnt fix a second, 'almost'-full table sized cell)
                    if second_bbox_contained_in_first_bbox(cell_bbox_ann['bbox'], current_merged_tabular_bbox, tolerance=2.5):
                        cell_ann['delete'] = True
                        cell_bbox_ann['delete'] = True

                    # Delete all cells smaller 2 pixels in width/height fully within another cell
                    elif cell_bbox_ann['bbox'][2] <= 2 or cell_bbox_ann['bbox'][3] <= 2:
                        cell_ann['delete'] = True
                        cell_bbox_ann['delete'] = True
                    #if cell is as wide as whole table
                    elif has_same_x_range_with_tolerance(cell_bbox_ann['bbox'], current_merged_tabular_bbox, tolerance=5) and not has_same_y_range_with_tolerance(cell_bbox_ann['bbox'], current_merged_tabular_bbox, tolerance=5):
                        cell_ann['category'] = 'table_row'
                    elif has_same_y_range_with_tolerance(cell_bbox_ann['bbox'], current_merged_tabular_bbox, tolerance=5) and not has_same_x_range_with_tolerance(cell_bbox_ann['bbox'], current_merged_tabular_bbox, tolerance=5):
                        cell_ann['category'] = 'table_col'


    annotation_list = [ann for ann in annotation_list if ann.get('delete', False) != True]
    


    return annotation_list

def remove_duplicate_bboxes_by_bbox(annotation_list):
    if annotation_list is None:
        return
    occurred_bbox_tuples = set()
    for ann in annotation_list:
        if ann['category'] == 'box':
            ann_bbox_tuple = tuple(ann['bbox'] + [ann['page']])
            if ann_bbox_tuple not in occurred_bbox_tuples:
                occurred_bbox_tuples.add(ann_bbox_tuple)
            else:
                ann['delete'] = True


    annotation_list = [ann for ann in annotation_list if ann.get('delete', False) != True]
    return annotation_list


def remove_duplicate_anns_by_id(annotation_list):
    if annotation_list is None:
        return
    ann_by_id = dict()
    for ann in annotation_list:
        if ann['id'] in ann_by_id:
            ann['delete'] = True
        else:
            ann_by_id[ann['id']] =  ann 


    annotation_list = [ann for ann in annotation_list if ann.get('delete', False) != True]
    return annotation_list



def delete_structure_annotations_without_children(annotation_list):
    if annotation_list is None:
        return
    remaining_children_ids = dict() 
    children_per_ann = {ann['id']:[] for ann in annotation_list}
    anns_without_parents = 1
    while(anns_without_parents > 0):
        anns_without_parents = 0
        for ann in annotation_list:
            if ann['parent'] is not None:
                if ann['parent'] in children_per_ann:
                    children_per_ann[ann['parent']].append(ann['id'])
                else:
                    #annotation is 'dangling' without a remaining parent annotation
                    #TODO: why are these boxes remaining without parent?
                    #answer: sometimes we remove structure anns, e.g. content lines. later, their children have to be cleaned up, as they are without parent
                    ann['delete'] = True
                    anns_without_parents += 1
        annotation_list = [ann for ann in annotation_list if ann.get('delete', False) != True]

    #find annotatinos without bbox child
    new_deletions = 0
    while_steps = 0
    for ann in annotation_list:
        if ann['category'] != 'box' and ann['parent'] is not None and len(children_per_ann[ann['id']]) == 0:
            ann['delete'] = True
            #print('deleting ann without bbox children: {}'.format(ann))
            new_deletions += 1
    while (new_deletions > 0):
        #while_steps += 1
        
        annotation_list = [ann for ann in annotation_list if ann.get('delete', False) != True]
        #print('{} annotations after {} loops'.format(len(annotation_list), while_steps))
#        if debug:
#            print(annotation_list)

        children_per_ann = {ann['id']:[] for ann in annotation_list}
        for ann in annotation_list:
            if ann['parent'] is not None:
                if ann['parent'] not in children_per_ann:
                    #print('parent ann missing for {}'.format(ann))
                    ann['delete'] = True
                else:
                    children_per_ann[ann['parent']].append(ann['id'])

        new_deletions = 0
        for ann in annotation_list:
            if ann['category'] != 'box' and ann['parent'] is not None and len(children_per_ann[ann['id']]) == 0:
                ann['delete'] = True
                new_deletions += 1

    #sanity check, TODO: Remove later
    children_per_ann = {ann['id']:[] for ann in annotation_list}
    children_without_parent = 1
    children_without_parent_total = 0
    while(children_without_parent > 0):
        annotation_list = [ann for ann in annotation_list if ann.get('delete', False) != True]
        children_without_parent = 0
        for ann in annotation_list:
            if ann['parent'] is not None:
                if ann['parent'] in children_per_ann:
                    children_per_ann[ann['parent']].append(ann['id'])
                else:
                    #annotation is 'dangling' without a remaining parent annotation
                    #TODO: why are these boxes remaining without parent?
                    #print('anns without parent left after no-children filtering'.format(ann))
                    ann['delete'] = True
                    children_without_parent += 1
                    children_without_parent_total += 1
    #print('{} anns without parent removed during filtering'.format(children_without_parent_total))


    return annotation_list

def correct_selected_categories_and_remove_invalid_docs(input_tuple):
    (doc, in_tag, out_tag, src_dir, dest_dir) = input_tuple

    pdf_path_for_doc = os.path.join(src_dir, doc, doc + '.pdf')

        

    if not os.path.exists(pdf_path_for_doc):
        debug_line = "{}, skipped: no pdf file found".format(doc)
        #print(debug_line)
        return 
    src_annotations_file = os.path.join(src_dir, doc, doc + '-{}.json'.format(in_tag))
    with open(src_annotations_file) as f:
        annotations = json.load(f)
    ann_by_id = {ann['id'] : ann for ann in annotations}

    if len(annotations) <= 1:
        debug_line = "{}, skipped: no annotations found".format(doc)
        #print(debug_line)
        return 


    #inline_equation_types = {'$'}
    equation_types = {'$$', 'equation', 'equation*'}
    figure_types = {'figure','figure*'}
    graphics_types = {'includegraphics','epsfig','epsffile', 'includegraphics*', 'epsfig*', 'epsffile*'}
    list_types = {'enumerate', 'itemize', 'description'}
    list_sub_types = {'item'}
    #lists with asterisk are inline, not accounting for them


    table_types = {'table','table*'}
    tabular_types = {'tabular','tabular*','tabularx','tabulary','tabu'}
    caption_types = {'caption','caption*'}


    count_bboxes = 0
    count_tables = 0
    count_figures = 0
    old_annotations_to_replace = set()
    #fix category of bounding box annotations and captions
    for ann in annotations:
        if ann['category'] == 'bbox':
            if 'bbox' in ann:  
                ann['category'] = 'box'
                bbox_page_merged = ann['bbox']
                bbox = bbox_page_merged[:-1]
                page = bbox_page_merged[-1]
                ann['bbox'] = bbox
                ann['page'] = page
                count_bboxes += 1
            else:#check for 'bbox' key to dismiss annotations that are not 'bboxes' bot got that category from a latex command
                ann['category'] = 'tex_bbox' 
        elif ann['category'] in equation_types:
            #print('equation found')
            ann['category'] = 'equation'
        elif ann['category'] in figure_types:
            #print('figure found')
            ann['category'] = 'figure'
        elif ann['category'] in graphics_types:
            #print('figure found')
            ann['category'] = 'figure_graphic'
        elif ann['category'] in table_types:
            #print('table found')
            ann['category'] = 'table'
            count_tables += 1
        elif ann['category'] in tabular_types:
            #print('tabular found')
            ann['category'] = 'tabular'
        elif ann['category'] in caption_types:
            #print('caption found')
            ann['category'] = 'caption'
        elif ann['category'] == 'header':
            #print('caption found')
            ann['category'] = 'heading'
        
    if count_bboxes == 0 or count_tables == 0:
        return #invalid doc


    dest_annotations_relpath = os.path.join(doc, doc + '-{}.json'.format(out_tag))
    dest_annotations_fullpath = os.path.join(dest_dir, dest_annotations_relpath)
    with open(dest_annotations_fullpath, 'w') as out_file:
        json.dump(annotations, out_file, indent=1, sort_keys=True)

    return doc 


def gather_all_children_for_ann_id(parent_ann_id, all_annotations, ann_by_id):
    all_children = []
    for ann in all_annotations:
        if parent_id_is_in_branch(ann, set([parent_ann_id]), ann_by_id) and ann['id'] != parent_ann_id:
            all_children.append(ann)
    return all_children


def parent_id_is_in_branch(ann, valid_parent_ids, ann_by_id):
    if ann['id'] in valid_parent_ids:
        return True
    if ann['parent'] is not None and ann['parent'] in ann_by_id:
        return parent_id_is_in_branch(ann_by_id[ann['parent']], valid_parent_ids, ann_by_id)
    else:
        return False


def clean_up_unused_categories_in_floats(input_tuple):
    (doc, in_tag, out_tag, src_dir, dest_dir) = input_tuple

    def create_fn_get_new_ann_id(max_id):
        current_max_id = max_id
        def get_new_ann_id():
            nonlocal current_max_id
            current_max_id += 1
            return current_max_id
        return get_new_ann_id
        

    src_annotations_file = os.path.join(src_dir, doc, doc + '-{}.json'.format(in_tag))
    with open(src_annotations_file) as f:
        annotations_from_file = json.load(f)
    annotations = annotations_from_file
    

    new_table_replacement_anns = [] 
    new_figure_replacement_anns = [] 

    ann_by_id = dict() 
    ann_children_ids = defaultdict(list)
    anns_by_cat = defaultdict(list)

    for ann in annotations:
        ann_by_id[ann['id']] = ann
        #ann_children_ids[a['parent']].append(a['id'])
        anns_by_cat[ann['category']].append(ann)
        if ann['parent'] is not None:
            ann_children_ids[ann['parent']].append(ann['id'])

    max_id = max(set(ann_by_id.keys()))
    get_new_ann_id = create_fn_get_new_ann_id(max_id)

    anns_to_delete_ids = set()
    for table_ann in anns_by_cat['table']:
        table_ann_id = table_ann['id']
        table_ann_children_ids =  get_all_children_ids_with_child_dictionary(table_ann_id, ann_children_ids)
        anns_to_delete_ids.update(set(table_ann_children_ids))
        all_tabulars = [ann_by_id[ann_id] for ann_id in table_ann_children_ids if ann_by_id[ann_id]['category'] == 'tabular']

        #find all 'box' annotations for tabulars and insert them under new 'table_cell' annotations
        for tabular_ann in all_tabulars:
            tabular_ann['category'] = 'tabular'
            tabular_ann['parent'] = table_ann_id
            new_table_replacement_anns += [tabular_ann]

            #tabular_ann_children = gather_all_children_for_ann_id(tabular_ann['id'], annotations, ann_by_id)
            tabular_ann_children_ids =  get_all_children_ids_with_child_dictionary(tabular_ann['id'], ann_children_ids)
            all_child_bboxes = [ann_by_id[ann_id] for ann_id in tabular_ann_children_ids if ann_by_id[ann_id]['category'] == 'box' and 'bbox' in ann_by_id[ann_id]]
            unique_bboxes = set()
            all_unique_child_bboxes = []
            for bbox_ann in all_child_bboxes:
                bbox_ann_tuple = tuple(bbox_ann['bbox'] + [bbox_ann['page']])

                if bbox_ann['bbox'][2] <= 4 or bbox_ann['bbox'][3] <= 4:
                    #disregard minimal size bboxes
                    continue
                if bbox_ann_tuple not in unique_bboxes:
                    unique_bboxes.add(bbox_ann_tuple)
                    all_unique_child_bboxes.append(bbox_ann)
            for bbox_ann in all_unique_child_bboxes:
                new_cell_ann_id =  get_new_ann_id()
                new_cell_ann = {'category':'table_cell', 'parent':tabular_ann['id'], 'id':new_cell_ann_id} 
                bbox_ann['parent'] = new_cell_ann_id
                new_table_replacement_anns += [new_cell_ann]
                new_table_replacement_anns += [bbox_ann]
         
        all_captions_ids = [ann_id for ann_id in table_ann_children_ids if ann_by_id[ann_id]['category'] == 'table_caption']

        #find all captions under the current table, update category and insert bboxes under them
        for caption_ann_id in all_captions_ids:
            #caption_ann_id = caption_ann['id']
            caption_ann = ann_by_id[caption_ann_id]
            
            caption_ann['parent'] = table_ann_id
            new_table_replacement_anns += [caption_ann]
            caption_ann_children = gather_all_children_for_ann_id(caption_ann['id'], annotations, ann_by_id)

            all_child_bboxes = [ann for ann in caption_ann_children if ann['category'] == 'box']
            unique_bboxes = set()
            for bbox_ann in all_child_bboxes:
                if bbox_ann['id'] not in unique_bboxes:
                    unique_bboxes.add(bbox_ann['id'])

                    new_content_line_ann_id =  get_new_ann_id()
                    new_content_line_ann = {'category':'content_line', 'parent':caption_ann['id'], 'id':new_content_line_ann_id} 
                    bbox_ann['parent'] = new_content_line_ann_id

                    new_table_replacement_anns += [new_content_line_ann]
                    new_table_replacement_anns += [bbox_ann]

    
    #do similar filtering for figures
      
    for figure_ann in anns_by_cat['figure']:
        figure_ann_id = figure_ann['id']
        figure_ann_children_ids =  get_all_children_ids_with_child_dictionary(figure_ann_id, ann_children_ids)
        anns_to_delete_ids.update(set(figure_ann_children_ids))
        all_graphics = [ann_by_id[ann_id] for ann_id in figure_ann_children_ids if ann_by_id[ann_id]['category'] == 'figure_graphic']

        for graphic_ann in all_graphics:
            graphic_ann['category'] = 'figure_graphic'
            graphic_ann['parent'] = figure_ann_id
            new_figure_replacement_anns += [graphic_ann]

            graphic_ann_children_ids =  get_all_children_ids_with_child_dictionary(graphic_ann['id'], ann_children_ids)
            all_child_bboxes = [ann_by_id[ann_id] for ann_id in graphic_ann_children_ids if ann_by_id[ann_id]['category'] == 'box' and 'bbox' in ann_by_id[ann_id]]
            unique_bboxes = set()
            all_unique_child_bboxes = []
            for bbox_ann in all_child_bboxes:
                bbox_ann_tuple = tuple(bbox_ann['bbox'] + [bbox_ann['page']])
                if bbox_ann['bbox'][2] <= 4 or bbox_ann['bbox'][3] <= 4:
                    #disregard minimal size bboxes
                    continue

                if bbox_ann_tuple not in unique_bboxes:
                    unique_bboxes.add(bbox_ann_tuple)
                    all_unique_child_bboxes.append(bbox_ann)
            for bbox_ann in all_unique_child_bboxes:
                bbox_ann['parent'] = graphic_ann['id'] 
                new_figure_replacement_anns += [bbox_ann]
         
        all_captions_ids = [ann_id for ann_id in figure_ann_children_ids if ann_by_id[ann_id]['category'] == 'figure_caption']

        for caption_ann_id in all_captions_ids:
            caption_ann = ann_by_id[caption_ann_id]
            
            caption_ann['parent'] = figure_ann_id
                
            new_figure_replacement_anns += [caption_ann]
            caption_ann_children = gather_all_children_for_ann_id(caption_ann['id'], annotations, ann_by_id)

            all_child_bboxes = [ann for ann in caption_ann_children if ann['category'] == 'box']
            unique_bboxes = set()
            for bbox_ann in all_child_bboxes:
                if bbox_ann['id'] not in unique_bboxes:
                    unique_bboxes.add(bbox_ann['id'])

                    new_content_line_ann_id =  get_new_ann_id()
                    new_content_line_ann = {'category':'content_line', 'parent':caption_ann['id'], 'id':new_content_line_ann_id} 
                    bbox_ann['parent'] = new_content_line_ann_id
                    new_figure_replacement_anns += [new_content_line_ann]
                    new_figure_replacement_anns += [bbox_ann]

    

    #print('deleting {} ids and adding {} table replacement anns + {} figure anns'.format(len(anns_to_delete_ids), len(new_table_replacement_anns), len(new_figure_replacement_anns))    )
    filtered_anns = [x for x in annotations if x['id'] not in anns_to_delete_ids] 
    filtered_anns += new_table_replacement_anns
    filtered_anns += new_figure_replacement_anns

    anns_by_cat = defaultdict(list)
    for ann in filtered_anns:
        anns_by_cat[ann['category']].append(ann)
    delete_ann_ids = set()
#    #delete zero anns
#    for bbox_ann in anns_by_cat['box']:
#        if 'bbox' in bbox_ann and bbox_ann
    dest_annotations_relpath = os.path.join(doc, doc + '-{}.json'.format(out_tag))
    dest_annotations_fullpath = os.path.join(dest_dir, dest_annotations_relpath)
    #print('saving to {}'.format(dest_annotations_fullpath))
    with open(dest_annotations_fullpath, 'w') as out_file:
        json.dump(filtered_anns, out_file, indent=1, sort_keys=True)
       

def use_heuristics_to_generate_rows_and_cols(input_tuple):
    (doc, in_tag, out_tag, src_dir, dest_dir) = input_tuple

    src_annotations_file = os.path.join(src_dir, doc, doc + '-{}.json'.format(in_tag))
    with open(src_annotations_file) as f:
        annotation_list = json.load(f)

    #print(annotation_list)
    def create_fn_get_new_ann_id(max_id):
        current_max_id = max_id
        def get_new_ann_id():
            nonlocal current_max_id
            current_max_id += 1
            return current_max_id
        return get_new_ann_id
    num_anns_start = len(annotation_list)

    
    ann_by_id = dict()
    anns_by_cat = defaultdict(list)
    ann_children_ids = defaultdict(list)
    #max_id = 0
    for ann in annotation_list:
        ann_children_ids[ann['parent']].append(ann['id'])
    for ann in annotation_list:
        ann_by_id[ann['id']] =  ann 
        anns_by_cat[ann['category']].append(ann)


    if len(ann_by_id) == 0:
        return #no annotations found
    max_id = max(set(ann_by_id.keys()))
    get_new_ann_id = create_fn_get_new_ann_id(max_id)


    for tabular_ann in anns_by_cat['tabular']:

        all_tabular_children_ids = get_all_children_recursive(annotation_list, set([tabular_ann['id']]))
        tabular_cell_anns = []
        tabular_row_anns = []
        #tabular_row_indeces_to_delete = set()
        for ann_id in all_tabular_children_ids:
            try:
                current_ann = ann_by_id[ann_id]
            except KeyError as e:
                print('Could not find an annotation by its id in doc {}: {}'.format(doc, e))
                return 
            if current_ann['category'] == 'table_cell':
                tabular_cell_anns.append(current_ann)
            elif current_ann['category'] == 'table_row':
                tabular_row_anns.append(current_ann)
        all_cell_bbox_anns = []
        for tabular_cell_ann in tabular_cell_anns:
            new_cell_children_ids = ann_children_ids[tabular_cell_ann['id']]
            all_cell_bbox_anns += [ann_by_id[child_id] for child_id in new_cell_children_ids if ann_by_id[child_id]['category'] == 'box']


        # Determine rows/columns bounding boxes for cells
        x_centers = defaultdict(list)
        y_centers = defaultdict(list)
        for cell_bbox_ann in all_cell_bbox_anns:
            x_centers[cell_bbox_ann['bbox'][0] + cell_bbox_ann['bbox'][2] / 2].append(cell_bbox_ann)
            y_centers[cell_bbox_ann['bbox'][1] + cell_bbox_ann['bbox'][3] / 2].append(cell_bbox_ann)

        
        x_center_values = sorted(list(x_centers.keys()))
        x_center_values_grouped = dict(enumerate(grouper(x_center_values, pixels_distance=5)))
        y_center_values = sorted(list(y_centers.keys()))
        y_center_values_grouped = dict(enumerate(grouper(y_center_values, pixels_distance=5)))

        #TODO: also check whether cells within one group are similarly wide/high (exclude multi-row/col)
        new_col_annotations = []
        for col_nr, x_center_value_group in x_center_values_grouped.items():
            all_cell_bbox_anns = []
            for x_center_value in x_center_value_group:
                if len(x_centers[x_center_value]) > 0:
                    all_cell_bbox_anns += x_centers[x_center_value]
            if len(all_cell_bbox_anns) <= 2:
                continue
            page = -1
            bboxes = []
            for cell_bbox_ann in all_cell_bbox_anns:
                page = cell_bbox_ann['page']
                bboxes.append(cell_bbox_ann['bbox'])
                cell_ann = ann_by_id[cell_bbox_ann['parent']]
                cell_ann['col_span'] = [col_nr,col_nr]
            merged_bbox = union(bboxes)
            new_col_id = get_new_ann_id()
            new_col_ann = {'category': 'table_col', 'id': new_col_id, 'parent':tabular_ann['id'], 'properties':col_nr}
            new_bbox_ann = {'category': 'box', 'id': get_new_ann_id(), 'parent':new_col_id, 'bbox':merged_bbox, 'page':page}
            new_col_annotations.append(new_col_ann)
            new_col_annotations.append(new_bbox_ann)
            #print('added new col with nr {}'.format(col_nr))


        new_row_annotations = []
        for row_nr, y_center_value_group in y_center_values_grouped.items():
            all_cell_bbox_anns = []
            for y_center_value in y_center_value_group:
                if len(y_centers[y_center_value]) > 0:
                    all_cell_bbox_anns += y_centers[y_center_value]
            if len(all_cell_bbox_anns) <= 1:
                continue
            page = -1
            bboxes = []
            for cell_bbox_ann in all_cell_bbox_anns:
                page = cell_bbox_ann['page']
                bboxes.append(cell_bbox_ann['bbox'])
                cell_ann = ann_by_id[cell_bbox_ann['parent']]
                cell_ann['row_span'] = [row_nr, row_nr]
                #cell['row'] = i + 1
            merged_bbox = union(bboxes)
            new_row_id = get_new_ann_id()
            new_row_ann = {'category': 'table_row', 'id': new_row_id, 'parent':tabular_ann['id'], 'properties':row_nr}
            new_bbox_ann = {'category': 'box', 'id': get_new_ann_id(), 'parent':new_row_id, 'bbox':merged_bbox, 'page':page}
            for i, existing_row in enumerate(tabular_row_anns):
                existing_row_bbox = [ann_by_id[child_id] for child_id in ann_children_ids[existing_row['id']] if ann_by_id[child_id]['category'] == 'box'][0]
                if second_bbox_contained_in_first_bbox(new_bbox_ann['bbox'], existing_row_bbox['bbox'], tolerance=5) or second_bbox_contained_in_first_bbox(existing_row_bbox['bbox'], new_bbox_ann['bbox'], tolerance=5):
                    #tabular_row_indeces_to_delete.append(i) 
                    existing_row['delete'] = True

            new_row_annotations.append(new_row_ann)
            new_row_annotations.append(new_bbox_ann)
            #print('added new row with nr {}'.format(row_nr))

           
        annotation_list += new_col_annotations
        annotation_list += new_row_annotations
        annotation_list = [ann for ann in annotation_list if ann.get('delete', False) != True]
 
         
    num_anns_end = len(annotation_list)
    #print('start anns: {}, end anns: {}'.format(num_anns_start, num_anns_end))
    dest_annotations_relpath = os.path.join(doc, doc + '-{}.json'.format(out_tag))
    dest_annotations_fullpath = os.path.join(dest_dir, dest_annotations_relpath)
    with open(dest_annotations_fullpath, 'w') as out_file:
        json.dump(annotation_list, out_file, indent=1, sort_keys=True)

    return doc 


def create_single_content_line_per_bbox_and_propagate_structure_anns(annotations):

    ann_by_id = dict() 
    ann_children_ids = defaultdict(list)
    anns_by_cat = defaultdict(list)
    insert_after_content_line_list = defaultdict(list)



    for ann in annotations:
        ann_by_id[ann['id']] = ann
        anns_by_cat[ann['category']].append(ann)
        if ann['parent'] is not None:
            ann_children_ids[ann['parent']].append(ann['id'])

    def create_fn_get_new_ann_id(max_id):
        current_max_id = max_id
        def get_new_ann_id():
            nonlocal current_max_id
            current_max_id += 1
            return current_max_id
        return get_new_ann_id


    max_id = max(set(ann_by_id.keys()))
    get_new_ann_id = create_fn_get_new_ann_id(max_id)

    for content_line_ann in anns_by_cat['content_line']:
        content_line_children_ids = set(ann_children_ids[content_line_ann['id']])
        current_child_bboxes = [ann_id for ann_id in content_line_children_ids if ann_by_id[ann_id]['category'] == 'box']
        bbox_id_set = set(current_child_bboxes)
        other_children = [ann_id for ann_id in content_line_children_ids if ann_id not in bbox_id_set]
        if len(current_child_bboxes) > 1:
            #create new content_line annotations for extra bboxes
            newly_created_content_lines = []
            for child_bbox_ann_id in current_child_bboxes[1:]:
                child_bbox_ann = ann_by_id[child_bbox_ann_id]
                new_content_line_ann_id =  get_new_ann_id()
                new_content_line_ann = {'category':'content_line', 'parent':content_line_ann['parent'], 'id':new_content_line_ann_id} 
                child_bbox_ann['parent'] = new_content_line_ann_id
                newly_created_content_lines.append(new_content_line_ann)
            insert_after_content_line_list[content_line_ann['id']] = newly_created_content_lines


        #getting other annotations out from below content lines
        while(len(other_children) > 0):
            ann_by_id = dict() 
            ann_children_ids = defaultdict(list)
            anns_by_cat = defaultdict(list)
            insert_after_content_line_list = defaultdict(list)
            for ann in annotations:
                ann_by_id[ann['id']] = ann
                anns_by_cat[ann['category']].append(ann)
                if ann['parent'] is not None:
                    ann_children_ids[ann['parent']].append(ann['id'])

            for content_line_ann in anns_by_cat['content_line']:
                content_line_children_ids = set(ann_children_ids[content_line_ann['id']])
                current_child_bboxes = set([ann_id for ann_id in content_line_children_ids if ann_by_id[ann_id]['category'] == 'box'])
                other_children = content_line_children_ids - current_child_bboxes
                if len(other_children) > 0:
                    newly_created_content_lines = []
                    for child_ann_id in other_children:
                        child_ann = ann_by_id[child_ann_id]
                        child_ann['parent'] = content_line_ann['parent'] #propagate children to the same level as content line


    for list_index, ann in reversed(list(enumerate(annotations))):
        if ann['id'] in insert_after_content_line_list:
            #print('inserting invidual content lines into annotation list..')
            annotations[list_index:list_index] = insert_after_content_line_list[ann['id']]

    return annotations


def column_number_of_bbox(bbox, column_ranges):
    current_bbox = bbox 
    current_bbox_x0 = current_bbox[0]
    current_bbox_x1 = current_bbox_x0 + current_bbox[2]
    tolerance = 5 
    [left_x0, left_x1] = column_ranges['left']
    if 'right' in column_ranges:
        [right_x0, right_x1] = column_ranges['right']

    if current_bbox_x0 >= left_x0 - tolerance and current_bbox_x1 <= left_x1 + tolerance:
        return 1
    elif 'right' in column_ranges and current_bbox_x0 >= right_x0 - tolerance and current_bbox_x1 <= right_x1 + tolerance:
        return 2
    elif 'right' in column_ranges and current_bbox_x0 >= left_x0 - tolerance and current_bbox_x1 <= right_x1 + tolerance: #double column bbox
        return 0
    else:
        return -1

def use_heuristics_to_order_contents(input_tuple):
    (doc, in_tag, out_tag, src_dir, dest_dir) = input_tuple

    src_annotations_file = os.path.join(src_dir, doc, doc + '-{}.json'.format(in_tag))
    with open(src_annotations_file) as f:
        annotations_from_file = json.load(f)
    annotations = annotations_from_file
    
    ann_by_id = dict() 
    ann_children_ids = defaultdict(list)
    anns_by_cat = defaultdict(list)

    for ann in annotations:
        ann_by_id[ann['id']] = ann
        #ann_children_ids[a['parent']].append(a['id'])
        anns_by_cat[ann['category']].append(ann)
        if ann['parent'] is not None:
            ann_children_ids[ann['parent']].append(ann['id'])

    all_bbox_anns = anns_by_cat['box']
    column_info =  get_columns_and_their_widths(anns_by_cat['content_line'], ann_children_ids, ann_by_id)
    if column_info is None:
        return 
    else:
        is_single_column, most_common_widths, column_ranges, all_bbox_anns_by_page = column_info
    most_common_width = most_common_widths[0][1][1] 
    #second_most_common_width = most_common_widths[1][1][1] 

   
    ann_lists_to_reinsert = dict() 
    tmp_remove_ids = set()
    for ann in annotations:
        if ann['category'] == 'content_line' or ann['category'] == 'box':
            continue
        current_ann_children_ids = ann_children_ids[ann['id']]
        current_content_line_anns = [ann_by_id[child_id] for child_id in current_ann_children_ids if ann_by_id[child_id]['category'] == 'content_line']
        current_non_content_line_anns = [ann_by_id[child_id] for child_id in current_ann_children_ids if ann_by_id[child_id]['category'] != 'content_line' and ann_by_id[child_id]['category'] != 'box']

#        if len(current_content_line_anns) == 0:
#            continue
        content_line_bbox_tuples = []
        bboxes_tuples_to_content_line_mapping = defaultdict(list)
        #bbox_ann_to_content_line_mapping = dict()
        for current_content_line_ann in current_content_line_anns:
            bbox_children = [ann_by_id[child_id] for child_id in ann_children_ids[current_content_line_ann['id']] if ann_by_id[child_id]['category'] == 'box']
            if len(bbox_children) > 1 or len(bbox_children) == 0:
                print('more than one or no bbox under content line! {}'.format(doc))
            bbox_ann = bbox_children[0]
            column_nr = column_number_of_bbox(bbox_ann['bbox'], column_ranges)
            ann_bbox_tuple = tuple(bbox_ann['bbox'] + [bbox_ann['page']] + [column_nr])
            content_line_bbox_tuples.append(ann_bbox_tuple)
#            if ann_bbox_tuple in bboxes_tuples_to_content_line_mapping:
#                print('error, duplicate bbox tuple!')
            bboxes_tuples_to_content_line_mapping[ann_bbox_tuple].append(current_content_line_ann)
        
        #get the merged bbox of structure anns
        for current_other_ann in current_non_content_line_anns:

            all_other_ann_children_ids = get_all_children_ids_with_child_dictionary(current_other_ann['id'], ann_children_ids)
            bbox_children = [ann_by_id[child_id] for child_id in all_other_ann_children_ids if ann_by_id[child_id]['category'] == 'box']
            #print('{} bbox children for other ann of type {}'.format(len(bbox_children), current_other_ann['category']))
            if len(bbox_children) < 1 :
                continue
            pages_in_children = set(bbox_child['page'] for bbox_child in bbox_children)
            first_page = sorted(list(pages_in_children))[0]
            #print('first page: {}'.format(first_page))
            bbox_children_on_first_page = [bbox_ann for bbox_ann in bbox_children if bbox_ann['page'] == first_page]
            #print('have {} bbox children of structure ann'.format(len(bbox_children_on_first_page)))
            merged_bbox = get_merged_bbox_from_list_of_bbox_anns(bbox_children_on_first_page)
            #bbox_ann = bbox_children[0]
            column_nr = column_number_of_bbox(merged_bbox, column_ranges)
            ann_bbox_tuple = tuple(merged_bbox + [first_page] + [column_nr])
            content_line_bbox_tuples.append(ann_bbox_tuple)
#            if ann_bbox_tuple in bboxes_tuples_to_content_line_mapping:
#                print('error, duplicate bbox tuple!')
            bboxes_tuples_to_content_line_mapping[ann_bbox_tuple].append(current_other_ann)
            

        if len(content_line_bbox_tuples) < 2:
            continue
        #sort by page, then column, then y0
        content_line_bbox_tuples_by_page_column_and_y = sorted(content_line_bbox_tuples, key = lambda x : (x[4], x[5], x[1], x[0]))
        #content_line_bbox_tuples_by_page_column_and_y = sorted(content_line_bbox_tuples, key = lambda x : (x[1]))
        reordered_content_lines = []
        for ann_tuple in content_line_bbox_tuples_by_page_column_and_y:
            reordered_content_lines += bboxes_tuples_to_content_line_mapping[ann_tuple]
        parent_ann_id = ann['id']
        for tmp_remove_ann in reordered_content_lines:
            tmp_remove_ids.add(tmp_remove_ann['id'])
            #tmp_remove_ann['delete'] = True
        ann_lists_to_reinsert[parent_ann_id] = reordered_content_lines

    #temporaily remove annotations
    prev_len = len(annotations)
    annotations = [ann for ann in annotations if ann['id'] not in tmp_remove_ids]
    len_diff = prev_len - len(annotations)
    #reinsert in ordered fashion
    while(len(ann_lists_to_reinsert) > 0):
        for list_index, ann in reversed(list(enumerate(annotations))):
            if ann['id'] in ann_lists_to_reinsert:
                #print('inserting invidual content lines into annotation list..')
                annotations[list_index:list_index] = ann_lists_to_reinsert.pop(ann['id'])
        
    dest_annotations_relpath = os.path.join(doc, doc + '-{}.json'.format(out_tag))
    dest_annotations_fullpath = os.path.join(dest_dir, dest_annotations_relpath)
    with open(dest_annotations_fullpath, 'w') as out_file:
        json.dump(annotations, out_file, indent=1, sort_keys=True)
         
        
    return doc
            
    
def get_columns_and_their_widths(selected_annotation_list, ann_children_ids, ann_by_id):
    all_bbox_anns_by_page = defaultdict(list)
    ann_widths_count = defaultdict(list)
    
    for candidate_ann in selected_annotation_list:
        all_candidate_bbox_anns = [ann_by_id[ann_id] for ann_id in ann_children_ids[candidate_ann['id']] if ann_by_id[ann_id]['category'] == 'box']
        for bbox_ann in all_candidate_bbox_anns:
#            current_bbox = bbox_ann['bbox'] 
            current_page = bbox_ann['page']
            all_bbox_anns_by_page[current_page].append(bbox_ann)
            ann_widths_count[(bbox_ann['bbox'][0], bbox_ann['bbox'][2])].append(bbox_ann)
   
     
    most_common_widths = sorted([[len(anns_for_width_tuple), width_tuple] for width_tuple, anns_for_width_tuple in ann_widths_count.items()], reverse=True)
    #print('most common width is : {}'.format(most_common_width))
    is_single_column = True


    if len(most_common_widths) == 0:
        print('found no content lines to identify columns with!')
        return None 

    left_column_range = most_common_widths[0][1]
    left_x0 = left_column_range[0]
    left_x1 = left_column_range[0] + left_column_range[1]


    if len(most_common_widths) == 1:
        #all annotations have same width or none exist
        print('only one column found!')
        column_ranges = {'left': [left_x0, left_x1]}
        return is_single_column, most_common_widths, column_ranges, all_bbox_anns_by_page
#    if most_common_widths[0][0] <= 40:
#        #print('most common width count is too low: {}'.format(most_common_widths[0]))
#        #print(most_common_widths)
#        return
    most_common_width = most_common_widths[0][1][1] 
    second_most_common_width = most_common_widths[1][1][1] 
    if most_common_widths[1][0] >= 0.7 * most_common_widths[0][0] and abs(most_common_width - second_most_common_width) <= 5:
        is_single_column = False


    right_column_range = most_common_widths[1][1]
    right_x0 = right_column_range[0]
    right_x1 = right_column_range[0] + left_column_range[1]
    column_ranges = {'left': [left_x0, left_x1], 'right': [right_x0, right_x1]}
        
    return is_single_column, most_common_widths, column_ranges, all_bbox_anns_by_page


def remove_bboxes_that_lie_within_equation_annotations(annotations, anns_by_cat, ann_by_id, ann_children_ids):
    #removing bboxes that lie within equations
    all_equation_bbox_tuples_by_page = defaultdict(set)
    all_equation_bbox_ids = set()
    for equation_ann in anns_by_cat['equation']:
        equation_children_ids =  get_all_children_ids_with_child_dictionary(equation_ann['id'], ann_children_ids)
        all_equation_bboxes_by_page = defaultdict(list)
        for ann_id in equation_children_ids:
            equation_bbox_ann = ann_by_id[ann_id]
            if 'bbox' in equation_bbox_ann:
                all_equation_bbox_ids.add(equation_bbox_ann['id'])
                #all_equation_bboxes_by_page[equation_bbox_ann['page']].append(equation_bbox_ann)
                all_equation_bbox_tuples_by_page[equation_bbox_ann['page']].add(tuple(equation_bbox_ann['bbox']))

    for bbox_ann in anns_by_cat['box']:
        if bbox_ann['id'] in all_equation_bbox_ids:
            #print('skipping equation bbox')
            continue
        elif bbox_ann['page'] not in all_equation_bbox_tuples_by_page:
            continue
        elif any(second_bbox_contained_in_first_bbox(eq_bbox_tuple, bbox_ann['bbox'], tolerance=3) for eq_bbox_tuple in all_equation_bbox_tuples_by_page[bbox_ann['page']]):
            #print('found bbox that lies within an equation. removing..')
            bbox_ann['delete'] = True

    annotations = [ann for ann in annotations if ann.get('delete', False) != True]
    return annotations


def use_heuristics_to_generate_amd_fix_equations(input_tuple):
    (doc, in_tag, out_tag, src_dir, dest_dir) = input_tuple

    
    src_annotations_file = os.path.join(src_dir, doc, doc + '-{}.json'.format(in_tag))
    with open(src_annotations_file) as f:
        annotations_from_file = json.load(f)
    annotations = annotations_from_file
    
    #fix content lines
    annotations = create_single_content_line_per_bbox_and_propagate_structure_anns(annotations)
    annotations = delete_structure_annotations_without_children(annotations)

    ann_by_id = dict() 
    ann_children_ids = defaultdict(list)
    anns_by_cat = defaultdict(list)

    for ann in annotations:
        ann_by_id[ann['id']] = ann
        #ann_children_ids[a['parent']].append(a['id'])
        anns_by_cat[ann['category']].append(ann)
        if ann['parent'] is not None:
            ann_children_ids[ann['parent']].append(ann['id'])

    def create_fn_get_new_ann_id(max_id):
        current_max_id = max_id
        def get_new_ann_id():
            nonlocal current_max_id
            current_max_id += 1
            return current_max_id
        return get_new_ann_id

    max_id = max(set(ann_by_id.keys()))
    get_new_ann_id = create_fn_get_new_ann_id(max_id)

    previous_equation_ann_ids = set([ann['id'] for ann in anns_by_cat['equation']])
    confirmed_previous_equation_ann_ids = set()
    all_candidate_anns = []
    #collect all high-level content lines (children of sections or root document)
    #initial classification with latex 'equation' tag is fairly inaccurate
    categories = ['document', 'section', 'equation']
    for cat in categories:
        annotations_for_current_category = anns_by_cat[cat]
        for ann_of_category in annotations_for_current_category:
            for child_ann_id in ann_children_ids[ann_of_category['id']]:
                child_ann = ann_by_id[child_ann_id]
                if child_ann['category'] == 'content_line':
                    all_candidate_anns.append(child_ann)

    column_info =  get_columns_and_their_widths(all_candidate_anns, ann_children_ids, ann_by_id)
    if column_info is None:
        return 
    else:
        is_single_column, most_common_widths, column_ranges, all_bbox_anns_by_page = column_info


    if most_common_widths[0][0] <= 40:
        return

    most_common_width = most_common_widths[0][1][1] 
    #second_most_common_width = most_common_widths[1][1][1] 
    insert_after_annotation_list = dict()

    for page, all_page_bbox_anns in all_bbox_anns_by_page.items():
        all_narrow_page_bbox_anns = [ann for ann in all_page_bbox_anns if ann['bbox'][2] < most_common_width]
        if is_single_column:
            all_narrow_page_bbox_anns_by_columns = [all_narrow_page_bbox_anns]
        else:
            all_narrow_page_bbox_anns_by_columns = [[], []]
            for bbox_ann in all_narrow_page_bbox_anns:
                current_bbox = bbox_ann['bbox']
                current_bbox_x0 = current_bbox[0]
                current_bbox_x1 = current_bbox_x0 + current_bbox[2]
                tolerance = 5 
                [left_x0, left_x1] = column_ranges['left']
                [right_x0, right_x1] = column_ranges['right']

                if current_bbox_x0 >= left_x0 - tolerance and current_bbox_x1 <= left_x1 + tolerance:
                    all_narrow_page_bbox_anns_by_columns[0].append(bbox_ann)
                elif current_bbox_x0 >= right_x0 - tolerance and current_bbox_x1 <= right_x1 + tolerance:
                    all_narrow_page_bbox_anns_by_columns[1].append(bbox_ann)
#        print('investigating {} narrow page bboxes'.format(len(all_narrow_page_bbox_anns)))
#        width_counts = dict()
#        height_counts = dict()
        
        #new_anns_insertion
        for all_narrow_page_bbox_anns_by_column in all_narrow_page_bbox_anns_by_columns:
            anns_by_their_y_center = defaultdict(list)
            for cell_bbox_ann in all_narrow_page_bbox_anns_by_column:
                anns_by_their_y_center[cell_bbox_ann['bbox'][1] + cell_bbox_ann['bbox'][3] / 2].append(cell_bbox_ann)

            y_center_values = sorted(list(anns_by_their_y_center.keys()))
            y_center_values_grouped = dict(enumerate(grouper(y_center_values, pixels_distance=5)))

            for cluster_nr, y_center_value_group in y_center_values_grouped.items():
                #all_bbox_anns = []

                    all_bbox_anns = [] 
                    for y_center_value in y_center_value_group:
                        center_value_bboxes = anns_by_their_y_center[y_center_value]
                        all_bbox_anns += center_value_bboxes 
                    #anns_by_their_y_center[y_center_value]
                    if len(all_bbox_anns) >= 2:

                        rightmost_ann = all_bbox_anns[0]
                        for bbox_ann in all_bbox_anns[1:]:
                            if bbox_ann['bbox'][0] > rightmost_ann['bbox'][0]:
                                rightmost_ann = bbox_ann
                        #print('found rightmost annotation with one left neighbor: {}'.format(rightmost_ann))
                        if rightmost_ann['bbox'][2] > 40:
                            #print('rightmost box too small')
                            continue
                        other_equation_bbox_anns = [bbox_ann for bbox_ann in all_bbox_anns if bbox_ann['id'] != rightmost_ann['id']]
                        qualifying_other_equation_bbox_anns = []
                        for other_equation_bbox_ann in other_equation_bbox_anns:
                            if other_equation_bbox_ann['bbox'][2] < 50:
                                continue
                            if second_bbox_contained_in_first_bbox(other_equation_bbox_ann['bbox'], rightmost_ann['bbox'], tolerance=5):
                                continue
                            qualifying_other_equation_bbox_anns.append(other_equation_bbox_ann)
    

                        if len(qualifying_other_equation_bbox_anns) < 1:
                            continue

                        #insert new annotations
                        new_insertion_list = []
                        parent_ann = ann_by_id[rightmost_ann['parent']]
                        if parent_ann['parent'] is not None:
                            parents_parent_ann = ann_by_id[parent_ann['parent']]
                        else:
                            parents_parent_ann = None

                        if parent_ann['category'] == 'equation':
                            parent_equation = parent_ann 
                            confirmed_previous_equation_ann_ids.add(parent_equation['id'])
                        elif parents_parent_ann is not None and parents_parent_ann['category'] == 'equation':
                            parent_equation = parents_parent_ann 
                            confirmed_previous_equation_ann_ids.add(parent_equation['id'])
                        else:
                            new_equation_ann_id =  get_new_ann_id()
                            new_equation_ann = {'category':'equation', 'parent':parent_ann['id'], 'id':new_equation_ann_id} 
                            parent_equation = new_equation_ann
                            new_insertion_list.append(parent_equation)


        #insert_after_content_line_list[content_line_ann['id']] = newly_created_content_lines
    
                        new_equation_label_ann_id =  get_new_ann_id()
                        new_equation_label_ann = {'category':'equation_label', 'parent':parent_equation['id'], 'id':new_equation_label_ann_id} 
                        new_insertion_list.append(new_equation_label_ann)
                        rightmost_ann['parent'] = new_equation_label_ann_id
                        for other_equation_ann in qualifying_other_equation_bbox_anns:
                            new_equation_formula_id =  get_new_ann_id()
                            new_equation_formula_ann = {'category':'equation_formula', 'parent':parent_equation['id'], 'id':new_equation_formula_id} 
                            other_equation_ann['parent'] = new_equation_formula_id 

                            new_insertion_list.append(new_equation_formula_ann)

                        insert_after_annotation_list[parent_ann['id']] = new_insertion_list                      
    if len(insert_after_annotation_list) > 0:
        #print('inserting {} new equations into list for {}'.format(len(insert_after_annotation_list), doc))
        for list_index, ann in reversed(list(enumerate(annotations))):
            if ann['id'] in insert_after_annotation_list:
                #print('inserting invidual content lines into annotation list..')
                annotations[list_index:list_index] = insert_after_annotation_list[ann['id']]
                
    #remove equations that were tagged based on latex but not confirmed
    invalid_equation_ann_ids = previous_equation_ann_ids - confirmed_previous_equation_ann_ids


    ann_by_id = dict() 
    ann_children_ids = defaultdict(list)
    anns_by_cat = defaultdict(list)

    for ann in annotations:
        if ann['id'] in invalid_equation_ann_ids:
            ann['category'] = 'content_line'

        ann_by_id[ann['id']] = ann
        anns_by_cat[ann['category']].append(ann)
        if ann['parent'] is not None:
            ann_children_ids[ann['parent']].append(ann['id'])


    remove_bboxes_that_lie_within_equation_annotations(annotations, anns_by_cat, ann_by_id, ann_children_ids)
    annotations = remove_recursive_content_lines(annotations, ann_children_ids, anns_by_cat, ann_by_id)

            
    annotations = create_single_content_line_per_bbox_and_propagate_structure_anns(annotations)
    annotations = delete_structure_annotations_without_children(annotations)
                        #if the rightmost annotation is very narrow and left ann is significantly wider, classify them as equation and equation label
                    
        
    dest_annotations_relpath = os.path.join(doc, doc + '-{}.json'.format(out_tag))
    dest_annotations_fullpath = os.path.join(dest_dir, dest_annotations_relpath)
    with open(dest_annotations_fullpath, 'w') as out_file:
        json.dump(annotations, out_file, indent=1, sort_keys=True)
         
        
    return doc

def remove_recursive_content_lines(annotations, ann_children_ids, anns_by_cat, ann_by_id):
    all_content_line_anns = anns_by_cat['content_line']
    all_content_line_ann_ids = [ann['id'] for ann in all_content_line_anns]
#    visited_nodes = set()
#    connected_groups = set()
    G = nx.DiGraph()
    for content_line_ann in all_content_line_anns:
        if content_line_ann['parent'] in all_content_line_ann_ids:
            G.add_edge(content_line_ann['parent'], content_line_ann['id'])
    weakly_connected_components = list(nx.weakly_connected_components(G))
    #print('found {} weakly connected components'.format(len(weakly_connected_components)))
    recursive_content_line_ids = []
    for c in weakly_connected_components:
        subgraph = G.subgraph(c)
        sorted_nodes = list(nx.topological_sort(subgraph))
        non_content_line_children_ids = []
        for non_root_node in sorted_nodes[1:]:
            recursive_content_line_ids.append(non_root_node)
            non_content_line_children_ids += [ann_id for ann_id in ann_children_ids[non_root_node] if ann_by_id[ann_id]['category'] != 'content_line']
        for child_id in non_content_line_children_ids:
            ann_by_id[child_id]['parent'] = sorted_nodes[0]
        #print('sorted nodes: {}'.format(sorted_nodes))
        #print('parent of second node ({}): {}'.format(sorted_nodes[1], ann_by_id[sorted_nodes[1]]['parent']))
    annotations = [ann for ann in annotations if ann['id'] not in recursive_content_line_ids] 
    return annotations

def remove_unused_annotation_classes(input_tuple):
    (doc, in_tag, out_tag, src_dir, dest_dir) = input_tuple


    root_types = {'document', 'meta'}
    list_types = {'enumerate', 'itemize', 'description'}
    list_sub_types = {'item'}
    equation_types = {'equation'}
    float_types = {'figure', 'table'}
    float_sub_types = {'figure_graphic', 'tabular', 'caption', 'figure_caption', 'table_caption'}
    bbox_types = {'box'}
    section_types = {'section', 'heading'}
    meta_section_types = ('thebibliography','affiliation','affil','author','title','abstract', 'date')
     
    content_line_types = {'paragraph', 'content', 'content_line'}
    structure_ann_types = set.union(list_types, float_types, meta_section_types, equation_types)

    all_non_content_types = set.union(structure_ann_types, section_types, root_types, bbox_types, float_sub_types, list_sub_types)

    src_annotations_file = os.path.join(src_dir, doc, doc + '-{}.json'.format(in_tag))
    with open(src_annotations_file) as f:
        annotations_from_file = json.load(f)
    annotations = annotations_from_file
  
    
 
    ann_by_id = dict() 
    anns_by_cat = defaultdict(list) 
    table_ann_dict = dict()
    new_table_replacement_anns = [] 
    table_anns_to_delete = []
    ann_children_ids = {ann['id']: [] for ann in annotations} 
    remove_ann_ids = set()
    root_ann_ids = set()

    for ann in annotations:
        if ann['category'] in content_line_types:
            ann['category'] = 'content_line'  #fix category that wasnt fixed in prior step
        elif ann['category'] not in all_non_content_types:
            ann['category'] = 'content_line'
        if ann['parent'] is not None:
            ann_children_ids[ann['parent']].append(ann['id'])
        else:
            root_ann_ids.add(ann['id'])

        ann_by_id[ann['id']] = ann
        anns_by_cat[ann['category']].append(ann)

    if len(annotations) > 100000:
        print('over 100000 annotations in {}. skipping.. '.format(doc))
        return
        
    #remove structure annotations without children
    #print('{} root ann ids'.format(len(root_ann_ids)))
    all_bbox_anns = anns_by_cat['box']
    all_bbox_ids = [ann['id'] for ann in all_bbox_anns]
    anns_without_children = set([ann_id for ann_id, ann_children_ids in ann_children_ids.items() if len(ann_children_ids) == 0 and ann_id not in root_ann_ids and ann_id not in all_bbox_ids])
    annotations = [ann for ann in annotations if ann['id'] not in anns_without_children] 
    #print('{}, {} anns left after filtering out {} annotations without children'.format(doc, len(annotations), len(anns_without_children)))
  
    annotations = remove_recursive_content_lines(annotations, ann_children_ids, anns_by_cat, ann_by_id)
            
    #refresh ann children ids
    ann_children_ids = {ann['id']: [] for ann in annotations} 
    anns_by_cat = defaultdict(list) 
    for ann in annotations:
        if ann['parent'] is not None:
            ann_children_ids[ann['parent']].append(ann['id'])
        anns_by_cat[ann['category']].append(ann)

    #remove unknown categories in float environments
    figure_anns = anns_by_cat['figure']
    for figure_ann in figure_anns:
        all_figure_child_ids = get_all_children_ids_with_child_dictionary(figure_ann['id'], ann_children_ids)
        for child_id in all_figure_child_ids:
            child_ann = ann_by_id[child_id]
            if child_ann['category'] == 'caption':
                child_ann['category'] = 'figure_caption'

    table_anns = anns_by_cat['table']
    for table_ann in table_anns:
        all_table_child_ids = get_all_children_ids_with_child_dictionary(table_ann['id'], ann_children_ids)
        for child_id in all_table_child_ids:
            child_ann = ann_by_id[child_id]
            if child_ann['category'] == 'caption':
                child_ann['category'] = 'table_caption'

    high_priority_ids = set()
    duplicate_ids_to_remove = set()
    for struct_ann_type in structure_ann_types:
        all_anns_of_type = anns_by_cat[struct_ann_type]
        for ann_of_type in all_anns_of_type:
            unique_child_bboxes_for_current_ann = set()
            all_struct_child_ids = get_all_children_ids_with_child_dictionary(ann_of_type['id'], ann_children_ids)
            all_bbox_children = [ann_by_id[ann_id] for ann_id in all_struct_child_ids if 'bbox' in ann_by_id[ann_id]]
            for bbox_ann in all_bbox_children:
                bbox_ann_tuple = tuple(bbox_ann['bbox'] + [bbox_ann['page']])
                if bbox_ann_tuple not in unique_child_bboxes_for_current_ann:
                    unique_child_bboxes_for_current_ann.add(bbox_ann_tuple)
                else:
                    if struct_ann_type not in float_types: #treat floats separately later
                        duplicate_ids_to_remove.add(bbox_ann['id'])
            high_priority_ids.update(unique_child_bboxes_for_current_ann)
    
    #print('filtered out {} duplicate bboxes within structure anns'.format(len(duplicate_ids_to_remove)))

    all_low_prio_bbox_anns = [ann for ann in anns_by_cat['box'] if ann['id'] not in high_priority_ids and 'bbox' in ann]

    unique_bboxes = set()
    for bbox_ann in all_low_prio_bbox_anns:
        bbox_ann_tuple = tuple(bbox_ann['bbox'] + [bbox_ann['page']])
        if bbox_ann_tuple not in unique_bboxes:
            unique_bboxes.add(bbox_ann_tuple)
        else:
            duplicate_ids_to_remove.add(bbox_ann['id'])
        
    annotations = [ann for ann in annotations if ann['id'] not in duplicate_ids_to_remove] 
    #print('{} anns remaining after removing {} duplicate bboxes'.format(len(annotations), len(duplicate_ids_to_remove)))

    #remove remaining annotations without children
    ann_children_ids = {ann['id']: [] for ann in annotations} 
    for ann in annotations:
        if ann['parent'] is not None:
            ann_children_ids[ann['parent']].append(ann['id'])
    anns_without_children = set([ann_id for ann_id, ann_children_ids in ann_children_ids.items() if len(ann_children_ids) == 0 and ann_id not in root_ann_ids and ann_id not in all_bbox_ids])
    annotations = [ann for ann in annotations if ann['id'] not in anns_without_children] 


            
    dest_annotations_relpath = os.path.join(doc, doc + '-{}.json'.format(out_tag))
    dest_annotations_fullpath = os.path.join(dest_dir, dest_annotations_relpath)
    with open(dest_annotations_fullpath, 'w') as out_file:
        json.dump(annotations, out_file, indent=1, sort_keys=True)
 
    return doc 

def assign_rowcol_numbers_to_cells_and_rowscols(input_tuple):
    (doc, in_tag, out_tag, src_dir, dest_dir) = input_tuple

    src_annotations_file = os.path.join(src_dir, doc, doc + '-{}.json'.format(in_tag))
    with open(src_annotations_file) as f:
        annotation_list = json.load(f)
    annotations = annotation_list

    annotations = remove_duplicate_bboxes_by_bbox(annotations)
    annotations = remove_duplicate_anns_by_id(annotations)
    annotations = delete_structure_annotations_without_children(annotations)
    


    ann_by_id = dict()
    anns_by_cat = defaultdict(list)
    ann_children_ids = defaultdict(list)
    for ann in annotations:
        if ann['id'] in ann_by_id:
            print('duplicates in list!')
        ann_by_id[ann['id']] =  ann 
        anns_by_cat[ann['category']].append(ann)
        ann_children_ids[ann['parent']].append(ann['id'])

    
    for tabular_ann in anns_by_cat['tabular']:
        #cell_anns = []
        cell_ann_bbox_tuples = []
        bbox_to_cell_mapping = dict()
        row_anns = []
        col_anns = []
        col_x_centers = defaultdict(list) 
        row_y_centers = defaultdict(list) 
        #bbox_to_ann
        tabular_child_ids = ann_children_ids[tabular_ann['id']]
        for child_id in tabular_child_ids:
            child_ann = ann_by_id[child_id]
            child_ann_bboxes = [ann_by_id[bbox_ann_id] for bbox_ann_id in ann_children_ids[child_ann['id']] if ann_by_id[bbox_ann_id]['category'] == 'box']
            if len(child_ann_bboxes) > 1:
                print('more than one bbox for table child. unexpected, {}: \n{}'.format(child_ann['category'], child_ann_bboxes))
            if len(child_ann_bboxes) == 0:
                child_ann['delete'] = True
                continue
            child_bbox_ann = child_ann_bboxes[0]

            x_center = child_bbox_ann['bbox'][0] + child_bbox_ann['bbox'][2] / 2
            y_center = child_bbox_ann['bbox'][1] + child_bbox_ann['bbox'][3] / 2

            if child_ann['category'] == 'table_cell':
                bbox_tuple = tuple(child_bbox_ann['bbox'])
                cell_ann_bbox_tuples.append(bbox_tuple)
                bbox_to_cell_mapping[bbox_tuple] = child_ann
            elif child_ann['category'] == 'table_row':
                #if y_center in row_y_centers:
                    #print('y center already exists in dictionary for {}'.format(doc))
                row_y_centers[y_center].append([child_ann, child_bbox_ann])

            elif child_ann['category'] == 'table_col':
                #if x_center in col_x_centers:
                    #print('x center already exists in dictionary for {}'.format(doc))
                col_x_centers[x_center].append([child_ann, child_bbox_ann])

        x_center_values = sorted(list(col_x_centers.keys()))
        y_center_values = sorted(list(row_y_centers.keys()))
        for col_nr, x_center_value in enumerate(x_center_values):
            for col_ann, _ in col_x_centers[x_center_value]:
                col_ann['col_nr'] = col_nr  
                col_ann['properties'] = col_nr  
        for row_nr, y_center_value in enumerate(y_center_values):
            for row_ann, _ in row_y_centers[y_center_value]:
                row_ann['row_nr'] = row_nr  
                row_ann['properties'] = row_nr  

        for bbox_tuple in cell_ann_bbox_tuples:
            x0 = bbox_tuple[0]
            x1 = bbox_tuple[0] + bbox_tuple[2]
            cell_width = bbox_tuple[2]
            y0 = bbox_tuple[1]
            y1 = bbox_tuple[1] + bbox_tuple[3]
            cell_width = bbox_tuple[3]
            col_start =None
            col_end = None
            row_start = None
            row_end =  None
            for col_nr, x_center_value in enumerate(x_center_values):
                [_, col_bbox_ann] = col_x_centers[x_center_value][0]
                col_x0 = col_bbox_ann['bbox'][0]
                col_x1 = col_bbox_ann['bbox'][0] + col_bbox_ann['bbox'][2]
                col_length = col_bbox_ann['bbox'][2]
                overlap = 0
                if x1 <= col_x0:
                    overlap = 0
                    continue
                elif x0 >= col_x1:
                    overlap = 0
                    continue
                overlap_x0 = max(x0, col_x0)
                overlap_x1 = min(x1, col_x1)
                overlap_length = overlap_x1 - overlap_x0
                if overlap_length >= col_length * 0.5 or overlap_length >= 0.75 * cell_width:
                    col_end = col_nr
                    if col_start is None:
                        col_start = col_nr 
            for row_nr, y_center_value in enumerate(y_center_values):
                [_, row_bbox_ann] = row_y_centers[y_center_value][0]
                row_y0 = row_bbox_ann['bbox'][1]
                row_y1 = row_bbox_ann['bbox'][1] + row_bbox_ann['bbox'][3]
                row_length = row_bbox_ann['bbox'][3]
                overlap = 0
                if y1 <= row_y0:
                    overlap = 0
                    continue
                elif y0 >= row_y1:
                    overlap = 0
                    continue
                overlap_y0 = max(y0, row_y0)
                overlap_y1 = min(y1, row_y1)
                overlap_length = overlap_y1 - overlap_y0
                if overlap_length >= row_length * 0.5 or overlap_length >= 0.75 * cell_width:
                    row_end = row_nr
                    if row_start is None:
                        row_start = row_nr 
            cell_ann = bbox_to_cell_mapping[bbox_tuple]
            cell_ann['row_range'] = [row_start, row_end]
            cell_ann['col_range'] = [col_start, col_end]
            cell_ann['properties'] = "{}-{},{}-{}".format(row_start, row_end, col_start, col_end)

    annotations = [ann for ann in annotations if ann.get('delete', False) != True]

    dest_annotations_relpath = os.path.join(doc, doc + '-{}.json'.format(out_tag))
    dest_annotations_fullpath = os.path.join(dest_dir, dest_annotations_relpath)
    with open(dest_annotations_fullpath, 'w') as out_file:
        json.dump(annotations, out_file, indent=1, sort_keys=True)
 
    return doc 


def merge_list_categories_and_fix_bibliography_cat(input_tuple):
    (doc, in_tag, out_tag, src_dir, dest_dir) = input_tuple

    src_annotations_file = os.path.join(src_dir, doc, doc + '-{}.json'.format(in_tag))
    with open(src_annotations_file) as f:
        annotation_list = json.load(f)
    annotations = annotation_list

    ann_by_id = dict()
    anns_by_cat = defaultdict(list)
    ann_children_ids = defaultdict(list)
    for ann in annotations:
        ann_by_id[ann['id']] =  ann 
        anns_by_cat[ann['category']].append(ann)
        ann_children_ids[ann['parent']].append(ann['id'])


    list_types = {'enumerate', 'itemize', 'description'}
    bib_types = ('thebibliography')

    for ann in annotations:
        if ann['category'] in list_types:
            ann['category'] = 'itemize'
        elif ann['category'] == 'thebibliography':
            ann['category'] = 'bibliography'
            

    dest_annotations_relpath = os.path.join(doc, doc + '-{}.json'.format(out_tag))
    dest_annotations_fullpath = os.path.join(dest_dir, dest_annotations_relpath)
    with open(dest_annotations_fullpath, 'w') as out_file:
        json.dump(annotations, out_file, indent=1, sort_keys=True)

    return doc

            
def filter_out_only_subset_of_structure_anns(input_tuple):
    (doc, in_tag, out_tag, src_dir, dest_dir) = input_tuple

    src_annotations_file = os.path.join(src_dir, doc, doc + '-{}.json'.format(in_tag))
    with open(src_annotations_file) as f:
        annotation_list = json.load(f)
    annotations = annotation_list

    ann_by_id = dict()
    anns_by_cat = defaultdict(list)
    ann_children_ids = defaultdict(list)
    for ann in annotations:
        ann_by_id[ann['id']] =  ann 
        anns_by_cat[ann['category']].append(ann)
        ann_children_ids[ann['parent']].append(ann['id'])


    root_types = {'document', 'meta', 'section'}
    list_types = {'enumerate', 'itemize', 'description'}
    equation_types = {'equation'}
    float_types = {'figure', 'table'}
    section_types = {'heading'}
    text_types = {'content_block'}
    meta_section_types = ('thebibliography','abstract')
    types_to_be_preserved = set.union(list_types, equation_types, float_types, section_types, text_types, meta_section_types)

    ann_ids_to_preserve = set()
    for root_type in root_types:
        for root_type_ann in anns_by_cat[root_type]:
            ann_ids_to_preserve.add(root_type_ann['id'])
    for nested_type in types_to_be_preserved:
        for nested_type_ann in anns_by_cat[nested_type]:
            ann_ids_to_preserve.add(nested_type_ann['id'])
            nested_children_ids = get_all_children_ids_with_child_dictionary(nested_type_ann['id'], ann_children_ids)
            ann_ids_to_preserve.update(set(nested_children_ids))
            
    annotations = [ann for ann in annotations if ann['id'] in ann_ids_to_preserve]

    dest_annotations_relpath = os.path.join(doc, doc + '-{}.json'.format(out_tag))
    dest_annotations_fullpath = os.path.join(dest_dir, dest_annotations_relpath)
    with open(dest_annotations_fullpath, 'w') as out_file:
        json.dump(annotations, out_file, indent=1, sort_keys=True)

    return doc


def split_doc_by_page(input_tuple):
    (doc, in_tag, out_tag, src_dir, dest_dir) = input_tuple

    src_doc_folder = os.path.join(src_dir, doc)
    return_values = []

    src_annotations_file = os.path.join(src_dir, doc, doc + '-{}.json'.format(in_tag))
    with open(src_annotations_file) as f:
        annotation_list = json.load(f)


    #folder_contents = os.listdir(src_doc_folder)
    pages_with_tables_figure_abstract_lists = defaultdict(set)


    annotations = annotation_list
    ann_by_id = dict()
    anns_by_cat = defaultdict(list)
    ann_children_ids = defaultdict(list)


    for ann in annotations:
        ann_by_id[ann['id']] =  ann 
        anns_by_cat[ann['category']].append(ann)
        ann_children_ids[ann['parent']].append(ann['id'])

    #for cat in ['table', 'figure', 'itemize', 'enumerate', 'abstract', 'thebibliography', 'equation']:
    for cat in ['table', 'figure', 'itemize', 'enumerate', 'abstract', 'thebibliography']:
        for parent_ann in anns_by_cat[cat]:
            child_bbox_anns = get_all_bbox_anns_for_current_id(parent_ann['id'], ann_children_ids, ann_by_id)
            pages = set(child_bbox_ann['page'] for child_bbox_ann in child_bbox_anns)
            pages_with_tables_figure_abstract_lists[cat].update(pages)
       
    all_relevant_page_sets = [x for x in pages_with_tables_figure_abstract_lists.values()] 
    if all(len(x) == 0 for x in all_relevant_page_sets):
        print('no relevant pages found in {}: {}'.format(doc, all_relevant_page_sets))
        return
    all_relevant_pages = set.union(*all_relevant_page_sets)

    all_bboxes_in_doc = anns_by_cat['box']
    for page in all_relevant_pages:
       
        dest_doc_id =  doc + '_{}'.format(page)
        dest_doc_dir = os.path.join(dest_dir, dest_doc_id)

        src_image = os.path.join(doc + '-{}.png'.format(page))

        src_image_path = os.path.join(src_doc_folder, src_image) 
        try:
            src_fullsize_image = Image.open(src_image_path)
        except FileNotFoundError as e:
            #print('could not find page {} ({}) of {}'.format(page, src_image_path, doc))
            return_values.append(None)
            continue
        #width, height = src_fullsize_image.size

        #dest_image = os.path.join(dest_doc_id + '-{}.png'.format(page))
        #NOTE: we save only one image per document and set the page of all annotations to 0
        dest_image = os.path.join(dest_doc_id + '-0.png')
        dest_image_path = os.path.join(dest_doc_dir, dest_image)

        #src_ann_name = os.path.join(doc + '-{}.json'.format(in_tag))
        dest_ann_name = os.path.join(dest_doc_id + '-{}.json'.format(out_tag))

        #src_ann_path = os.path.join(src_doc_folder, src_ann_name)
        dest_ann_path = os.path.join(dest_doc_dir, dest_ann_name)




        bboxes_to_remove_ids_for_page = set(bbox_ann['id'] for bbox_ann in all_bboxes_in_doc if bbox_ann['page'] != page)
        annotations_for_page = copy.deepcopy(annotations)
        annotations_for_page = [ann for ann in annotations_for_page if ann['id'] not in bboxes_to_remove_ids_for_page] 
        annotations_for_page = delete_structure_annotations_without_children(annotations_for_page)
        for ann in annotations_for_page:
            if 'page' in ann:
                ann['page'] = 0
        #set remaining page numbers to '0' in the bbox annotations


        if not os.path.exists(dest_doc_dir):
            os.mkdir(dest_doc_dir)


        dest_meta_ann_name = os.path.join(dest_doc_id + '.json'.format(out_tag))
        dest_meta_ann_path = os.path.join(dest_doc_dir, dest_meta_ann_name)
        new_meta_contents = {'id': dest_doc_id, 'title':dest_doc_id, 'pages':1}
        #print('new meta json: {}'.format(dest_meta_ann_path))
        with open(dest_meta_ann_path, 'w') as out_file:
            json.dump(new_meta_contents, out_file, indent=1, sort_keys=True)


        #print('copy image from {} to {}'.format(src_image, dest_image)) 
        copyfile(src_image_path, dest_image_path)
        #print('create subset annotations from {} to {}'.format(src_annotations_file, dest_ann_name)) 
        with open(dest_ann_path, 'w') as out_file:
            json.dump(annotations_for_page, out_file, indent=1, sort_keys=True)
        return_values.append(dest_doc_id)

    return return_values

def split_doc_by_tabular(input_tuple):
    (doc, in_tag, out_tag, src_dir, dest_dir) = input_tuple

    src_doc_folder = os.path.join(src_dir, doc)
    return_values = []

    src_annotations_file = os.path.join(src_dir, doc, doc + '-{}.json'.format(in_tag))
    with open(src_annotations_file) as f:
        annotation_list = json.load(f)


    #pages_with_tables_figure_abstract_lists = defaultdict(set)


    annotations = annotation_list
    ann_by_id = dict()
    anns_by_cat = defaultdict(list)
    ann_children_ids = defaultdict(list)


    for ann in annotations:
        ann_by_id[ann['id']] =  ann 
        anns_by_cat[ann['category']].append(ann)
        ann_children_ids[ann['parent']].append(ann['id'])


    all_bbox_anns = anns_by_cat['box']
    for table_nr, table_ann in enumerate(anns_by_cat['table']):
        
        page = 0 #NOTE: Hardcoded, because of prior splitting into single pages

        child_bbox_anns = get_all_bbox_anns_for_current_id(table_ann['id'], ann_children_ids, ann_by_id)
        valid_child_bbox_ann_ids = set([ann['id'] for ann in child_bbox_anns])
        invalid_bbox_ids = [ann['id'] for ann in all_bbox_anns if ann['id'] not in valid_child_bbox_ann_ids]
        table_annotations_subset = copy.deepcopy(annotations)
        table_annotations_subset = [ann for ann in table_annotations_subset if ann['id'] not in invalid_bbox_ids] 
        table_annotations_subset = delete_structure_annotations_without_children(table_annotations_subset)


        sub_ann_by_id = dict()
        sub_anns_by_cat = defaultdict(list)
        sub_ann_children_ids = defaultdict(list)


        for ann in table_annotations_subset:
            sub_ann_by_id[ann['id']] =  ann 
            sub_anns_by_cat[ann['category']].append(ann)
            sub_ann_children_ids[ann['parent']].append(ann['id'])


        for tabular_nr, tabular_ann in enumerate(sub_anns_by_cat['tabular']):

            tabular_child_bbox_anns = get_all_bbox_anns_for_current_id(tabular_ann['id'], ann_children_ids, ann_by_id)
            tabular_valid_child_bbox_ann_ids = set([ann['id'] for ann in tabular_child_bbox_anns])
            tabular_invalid_bbox_ids = [ann['id'] for ann in all_bbox_anns if ann['id'] not in tabular_valid_child_bbox_ann_ids]
            tabular_annotations_subset = copy.deepcopy(table_annotations_subset)
            tabular_annotations_subset = [ann for ann in tabular_annotations_subset if ann['id'] not in tabular_invalid_bbox_ids] 
            tabular_annotations_subset = delete_structure_annotations_without_children(tabular_annotations_subset)




        
       
            dest_doc_id =  doc + '_table{}_tabular{}'.format(table_nr, tabular_nr)
            dest_doc_dir = os.path.join(dest_dir, dest_doc_id)

            src_image = os.path.join(doc + '-{}.png'.format(page))

            src_image_path = os.path.join(src_doc_folder, src_image) 
            try:
                src_fullsize_image = Image.open(src_image_path)
            except FileNotFoundError as e:
                print('could not find page {} ({}) of {}'.format(page, src_image_path, doc))
                return_values.append(None)
                continue

            width, height = src_fullsize_image.size

            tabular_bbox = get_merged_bbox_from_list_of_bbox_anns(tabular_child_bbox_anns)
            [x0, y0, w, h] = tabular_bbox
            x1 = x0 + w
            y1 = y0 + h

            x_margin = 0.05 * w
            y_margin = 0.05 * h


            crop_x0 = max(0, int(x0-x_margin))
            crop_y0 = max(0, int(y0-y_margin))
            crop_x1 = min(width, int(x1+x_margin))
            crop_y1 = min(height, int(y1+y_margin))

            if crop_x0 < 0 or crop_x1 > width or crop_y0 < 0 or crop_y1 > height or crop_x0 > crop_x1 or crop_y0 > crop_y1:
                print('crop region out of bounds for {}, table nr {}, tabular nr {}'.format(doc, table_nr, tabular_nr))
                return_values.append(None)
                continue
                



            im_tabular_crop = src_fullsize_image.crop((crop_x0, crop_y0, crop_x1, crop_y1))
            crop_info_name = dest_doc_id + '_cropinfo-{}.txt'.format(out_tag)
            dest_crop_info_path = os.path.join(dest_doc_dir, crop_info_name)

            tabular_annotations_subset = move_all_bbox_annotations_by_offset(tabular_annotations_subset, crop_x0, crop_y0)

            #debug
            unique_ids = set()
            for ann in tabular_annotations_subset:
                if ann['id'] not in unique_ids:
                    unique_ids.add(ann['id'])
                else:
                    print('duplicate ann detected for doc {}. type: {}, id: {}'.format(doc, ann['category'], ann['id']))

            dest_image = os.path.join(dest_doc_id + '-0.png')
            dest_image_path = os.path.join(dest_doc_dir, dest_image)

            dest_ann_name = os.path.join(dest_doc_id + '-{}.json'.format(out_tag))
            dest_ann_path = os.path.join(dest_doc_dir, dest_ann_name)

            if not os.path.exists(dest_doc_dir):
                os.mkdir(dest_doc_dir)

            dest_meta_ann_name = os.path.join(dest_doc_id + '.json'.format(out_tag))
            dest_meta_ann_path = os.path.join(dest_doc_dir, dest_meta_ann_name)
            new_meta_contents = {'id': dest_doc_id, 'title':dest_doc_id, 'pages':1}
            with open(dest_meta_ann_path, 'w') as out_file:
                json.dump(new_meta_contents, out_file, indent=1, sort_keys=True)


            crop_info_text = "crop info for source file: {} and dest file: {}, page: {}, table id: {}, tabular id: {} (format: crop_x0; crop_y0; crop_x1, crop_y1) \n{}, {}, {}, {}".format(os.path.basename(src_annotations_file), dest_ann_path, page, table_ann['id'], tabular_ann['id'], crop_x0, crop_y0, crop_x1, crop_y1)
            with open(dest_crop_info_path, 'w') as out_file:
                out_file.write(crop_info_text)


            #copyfile(src_image_path, dest_image_path)
            try:
                im_tabular_crop.save(dest_image_path)
            except (SystemError, AttributeError) as e:
                print('orig shape: {}, img shape: {}, crops: {}'.format(src_fullsize_image.size, im_tabular_crop.size, [crop_x0, crop_x1, crop_y0, crop_y1]))
                print('could not save image for table nr {}/tabular nr {} of {}'.format(table_nr, tabular_nr, doc))
                return_values.append(None)
                #continue
                raise
            #dest_fullsize_image.save(dest_image_path)
            with open(dest_ann_path, 'w') as out_file:
                json.dump(tabular_annotations_subset, out_file, indent=1, sort_keys=True)
            return_values.append(dest_doc_id)

    return return_values



def split_doc_by_table(input_tuple):
    (doc, in_tag, out_tag, src_dir, dest_dir) = input_tuple

    src_doc_folder = os.path.join(src_dir, doc)
    return_values = []

    src_annotations_file = os.path.join(src_dir, doc, doc + '-{}.json'.format(in_tag))
    with open(src_annotations_file) as f:
        annotation_list = json.load(f)


    #pages_with_tables_figure_abstract_lists = defaultdict(set)


    annotations = annotation_list
    ann_by_id = dict()
    anns_by_cat = defaultdict(list)
    ann_children_ids = defaultdict(list)


    for ann in annotations:
        ann_by_id[ann['id']] =  ann 
        anns_by_cat[ann['category']].append(ann)
        ann_children_ids[ann['parent']].append(ann['id'])


    all_bbox_anns = anns_by_cat['box']
    for table_nr, table_ann in enumerate(anns_by_cat['table']):
        
        page = 0 #NOTE: Hardcoded, because of prior splitting into single pages

        child_bbox_anns = get_all_bbox_anns_for_current_id(table_ann['id'], ann_children_ids, ann_by_id)
        valid_child_bbox_ann_ids = set([ann['id'] for ann in child_bbox_anns])
        invalid_bbox_ids = [ann['id'] for ann in all_bbox_anns if ann['id'] not in valid_child_bbox_ann_ids]
        table_annotations_subset = copy.deepcopy(annotations)
        table_annotations_subset = [ann for ann in table_annotations_subset if ann['id'] not in invalid_bbox_ids] 
        table_annotations_subset = delete_structure_annotations_without_children(table_annotations_subset)
        
       
        dest_doc_id =  doc + '_table{}'.format(table_nr)
        dest_doc_dir = os.path.join(dest_dir, dest_doc_id)

        src_image = os.path.join(doc + '-{}.png'.format(page))

        src_image_path = os.path.join(src_doc_folder, src_image) 
        try:
            src_fullsize_image = Image.open(src_image_path)
        except FileNotFoundError as e:
            print('could not find page {} ({}) of {}'.format(page, src_image_path, doc))
            return_values.append(None)
            continue

        width, height = src_fullsize_image.size

        table_bbox = get_merged_bbox_from_list_of_bbox_anns(child_bbox_anns)
        [x0, y0, w, h] = table_bbox
        x1 = x0 + w
        y1 = y0 + h

        x_margin = 0.05 * w
        y_margin = 0.05 * h


        crop_x0 = max(0, int(x0-x_margin))
        crop_y0 = max(0, int(y0-y_margin))
        crop_x1 = min(width, int(x1+x_margin))
        crop_y1 = min(height, int(y1+y_margin))

        if crop_x0 < 0 or crop_x1 > width or crop_y0 < 0 or crop_y1 > height or crop_x0 > crop_x1 or crop_y0 > crop_y1:
            print('crop region out of bounds for {}, table nr {}'.format(doc, table_nr))
            return_values.append(None)
            continue
            



        im_table_crop = src_fullsize_image.crop((crop_x0, crop_y0, crop_x1, crop_y1))
        crop_info_name = dest_doc_id + '_cropinfo-{}.txt'.format(out_tag)
        dest_crop_info_path = os.path.join(dest_doc_dir, crop_info_name)

        table_annotations_subset = move_all_bbox_annotations_by_offset(table_annotations_subset, crop_x0, crop_y0)


        dest_image = os.path.join(dest_doc_id + '-0.png')
        dest_image_path = os.path.join(dest_doc_dir, dest_image)

        dest_ann_name = os.path.join(dest_doc_id + '-{}.json'.format(out_tag))
        dest_ann_path = os.path.join(dest_doc_dir, dest_ann_name)

        if not os.path.exists(dest_doc_dir):
            os.mkdir(dest_doc_dir)

        dest_meta_ann_name = os.path.join(dest_doc_id + '.json'.format(out_tag))
        dest_meta_ann_path = os.path.join(dest_doc_dir, dest_meta_ann_name)
        new_meta_contents = {'id': dest_doc_id, 'title':dest_doc_id, 'pages':1}
        with open(dest_meta_ann_path, 'w') as out_file:
            json.dump(new_meta_contents, out_file, indent=1, sort_keys=True)


        crop_info_text = "crop info for source file: {} and dest file: {}, page: {}, table id: {} (format: crop_x0; crop_y0; crop_x1, crop_y1) \n{}, {}, {}, {}".format(os.path.basename(src_annotations_file), dest_ann_path, page, table_ann['id'], crop_x0, crop_y0, crop_x1, crop_y1)
        with open(dest_crop_info_path, 'w') as out_file:
            out_file.write(crop_info_text)


        #copyfile(src_image_path, dest_image_path)
        try:
            im_table_crop.save(dest_image_path)
        except (SystemError, AttributeError) as e:
            print('orig shape: {}, img shape: {}, crops: {}'.format(src_fullsize_image.size, im_table_crop.size, [crop_x0, crop_x1, crop_y0, crop_y1]))
            print('could not save image for table nr {} of {}'.format(table_nr, doc))
            return_values.append(None)
            #continue
            raise
        #dest_fullsize_image.save(dest_image_path)
        with open(dest_ann_path, 'w') as out_file:
            json.dump(table_annotations_subset, out_file, indent=1, sort_keys=True)
        return_values.append(dest_doc_id)

    return return_values


def split_content_line_blocks_by_page_and_column(consecutive_content_line_block, ann_children_ids, ann_by_id, column_ranges):
    initial_nr_of_lines = len(consecutive_content_line_block)
    separate_blocks = []
    bbox_to_content_line_mapping = dict()
    all_bbox_tuples = []
    for content_line_ann_id in consecutive_content_line_block:
        content_line_bboxes = get_all_bbox_anns_for_current_id(content_line_ann_id, ann_children_ids, ann_by_id)
        if len(content_line_bboxes) > 1:
            print('there should only be one bbox per content line: {}'.format(ann_by_id[content_line_ann_id]))
            continue
        elif len(content_line_bboxes) == 0:
            print('no bbox found for content line')
            continue
        bbox_ann = content_line_bboxes[0]
        column_nr = column_number_of_bbox(bbox_ann['bbox'], column_ranges)
        ann_bbox_tuple = tuple(bbox_ann['bbox'] + [bbox_ann['page']] + [column_nr])
        all_bbox_tuples.append(ann_bbox_tuple)
        if ann_bbox_tuple in bbox_to_content_line_mapping:
            print('different content lines have the same bbox! only one of them will be added to a block')
            continue
        bbox_to_content_line_mapping[ann_bbox_tuple] = content_line_ann_id
            

   
    all_tuple_blocks  = []
    current_tuple_block = []
    for i,bbox_tuple in enumerate(all_bbox_tuples):
        is_last_item = False
        if i == (len(all_bbox_tuples) - 1):
            is_last_item = True
    
        if len(current_tuple_block) == 0:
            current_tuple_block.append(bbox_tuple)
        else:
            if current_tuple_block[-1][4] == bbox_tuple[4] and current_tuple_block[-1][5] == bbox_tuple[5] and abs(current_tuple_block[-1][1] - bbox_tuple[1]) < 150:
                current_tuple_block.append(bbox_tuple)
            else:
                all_tuple_blocks.append(current_tuple_block)                
                current_tuple_block = [bbox_tuple]


    if len(current_tuple_block) > 0:
        all_tuple_blocks.append(current_tuple_block)                
    for tuple_block in all_tuple_blocks:
        content_line_block = [bbox_to_content_line_mapping[ann_tuple] for ann_tuple in tuple_block]
        separate_blocks.append(content_line_block)
    return separate_blocks
        


def cluster_bibliography(input_tuple):
    (doc, in_tag, out_tag, src_dir, dest_dir) = input_tuple

    src_doc_folder = os.path.join(src_dir, doc)
    return_values = []

    src_annotations_file = os.path.join(src_dir, doc, doc + '-{}.json'.format(in_tag))
    with open(src_annotations_file) as f:
        annotation_list = json.load(f)




    annotations = annotation_list

    annotations = cluster_content_blocks_under_categories(annotations, root_types_to_consider = ['bibliography'], new_block_category='bib_block')

    dest_annotations_relpath = os.path.join(doc, doc + '-{}.json'.format(out_tag))
    dest_annotations_fullpath = os.path.join(dest_dir, dest_annotations_relpath)
    with open(dest_annotations_fullpath, 'w') as out_file:
        json.dump(annotations, out_file, indent=1, sort_keys=True)
         
        
    return doc
    


def cluster_content_lines_and_fix_headers(input_tuple):
    (doc, in_tag, out_tag, src_dir, dest_dir) = input_tuple

    src_doc_folder = os.path.join(src_dir, doc)
    return_values = []

    src_annotations_file = os.path.join(src_dir, doc, doc + '-{}.json'.format(in_tag))
    with open(src_annotations_file) as f:
        annotation_list = json.load(f)


    #pages_with_tables_figure_abstract_lists = defaultdict(set)


    annotations = annotation_list
    ann_by_id = dict()
    anns_by_cat = defaultdict(list)
    ann_children_ids = defaultdict(list)


    for ann in annotations:
        ann_by_id[ann['id']] =  ann 
        anns_by_cat[ann['category']].append(ann)
        ann_children_ids[ann['parent']].append(ann['id'])

    #remove small nested content lines
    small_content_lines_to_remove = set()
    #all_content_line_ids = set(ann['id'] for ann in anns_by_cat['content_line'])
    small_content_line_ids = dict()
    all_big_content_line_tuples = set()
    for content_line_ann in anns_by_cat['content_line']:
        content_line_bboxes = get_all_bbox_anns_for_current_id(content_line_ann['id'], ann_children_ids, ann_by_id)
        if len(content_line_bboxes) == 0:
            continue
    
        current_merged_content_line_bbox = get_merged_bbox_from_list_of_bbox_anns(content_line_bboxes)

        bbox_tuple = tuple(current_merged_content_line_bbox + [content_line_bboxes[0]['page']])
        if current_merged_content_line_bbox[2] < 40:
            small_content_line_ids[content_line_ann['id']] = bbox_tuple
        else:
            all_big_content_line_tuples.add(bbox_tuple)
            #all_big_content_line_ids.add(content_line_ann['id'])

    for small_content_line_id, small_bbox_tuple in small_content_line_ids.items():
        if any(big_bbox_tuple[4] == small_bbox_tuple[4] and second_bbox_contained_in_first_bbox(big_bbox_tuple[:4], small_bbox_tuple[:4], tolerance=5) for big_bbox_tuple in all_big_content_line_tuples):
            small_content_lines_to_remove.add(small_content_line_id)

    bad_headings_to_remove = set()
    for heading_ann in anns_by_cat['heading']:
        heading_bboxes = get_all_bbox_anns_for_current_id(heading_ann['id'], ann_children_ids, ann_by_id)
        if len(heading_bboxes) > 2:
            bad_headings_to_remove.add(heading_ann['id'])
        elif len(heading_bboxes) > 2:
            #too far away to truly belong to the same heading
            if abs(heading_bboxes[0][1] - heading_bboxes[1][0]) > 100:
                bad_headings_to_remove.add(heading_ann['id'])
        
            

    #remove and refresh

    annotations = [ann for ann in annotations if ann['id'] not in small_content_lines_to_remove and ann['id'] not in bad_headings_to_remove]
    #print('removed {} nested content lines'.format(len(small_content_lines_to_remove)))
    annotations = delete_structure_annotations_without_children(annotations)

    annotations = cluster_content_blocks_under_categories(annotations)

    dest_annotations_relpath = os.path.join(doc, doc + '-{}.json'.format(out_tag))
    dest_annotations_fullpath = os.path.join(dest_dir, dest_annotations_relpath)
    with open(dest_annotations_fullpath, 'w') as out_file:
        json.dump(annotations, out_file, indent=1, sort_keys=True)
         
        
    return doc


def cluster_content_blocks_under_categories(annotations, root_types_to_consider = ['document', 'section'], new_block_category='content_block'):

    ann_by_id = dict()
    anns_by_cat = defaultdict(list)
    ann_children_ids = defaultdict(list)

    for ann in annotations:
        ann_by_id[ann['id']] =  ann 
        anns_by_cat[ann['category']].append(ann)
        ann_children_ids[ann['parent']].append(ann['id'])

    def create_fn_get_new_ann_id(max_id):
        current_max_id = max_id
        def get_new_ann_id():
            nonlocal current_max_id
            current_max_id += 1
            return current_max_id
        return get_new_ann_id

    max_id = max(set(ann_by_id.keys()))
    get_new_ann_id = create_fn_get_new_ann_id(max_id)

    all_potential_column_anns = anns_by_cat['content_line'] 
    content_lines_anns = anns_by_cat['content_lines']
    if len(content_lines_anns) > 0:
        #print("found 'content_lines' type")
        all_potential_column_anns += content_lines_anns
    if len(all_potential_column_anns) == 0:
        #print('no content lines to work with..')
        return annotations
    column_info =  get_columns_and_their_widths(all_potential_column_anns, ann_children_ids, ann_by_id)
    if column_info is None:
        print('couldnt find columns!')
        return annotations
    else:
        is_single_column, most_common_widths, column_ranges, all_bbox_anns_by_page = column_info


    valid_line_types = {'content_line', 'content_lines'}
    regular_content_root_types =root_types_to_consider 
    #all_content_line_blocks = []
    content_blocks_at_insertion_points = dict()
    for root_type in regular_content_root_types:
        for root_ann in anns_by_cat[root_type]:
            consecutive_content_line_block = []
            root_ann_children_ids = ann_children_ids[root_ann['id']]
            #print('{} children for {}'.format(len(root_ann_children_ids), root_ann['category']))
            for i, child_ann_id in enumerate(root_ann_children_ids):
                is_last_item = False
                if i == (len(root_ann_children_ids) - 1):
                    is_last_item = True
                child_ann = ann_by_id[child_ann_id]
                if child_ann['category'] in valid_line_types :
                    consecutive_content_line_block.append(child_ann['id'])
                if child_ann['category'] not in  valid_line_types or is_last_item:
                    if len(consecutive_content_line_block) > 0:
                        split_content_line_blocks = split_content_line_blocks_by_page_and_column(consecutive_content_line_block, ann_children_ids, ann_by_id, column_ranges)

                        if len(split_content_line_blocks) > 0:
                            for content_line_block in split_content_line_blocks:
#                                if len(content_line_block) < 2:
#                                    print('less than two content line in block. not creating block..')
#                                    continue
                                insertion_point = content_line_block[0]
                                new_content_block_ann_id = get_new_ann_id()
                                new_block_ann = {'category':new_block_category, 'parent':root_ann['id'], 'id':new_content_block_ann_id}
                                for content_line_id in content_line_block:
                                    content_line_ann = ann_by_id[content_line_id]
                                    content_line_ann['parent'] = new_content_block_ann_id
                                content_blocks_at_insertion_points[insertion_point] = new_block_ann
                        #else:
                            #print('no more content blocks after splitting for {}'.format(doc))
                        consecutive_content_line_block = []
                                #all_content_line_blocks.append(split_content_line_blocks)
                elif child_ann['category'] == 'box':
                    print('content block has immediate "box" child')

    list_indeces_for_insertion_ids = dict()
    insertion_ids = set( content_blocks_at_insertion_points.keys())
    #print('{} insertion ids'.format(len(insertion_ids)))
    for list_index, ann in enumerate(annotations):
        if ann['id'] in insertion_ids:
            list_index_to_insert = list_index #insert _before_ original annotation placement
            list_indeces_for_insertion_ids[list_index_to_insert] = ann['id']

    for list_index_to_insert in sorted(list(list_indeces_for_insertion_ids.keys()), reverse=True):
        insertion_ann_id = list_indeces_for_insertion_ids[list_index_to_insert]
        content_block_to_be_inserted = content_blocks_at_insertion_points[insertion_ann_id]
        annotations[list_index_to_insert:list_index_to_insert] = [content_block_to_be_inserted]
    return annotations


def create_pdfminer_xml(input_tuple):
    (doc, in_tag, out_tag, src_dir, dest_dir) = input_tuple
    

    pdf_path_for_doc = os.path.join(src_dir, doc, doc + '.pdf')
    dest_xml_path_for_doc = pdf_path_for_doc.replace('.pdf', '_pdfminer.xml')
    try:
        create_xml_for_page(dest_xml_path_for_doc, pdf_path_for_doc)
        #print('pdfminer success for {}'.format(doc))
    except pdfminer.pdfparser.PDFSyntaxError as e:
        print('pdf syntaxerror in {}'.format(doc))
        if os.path.isfile(dest_xml_path_for_doc):
            os.remove(dest_xml_path_for_doc)
        return None
    except pdfminer.psparser.PSSyntaxError as e:
        print('ps syntaxerror in {}'.format(doc))
        if os.path.isfile(dest_xml_path_for_doc):
            os.remove(dest_xml_path_for_doc)
        return None
    except (TypeError, AttributeError, ValueError) as e:
        print('pdfminer error in {}'.format(doc))
        if os.path.isfile(dest_xml_path_for_doc):
            os.remove(dest_xml_path_for_doc)
        return None

    return doc
