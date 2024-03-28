from xmlrpc.client import MAXINT
import torch
import os
import glob
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')  # turn off gui
from segmentationsg.utils.visualizer import SGVisualizer
from detectron2.utils.visualizer import ColorMode#, Visualizer
from detectron2.utils.file_io import PathManager
from PIL import Image
from detectron2.data.detection_utils import _apply_exif_orientation
from detectron2.data.detection_utils import convert_PIL_to_numpy
from detectron2.structures.instances import Instances
from detectron2.structures.boxes import Boxes
import sys
import networkx as nx
import matplotlib.pyplot as plt
import copy

import argparse
import multiprocessing as mp
from multiprocessing import Pool
import time
from tqdm import *
from pathlib import Path

from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from segmentationsg.data import add_dataset_config, register_datasets
from segmentationsg.modeling.roi_heads.scenegraph_head import add_scenegraph_config
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

from detectron2.config import get_cfg

import xml.etree.ElementTree as ET
from operator import itemgetter

import warnings

"""
Example, for code usage:

python postprocessing.py --config-file ./data/checkpoints/03_213_sgg_end2end_EP_WSFT_unionfeat/config.yaml --input raw_tensors/*.pt --output output_folder --images images/* --confidence-threshold 0.5 --opts MODEL.WEIGHTS ./data/checkpoints/03_213_sgg_end2end_EP_WSFT_unionfeat/model_0199999.pth --visualize yes --hocr yes

This will create postprocessed tensor files and usual visualizations of the instances and relations if --visualize "any_string" is set,
if you don't need visualizations just don't specify the --visualize and --images argument at all

The --images argument is needed if visualization is wanted, the image names should be equivalent to the raw tensor names.


"""

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_dataset_config(cfg)
    add_scenegraph_config(cfg)
    assert(cfg.MODEL.ROI_SCENEGRAPH_HEAD.MODE in ['predcls', 'sgls', 'sgdet']) , "Mode {} not supported".format(cfg.MODEL.ROI_SCENEGRaGraph.MODE)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold

    #default_setup(cfg, args)
    cfg.freeze()
    register_datasets(cfg)
    #default_setup(cfg, args)
    #setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="LSDA")

    #print(cfg)
    return cfg

# if isolates exist they will be added to either article or meta, article and meta kids list determines to which they get added
def create_article_and_meta_kids_list(class_mapping_list):
    articlekids_list = []
    metakids_list = []
    
    for c in class_mapping_list:
        if c == 'pagenr' or c == 'foot' or c=='footnote' or c=='head' or c=='subject' or c =='date':
            metakids_list.append(c)
        if c == 'author' or c=='backgroundfigure' or c=='col' or c=='contentblock' or c=='figure' or c=='figurecaption' or c=='figuregraphic' or c=='header' or c=='item' or c=='itemize' or c=='orderedgroup' or c=='row' or c=='table' or c=='tabular' or c=='unorderedgroup':
            articlekids_list.append(c) 
    #includes arxiv and eperiodica classes
    #for c in class_mapping_list:
    #    if c=='pagenr' or c =='foot' or c=='footnote' or c=='head' or c=='subject' or c=='date':
    #        metakids_list.append(c)
    #for c in class_mapping_list:
    #    if c !='pagenr' or c != 'foot' or c!= 'footnote' or c!='head' or c !='subject' or c!='date' or c != 'documentroot' or c!='meta' or c!='article' or c!='tableofcontent':
    #        articlekids_list.append(c)
            
    return articlekids_list, metakids_list

#checks if raw_tensor has a documentroot
def has_root_article_meta_toc(raw_tensor, class_mapping_list):
    has_root = False
    has_article = False
    has_meta = False
    has_toc = False

    #index corresponding to docroot, artile and meta
    root_index = class_mapping_list.index("documentroot")
    article_index = class_mapping_list.index("article")
    meta_index = class_mapping_list.index("meta")
    toc_index = class_mapping_list.index("tableofcontent")
    
    count_root = 0
    for class_index in (raw_tensor["instances"].pred_classes).tolist():
        if class_index == root_index:
            has_root = True
            count_root += 1
        if class_index == article_index:
            has_article = True
        if class_index == meta_index:
            has_meta = True
        if class_index == toc_index:
            has_toc = True
    
    if count_root > 1:
        warnings.warn("More than 1 root was detected, this may lead to undefined behavior", category=UserWarning)
    return has_root, has_article, has_meta, has_toc

def create_root(raw_tensor, class_mapping_list):

    root_index = class_mapping_list.index("documentroot")

    # adding the instance for documentroot
    new_instance = Instances(raw_tensor["instances"].image_size)
    documentrootbox = torch.tensor([0,0,raw_tensor["instances"].image_size[1],raw_tensor["instances"].image_size[0]]).unsqueeze(0)
    new_instance.pred_boxes = Boxes(documentrootbox)
    
    new_instance.pred_classes = torch.tensor([root_index])
    tensor_index = len(raw_tensor["instances"].pred_classes)
    new_instance.scores = torch.tensor([1])
    
    pred_class_prob_list = []
    for i in range(0, len(class_mapping_list)+1):
        if i == root_index:
            pred_class_prob_list.append(1.0)
        else:
            pred_class_prob_list.append(0.0)
    new_instance.pred_class_prob = torch.tensor(pred_class_prob_list).unsqueeze(0)
    
    # now concatenate the old instances with the new instances
    
    used_device = raw_tensor["instances"].pred_classes.get_device()
    
    instanceslist = [raw_tensor["instances"], new_instance.to(used_device)]
    raw_tensor["instances"] = Instances.cat(instanceslist)

    return raw_tensor

def create_article(raw_tensor, class_mapping_list):
    article_index = class_mapping_list.index("article")

    # adding the instance for documentroot
    new_instance = Instances(raw_tensor["instances"].image_size)
    articlebox = torch.tensor([0,0,raw_tensor["instances"].image_size[1],raw_tensor["instances"].image_size[0]]).unsqueeze(0)
    new_instance.pred_boxes = Boxes(articlebox)
    
    new_instance.pred_classes = torch.tensor([article_index])
    tensor_index = len(raw_tensor["instances"].pred_classes)
    new_instance.scores = torch.tensor([1])
    
    pred_class_prob_list = []
    for i in range(0, len(class_mapping_list)+1):
        if i == article_index:
            pred_class_prob_list.append(1.0)
        else:
            pred_class_prob_list.append(0.0)
    new_instance.pred_class_prob = torch.tensor(pred_class_prob_list).unsqueeze(0)
    
    # now concatenate the old instances with the new instances
    
    used_device = raw_tensor["instances"].pred_classes.get_device()
    
    instanceslist = [raw_tensor["instances"], new_instance.to(used_device)]
    raw_tensor["instances"] = Instances.cat(instanceslist)

    return raw_tensor

def create_meta(raw_tensor, class_mapping_list):
    meta_index = class_mapping_list.index("meta")

    # adding the instance for documentroot
    new_instance = Instances(raw_tensor["instances"].image_size)
    metabox = torch.tensor([0,0,raw_tensor["instances"].image_size[1],raw_tensor["instances"].image_size[0]]).unsqueeze(0)
    new_instance.pred_boxes = Boxes(metabox)
    
    new_instance.pred_classes = torch.tensor([meta_index])
    tensor_index = len(raw_tensor["instances"].pred_classes)
    new_instance.scores = torch.tensor([1])
    
    pred_class_prob_list = []
    for i in range(0, len(class_mapping_list)+1):
        if i == meta_index:
            pred_class_prob_list.append(1.0)
        else:
            pred_class_prob_list.append(0.0)
    new_instance.pred_class_prob = torch.tensor(pred_class_prob_list).unsqueeze(0)
    
    # now concatenate the old instances with the new instances
    
    used_device = raw_tensor["instances"].pred_classes.get_device()
    
    instanceslist = [raw_tensor["instances"], new_instance.to(used_device)]
    raw_tensor["instances"] = Instances.cat(instanceslist)

    return raw_tensor


def create_parentof_and_followedby_matrices(raw_tensor):
    num_instances = len(raw_tensor["instances"])
    parentof_matrix = np.zeros([num_instances,num_instances])
    followedby_matrix = np.zeros([num_instances,num_instances])
    
    parentof_matrix_full = np.zeros([num_instances,num_instances])
    followedby_matrix_full = np.zeros([num_instances,num_instances])
    
    #create full matrices
    for i, pair in enumerate(raw_tensor["rel_pair_idxs"]):
        rel_followedby = raw_tensor["pred_rel_scores"][i][0]
        rel_parentof = raw_tensor["pred_rel_scores"][i][1]
        followedby_matrix_full[pair[0], pair[1]] = rel_followedby
        parentof_matrix_full[pair[0], pair[1]] = rel_parentof
    
    #create normal matrices
    for i, pair in enumerate(raw_tensor["rel_pair_idxs"]):
        rel = torch.argmax(raw_tensor["pred_rel_scores"][i])
        if rel == 0: #followedby
            followedby_matrix[pair[0]][pair[1]] = raw_tensor["pred_rel_scores"][i][rel]
        elif rel == 1: #parentof
            parentof_matrix[pair[0]][pair[1]]= raw_tensor["pred_rel_scores"][i][rel]
    return parentof_matrix, followedby_matrix, parentof_matrix_full, followedby_matrix_full 

# fix article, meta, tableofcontent
def fix_amt(parentof_matrix, followedby_matrix, parentof_matrix_full, followedby_matrix_full, class_mapping_list, raw_tensor):
    root_index, article_index, meta_index, toc_index = class_mapping_list.index("documentroot"), class_mapping_list.index("article"), class_mapping_list.index("meta"), class_mapping_list.index("tableofcontent")
    tensor_root_index = ((raw_tensor["instances"].pred_classes).tolist()).index(root_index)
    tensor_meta_index = ((raw_tensor["instances"].pred_classes).tolist()).index(meta_index)
    # toc might not always be 
    test = lambda l, e: l.index(e) if e in l else None
    tensor_toc_index = test(((raw_tensor["instances"].pred_classes).tolist()), toc_index)
    tensor_article_index = test(((raw_tensor["instances"].pred_classes).tolist()), article_index)

    # make sure that documentroot is the only parent of meta, article and toc
    if parentof_matrix[tensor_root_index, tensor_meta_index] == 0.0:
        parentof_matrix[tensor_root_index, tensor_meta_index] = 1.0
    if tensor_article_index:
        if parentof_matrix[tensor_root_index, tensor_article_index] == 0.0:
            parentof_matrix[tensor_root_index, tensor_article_index] = 1.0
    if tensor_toc_index:
        if parentof_matrix[tensor_root_index, tensor_toc_index] == 0.0:
            parentof_matrix[tensor_root_index, tensor_toc_index] = 1.0
    
    
    num_instances = followedby_matrix[0].size
    for i in range(num_instances):
        if parentof_matrix[i, tensor_meta_index] != 0.0:
            if i != tensor_root_index:
                parentof_matrix[i, tensor_meta_index] = 0.0
                parentof_matrix_full[i, tensor_meta_index] = 0.0
        if tensor_article_index:
            if parentof_matrix[i, tensor_article_index] != 0.0:
                if i != tensor_root_index:
                    parentof_matrix[i, tensor_article_index] = 0.0
                    parentof_matrix_full[i, tensor_article_index] = 0.0
        if tensor_toc_index:
            if parentof_matrix[i, tensor_toc_index] != 0.0:
                if i != tensor_root_index:
                    parentof_matrix[i, tensor_toc_index] = 0.0
                    parentof_matrix_full[i, tensor_toc_index] = 0.0
    
    return parentof_matrix, parentof_matrix_full, followedby_matrix, followedby_matrix_full

def force_followedby(parentof_matrix, parentof_matrix_full, followedby_matrix, followedby_matrix_full):
    num_instances = followedby_matrix[0].size
    # no two followedby end in the same node:
    for i in range(num_instances):
        maxidx = np.argmax(followedby_matrix[:,i]) #index of max value
        maxvalue = followedby_matrix[maxidx,i]
        followedby_matrix[:,i] = np.zeros([num_instances])
        followedby_matrix[maxidx,i] = maxvalue
    
    # no two followedby start in the same node
    for i in range(num_instances):
        maxidx = np.argmax(followedby_matrix[i,:]) #index of max value
        maxvalue = followedby_matrix[i,maxidx]
        followedby_matrix[i,:] = np.zeros([num_instances])
        followedby_matrix[i,maxidx] = maxvalue
    
    return parentof_matrix, parentof_matrix_full, followedby_matrix, followedby_matrix_full
    
        
def force_antisymmetry(parentof_matrix, parentof_matrix_full, followedby_matrix, followedby_matrix_full):
    num_instances = followedby_matrix[0].size
    
    # followedby is antisymmetric
    for i in range(num_instances):
        for j in range(num_instances):
            if (followedby_matrix[i][j]!= 0.0 and followedby_matrix[j][i] != 0.0):
                if(followedby_matrix[i][j] > followedby_matrix[j][i]):
                    followedby_matrix[j][i] = 0.0
                    followedby_matrix_full[j][i] = 0.0
                else:
                    followedby_matrix[i][j] = 0.0
                    followedby_matrix_full[i][j] = 0.0
            if (parentof_matrix[i][j]!= 0.0 and parentof_matrix[j][i] != 0.0):
                if(parentof_matrix[i][j] > parentof_matrix[j][i]):
                    parentof_matrix[j][i] = 0.0
                    parentof_matrix_full[j][i] = 0.0
                else:
                    parentof_matrix[i][j] = 0.0
                    parentof_matrix_full[i][j] = 0.0
            if (parentof_matrix[i][j] != 0.0 and followedby_matrix[j][i] != 0.0):
                if(parentof_matrix[i][j] > followedby_matrix[j][i]):
                    followedby_matrix[j][i] = 0.0
                    followedby_matrix_full[j][i] = 0.0
                else:
                    parentof_matrix[i][j] = 0.0
                    parentof_matrix_full[i][j] = 0.0

    return parentof_matrix, parentof_matrix_full, followedby_matrix, followedby_matrix_full

def check_for_cycles(parentof_matrix, followedby_matrix, parentof_matrix_full, followedby_matrix_full):
    #create adjacency matrix that combines followedby and parentof relations
    num_instances = parentof_matrix[0].size
    sg_combined = np.zeros([num_instances,num_instances])
    for i in range(num_instances):
        for j in range(num_instances):
            if (parentof_matrix[i][j] != 0.0 or followedby_matrix[i][j]!= 0.0):
                sg_combined[i][j] =1
    #creates graph with weights
    pred_graph = nx.OrderedDiGraph()
    for i in range(num_instances):
        pred_graph.add_node(i)
        for j in range(num_instances):
            pred_graph.add_node(j)
            if parentof_matrix[i][j] != 0.0:
                pred_graph.add_edge(i,j, label="parentof", weight=parentof_matrix[i][j])
            if followedby_matrix[i][j] != 0.0:
                pred_graph.add_edge(i,j, label="followedby", weight=followedby_matrix[i][j])
                
    cycles = nx.simple_cycles(pred_graph)
    cycles_list = list(cycles)
    
    
    return cycles_list != []


def force_no_cycles(parentof_matrix, followedby_matrix, parentof_matrix_full, followedby_matrix_full):
    #create adjacency matrix that combines followedby and parentof relations
    num_instances = parentof_matrix[0].size
    sg_combined = np.zeros([num_instances,num_instances])
    for i in range(num_instances):
        for j in range(num_instances):
            if (parentof_matrix[i][j] != 0.0 or followedby_matrix[i][j]!= 0.0):
                sg_combined[i][j] =1
    #creates graph with weights
    pred_graph = nx.OrderedDiGraph()
    for i in range(num_instances):
        pred_graph.add_node(i)
        for j in range(num_instances):
            pred_graph.add_node(j)
            if parentof_matrix[i][j] != 0.0:
                pred_graph.add_edge(i,j, label="parentof", weight=parentof_matrix[i][j])
            if followedby_matrix[i][j] != 0.0:
                pred_graph.add_edge(i,j, label="followedby", weight=followedby_matrix[i][j])
                
    cycles = nx.simple_cycles(pred_graph)
    cycles_list = list(cycles)
    MAX_ITERATIONS = 1000
    counter = 0
    
    #remove all cycles
    while(cycles_list!=[] and counter < MAX_ITERATIONS):
        counter += 1
        for cycle in cycles_list:
            min_cycle_score = MAXINT
            min_cycle_edge = [None, None]
            min_cycle_rel = None
            for v in range(len(cycle)):
                # an edge is v and v+1
                v1 = cycle[v]
                v2 = cycle[(v+1)%(len(cycle))]
                if(pred_graph[v1][v2]["weight"] < min_cycle_score):
                    min_cycle_score = pred_graph[v1][v2]["weight"]
                    min_cycle_pair = [v1,v2]
                    min_cycle_rel = pred_graph[v1][v2]["label"]
            
            assert(min_cycle_score!=MAXINT) #If this happens we must have overflowed or something. do not keep working.
            if(min_cycle_rel == "parentof"):
                parentof_matrix[min_cycle_pair[0]][min_cycle_pair[1]] = 0
                parentof_matrix_full[min_cycle_pair[0]][min_cycle_pair[1]] = 0
            if(min_cycle_rel == "followedby"):
                followedby_matrix[min_cycle_pair[0]][min_cycle_pair[1]] = 0
                followedby_matrix_full[min_cycle_pair[0]][min_cycle_pair[1]] = 0
            sg_combined[min_cycle_pair[0],min_cycle_pair[1]] = 0
            pred_graph.remove_edge(min_cycle_pair[0],min_cycle_pair[1])
            break 
        cycles = nx.simple_cycles(pred_graph)
        cycles_list = list(cycles)
    return parentof_matrix, followedby_matrix, parentof_matrix_full, followedby_matrix_full

def force_unorderedgroup(followedby_matrix, followedby_matrix_full, class_mapping_list, raw_tensor):
    num_instances = followedby_matrix[0].size
    unorderedgroup_index = class_mapping_list.index("unorderedgroup")
    # find all tensor_indices of unorderedgroups
    unorderedgroup_list = []
    for i, class_index in enumerate((raw_tensor["instances"].pred_classes).tolist()):
        if class_index == unorderedgroup_index:
            unorderedgroup_list.append(i)
    for j in unorderedgroup_list:
        followedby_matrix[:,j] = np.zeros([num_instances])
        followedby_matrix[j,:] = np.zeros([num_instances])
        followedby_matrix_full[:,j] = np.zeros([num_instances])
        followedby_matrix_full[j,:] = np.zeros([num_instances])
    
    return followedby_matrix, followedby_matrix_full





def force_parentof(parentof_matrix, parentof_matrix_full, followedby_matrix, followedby_matrix_full, class_mapping_list, raw_tensor, has_toc):
    num_instances = parentof_matrix.shape[0]  # Adjusted to use shape for consistency
    no_parent_nodes = []  # List to keep track of nodes without manageable parents

    # Create combined adjacency matrix
    sg_combined = np.zeros([num_instances, num_instances])
    for i in range(num_instances):
        for j in range(num_instances):
            if parentof_matrix[i][j] != 0.0 or followedby_matrix[i][j] != 0.0:
                sg_combined[i][j] = 1

    # Function to check if adding an edge creates a cycle in the combined graph
    def creates_cycle(combined_matrix, child, parent):
        if child == parent:
            return True
        visited = set()
        def dfs(v):
            if v == parent:
                return True
            visited.add(v)
            for w in range(num_instances):
                if combined_matrix[v, w] == 1 and w not in visited and dfs(w):
                    return True
            return False
        return dfs(child)
    
    

    
    for j in range(num_instances):
        if (any(parentof_matrix[:,j])):
            maxidx = np.argmax(parentof_matrix[:,j]) #index of max value
            maxvalue = parentof_matrix[maxidx,j]
            
            # if most likely class is docroot take second most likely parent
            
            classx = class_mapping_list[int(raw_tensor['instances'].pred_classes[j])]
            class_maxidx = class_mapping_list[int(raw_tensor['instances'].pred_classes[maxidx])]
            if class_maxidx == "documentroot":
                if not (classx == "meta" or classx == "article" or classx == "tableofcontent"):
                    parentof_matrix[maxidx, j] = 0.0
                    parentof_matrix_full[maxidx, j] = 0.0

                    maxidx = np.argmax(parentof_matrix[:,j]) #index of max value
                    maxvalue = parentof_matrix[maxidx,j]


                

            parentof_matrix[:,j] = np.zeros([num_instances])
            parentof_matrix[maxidx,j] = maxvalue
        else:
            classx = class_mapping_list[int(raw_tensor['instances'].pred_classes[j])]
            if classx == "documentroot":
                continue
            if classx == "table" and has_toc:
                tensor_toc_index = ((raw_tensor["instances"].pred_classes).tolist()).index(class_mapping_list.index("tableofcontent"))
                parentof_matrix[tensor_toc_index, j] = 1.0
                parentof_matrix_full[tensor_toc_index, j] = 1.0
                continue

            potential_parents = np.copy(parentof_matrix_full[:, j])
            tries = 0
            failed = False
            while np.any(potential_parents):
                maxidx = np.argmax(potential_parents)
                class_maxidx = class_mapping_list[int(raw_tensor['instances'].pred_classes[maxidx])]
                maxvalue = potential_parents[maxidx]

                if class_maxidx == "documentroot":
                    if not (classx == "meta" or classx == "article" or classx == "tableofcontent"):
                        parentof_matrix[maxidx, j] = 0.0
                        parentof_matrix_full[maxidx, j] = 0.0

                        maxidx = np.argmax(parentof_matrix[:,j]) #index of max value
                        maxvalue = parentof_matrix[maxidx,j]

                # Temporarily add edge to combined matrix to check for cycles
                sg_combined_temp = np.copy(sg_combined)
                sg_combined_temp[maxidx, j] = 1

                if not creates_cycle(sg_combined_temp, j, maxidx):
                    parentof_matrix[:, j] = 0
                    parentof_matrix[maxidx, j] = maxvalue
                    sg_combined[maxidx, j] = 1  # Update combined matrix with valid parent
                    break
                else:
                    potential_parents[maxidx] = 0  # Invalidate this potential parent
                    parentof_matrix_full[maxidx, j] = 0
                tries += 1
                if tries < 3:
                    failed = True
                    break
            if failed:
                no_parent_nodes.append(j)  # No valid parent found
    
    # Ensure a valid tree structure with fallback heuristics
    article_kids_list, meta_kids_list = create_article_and_meta_kids_list(class_mapping_list)
    has_toc_and_no_article = False

    # Fallback heuristic to ensure documentroot has the necessary children
    for j in range(num_instances):
        classx = class_mapping_list[int(raw_tensor['instances'].pred_classes[j])]
        if not any(parentof_matrix[:, j]):
            if classx in meta_kids_list:
                try:
                    tensor_meta_index = ((raw_tensor["instances"].pred_classes).tolist()).index(class_mapping_list.index("meta"))
                except ValueError:
                    continue  # Handle case where 'meta' does not exist
                parentof_matrix[tensor_meta_index, j] = parentof_matrix_full[tensor_meta_index, j] = 1.0
            elif classx in article_kids_list:
                try:
                    tensor_article_index = ((raw_tensor["instances"].pred_classes).tolist()).index(class_mapping_list.index("article"))
                except ValueError:
                    try:
                        tensor_article_index = ((raw_tensor["instances"].pred_classes).tolist()).index(class_mapping_list.index("tableofcontent"))
                        has_toc_and_no_article = True
                    except ValueError:
                        continue  # Handle case where neither 'article' nor 'tableofcontent' exists
                parentof_matrix[tensor_article_index, j] = parentof_matrix_full[tensor_article_index, j] = 1.0
                
    

    return parentof_matrix, parentof_matrix_full


def create_postprocessed_tensor_from_matrices(parentof_matrix, followedby_matrix, raw_tensor):
    #### back to tensors ##########
    rel_pair_idxs_list = []
    pred_rel_scores_list = []
    for i in range(len(followedby_matrix)):
        for j in range(len(followedby_matrix)):
            if followedby_matrix[i][j] != 0.0:
                rel_pair_idxs_list.append([i,j])
                pred_rel_scores_list.append([followedby_matrix[i][j],0,0])
    for i in range(len(parentof_matrix)):
        for j in range(len(parentof_matrix)):
            if parentof_matrix[i][j] != 0.0:
                rel_pair_idxs_list.append([i,j])
                pred_rel_scores_list.append([0,parentof_matrix[i][j],0])
    
    assert(len(rel_pair_idxs_list) == len(pred_rel_scores_list))
    raw_tensor["pred_rel_scores"] = torch.tensor(pred_rel_scores_list)
    raw_tensor["rel_pair_idxs"] = torch.tensor(rel_pair_idxs_list)
    
    return raw_tensor
    
## given tensor we'll create a valid tree structured tensor
def postprocess_raw_tensor(raw_tensor, class_mapping_list):
    raw_tensor_before_postprocessing = raw_tensor.copy()
    # we first make sure that there is a documentroot, article and meta
    # no relations added this just adds the instance with scores to the raw_tensor
    
    has_root, has_article, has_meta, has_toc = has_root_article_meta_toc(raw_tensor, class_mapping_list)
    if not has_root:
        raw_tensor = create_root(raw_tensor, class_mapping_list)
    # only create new article when there's no article or tableofcontent present
    if not has_article and not has_toc:
        raw_tensor = create_article(raw_tensor, class_mapping_list)
        warnings.warn("no article or tableofcontent instance found, this may lead to undefined behavior in postprocessing and hocr file creation", category=UserWarning)
    if not has_meta:
        raw_tensor = create_meta(raw_tensor, class_mapping_list)
    
    # create 4 matrices for parentof and followedby for easier checking and handling
    parentof_matrix, followedby_matrix, parentof_matrix_full, followedby_matrix_full= create_parentof_and_followedby_matrices(raw_tensor)
    #print(f"parent of matrix: \n {parentof_matrix}\nfollowedby matrix: \n{followedby_matrix}\nparentof matrix full: \n{parentof_matrix_full}\nfollowedby matrix full:\n{followedby_matrix_full}")

    # we do not want documentroot to have a parent, nor a followedby relation:
    num_instances = len(raw_tensor["instances"])
    root_index = class_mapping_list.index("documentroot")
    tensor_root_index = ((raw_tensor["instances"].pred_classes).tolist()).index(root_index)
    for i in range(num_instances):
        parentof_matrix[i,tensor_root_index] = 0.0
        followedby_matrix[i,tensor_root_index] = 0.0
        followedby_matrix[tensor_root_index,i] = 0.0

        parentof_matrix_full[i,tensor_root_index] = 0.0
        followedby_matrix_full[i,tensor_root_index] = 0.0
        followedby_matrix_full[tensor_root_index,i] = 0.0
    
    # we make sure that article, meta and tableofcontent are the only possible children of documentroot
    parentof_matrix, parentof_matrix_full, followedby_matrix, followedby_matrix_full = fix_amt(parentof_matrix, followedby_matrix, parentof_matrix_full, followedby_matrix_full, class_mapping_list, raw_tensor)

    parentof_matrix, parentof_matrix_full, followedby_matrix, followedby_matrix_full = force_followedby(parentof_matrix, parentof_matrix_full, followedby_matrix, followedby_matrix_full)

    # we force every relation to be anti-symmetric, this will remove relations at best
    parentof_matrix, parentof_matrix_full, followedby_matrix, followedby_matrix_full = force_antisymmetry(parentof_matrix, parentof_matrix_full, followedby_matrix, followedby_matrix_full)

    #removes cycles in combined graph, again will at best just remove relations
    parentof_matrix, followedby_matrix, parentof_matrix_full, followedby_matrix_full = force_no_cycles(parentof_matrix, followedby_matrix, parentof_matrix_full, followedby_matrix_full)

    # unordered groups shouldn't be involved with any followedby relations
    followedby_matrix, followedby_matrix_full = force_unorderedgroup(followedby_matrix, followedby_matrix_full, class_mapping_list, raw_tensor)

    # finally if a node has more than 1 parent only leave the most likely parent, and complete the graph so that each node has a parent and only one parent
    parentof_matrix, parentof_matrix_full = force_parentof(parentof_matrix, parentof_matrix_full, followedby_matrix, followedby_matrix_full, class_mapping_list, raw_tensor, has_toc)
    
    # creates the new tensor from the two matrices
    postprocessed_tensor = create_postprocessed_tensor_from_matrices(parentof_matrix, followedby_matrix, raw_tensor)

    return raw_tensor_before_postprocessing, postprocessed_tensor


def postprocess_prediction_instances(predictions, orig_class_mapping_list):
    class_mapping_list = orig_class_mapping_list.copy()
    raw_tensor_dict = dict()
    raw_tensor_dict['instances'] = predictions
    pred_rel_labels = predictions._pred_rel_scores
    #num_rel_class_without_background = 2 #self.num_rel_categories_without_bg
    #background_removal_mask = (pred_rel_labels != num_rel_class_without_background)
    filtered_rel_pair_idx = predictions._rel_pair_idxs
    filtered_rel_class_prob = predictions._pred_rel_scores
    filtered_rel_labels = pred_rel_labels
    raw_tensor_dict['rel_pair_idxs'] = filtered_rel_pair_idx
    raw_tensor_dict['pred_rel_scores'] = filtered_rel_class_prob
    raw_tensor_dict['pred_rel_scores'] = filtered_rel_labels
    
    if 'documentroot' not in class_mapping_list:
        is_arxivdocs_mode = True
    else:
        is_arxivdocs_mode = False
    if is_arxivdocs_mode is True:
        
        num_classes_before = len(class_mapping_list) + 1 # +1 for background
        class_mapping_list[class_mapping_list.index('document')]='article'
        class_mapping_list.append("documentroot")
        class_mapping_list.append("tableofcontent")
        num_classes_after = len(class_mapping_list) + 1 # +1 for background
       
        num_instances = len(predictions)
    
        used_device = raw_tensor_dict['instances'].pred_class_prob.get_device()
       
        padded = torch.zeros(num_instances, num_classes_after)
        padded[:, :num_classes_before] = raw_tensor_dict['instances'].pred_class_prob
        raw_tensor_dict['instances'].pred_class_prob = padded.to(used_device) 
        
    
    _dictensor_before_postprocessing, postprocessed_tensor =  postprocess_raw_tensor(raw_tensor_dict, class_mapping_list)
    
    results_after = postprocessed_tensor['instances']
    results_after._rel_pair_idxs = postprocessed_tensor['rel_pair_idxs']
    results_after._pred_rel_scores = postprocessed_tensor['pred_rel_scores']
    return results_after

### methods for visualization ####

def load_image_and_metadata(image_path, cfg):
    image = read_image(image_path, format="BGR")
    image = image[:, :, ::-1]
    
    image_name = image_path.split("/")[-1]
    image_format = image_name.split(".")[-1]
    
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused")
    
    return image, metadata, image_name, image_format

def visualize_instances(tensor, image_path, cfg, output_folder, filename):
    image, metadata, image_name, image_format = load_image_and_metadata(image_path, cfg)
    
    visualizer = SGVisualizer(image, metadata, instance_mode=ColorMode.IMAGE)
    
    if "instances" in tensor:
        instances = tensor["instances"].to("cpu")
        vis_output_instances = visualizer.draw_instance_predictions(predictions=instances)
        vis_output_instances.save(output_folder+'/'+filename)
    plt.clf()
    plt.cla()
    plt.close()
    
    
def visualize_relations(tensor, class_mapping_list, output_folder, filename):
    pred_graph = nx.DiGraph()
    node_labels = []
    edge_labels = []
    
    #for each instance create a node and give it the label of it's class
    for i, box in enumerate(tensor["instances"].pred_boxes):
        pred_graph.add_node(i, label=class_mapping_list[tensor["instances"].pred_classes[i]])
        node_labels.append(class_mapping_list[tensor["instances"].pred_classes[i]])
    #create edges    
    for i, rel in enumerate(tensor["rel_pair_idxs"]):
        x = torch.argmax(tensor["pred_rel_scores"][i])
        if x == 0:
            pred_graph.add_edge(int(rel[0]), int(rel[1]), label= 'followedby')
            edge_labels.append('followedby')
        else:
            pred_graph.add_edge(int(rel[0]), int(rel[1]), label= 'parentof')
            edge_labels.append('parentof')
    
    pos=nx.nx_agraph.graphviz_layout(pred_graph, prog="dot")
    #draw the graph
    nx.draw(pred_graph, pos, arrows=True)
    
    #draw the node labels
    node_labels = nx.get_node_attributes(pred_graph, 'label')
    for i in node_labels:
        node_labels[i] = "#"+str(i)+" - "+node_labels[i]
    nx.draw_networkx_labels(pred_graph, pos, node_labels, font_size=9)
    
    #draw the edge labels
    edge_labels = nx.get_edge_attributes(pred_graph, 'label')
    nx.draw_networkx_edge_labels(pred_graph, pos, edge_labels, font_size=7)
    
    #save image and clear
    plt.savefig(output_folder + filename, format="PNG")
    plt.clf()
    plt.cla()
    plt.close()
    

    
    
#doesn't need filenames created them from image_path
def visualize_instances_and_relations(tensor, image_path, cfg, output_folder, tensor_before=None):
    image, metadata, image_name, image_format = load_image_and_metadata(image_path, cfg)
    class_mapping_list = MetadataCatalog.get(cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused").thing_classes
    
    filename_instances = image_name.rsplit(".", 1)[0] + "_instances.png"
    filename_relations = image_name.rsplit(".", 1)[0] + "_relations.png"
    filename_relations_before = image_name.rsplit(".", 1)[0] + "_relations_before.png"
    
    visualize_instances(tensor, image_path, cfg, output_folder, filename_instances)
    
    visualize_relations(tensor, class_mapping_list, output_folder, filename_relations)
    
    if tensor_before:
        visualize_relations(tensor_before, class_mapping_list, output_folder, filename_relations_before)
        
    print(image_name+" visualized")

def get_parser():
    parser = argparse.ArgumentParser(description="Post processing for raw tensors")
    parser.add_argument(
        "--config-file",
        default='/mnt/ds3lab-scratch/gusevm/sgg_segm_private_copy/sgg_segm_private/data/checkpoints/03_213_sgg_end2end_EP_WSFT_unionfeat/config.yaml',
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output tensors and visualizations",

    )
    parser.add_argument(
        "--input_raw_tensors_dir",
        type=str,
        help="A directory with images that corresponds to the raw tensors",
    )

    parser.add_argument(
        "--images_dir",
        type=str,
        help="A directory with images that corresponds to the raw tensors",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=["MODEL.WEIGHTS",'/mnt/ds3lab-scratch/gusevm/sgg_segm_private_copy/sgg_segm_private/data/checkpoints/03_213_sgg_end2end_EP_WSFT_unionfeat/model_0199999.pth'],
        nargs="+",
    )
    
    
    parser.add_argument(
        "--visualize_without_before",
        help="Puts .png files of instances visualized and ralationships between instances"
    )
    parser.add_argument(
        "--visualize_with_before",
        help="Additionally puts the relationships between instances before post processing into output folder"
    )
    
    return parser
    
def _process_document(input_tuple):
    raw_tensor_path, output_folder, class_mapping_list, cfg, image_path = input_tuple[0],input_tuple[1],input_tuple[2],input_tuple[3],input_tuple[4]
    raw_tensor = torch.load(raw_tensor_path)
    tensor_before, tensor_after = postprocess_raw_tensor(raw_tensor, class_mapping_list)
    
    filename = raw_tensor_path.split("/")[-1]
    torch.save(tensor_after, output_folder+"/"+filename)
     
    
    return tensor_before, tensor_after, image_path

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    class_mapping_list = MetadataCatalog.get(cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused").thing_classes
    
    #raw_tensor_paths = args.input
    raw_tensor_paths = glob.glob(args.input_raw_tensors_dir+"/*.pt")
    raw_tensor_paths.sort()
   
    image_paths = glob.glob(args.images_dir+"/*.png")
    #image_paths = args.images
    if image_paths:
        image_paths.sort()
        
    visualization_parameter_list = []

    all_worker_inputs = [(raw_tensor_path, args.output, class_mapping_list, cfg, image_path) for raw_tensor_path, image_path in zip(raw_tensor_paths, image_paths)]
    
    
    n_processes=5
    print("n_processes for pool: " + str(n_processes))
    with Pool(processes=n_processes) as p:
        max_ = len(raw_tensor_paths)
        with tqdm(total=max_) as pbar:
            for tensor_before, tensor_after, image_path in p.imap_unordered(_process_document, all_worker_inputs):
                pbar.update()
                visualization_parameter_list.append([tensor_before, tensor_after, image_path])
    
    if args.visualize_with_before:
        assert(image_paths)
        
        for triplet in visualization_parameter_list:
            tensor_before = triplet[0]
            tensor_postprocessed = triplet[1]
            image_path = triplet[2]
            visualize_instances_and_relations(tensor_postprocessed, image_path, cfg, args.output, tensor_before)
        
        
        
    if args.visualize_without_before:
        assert(image_paths)
        
        for triplet in visualization_parameter_list:
            tensor = triplet[1]
            image_path = triplet[2]
            visualize_instances_and_relations(tensor, image_path, cfg, args.output, tensor_before=None)