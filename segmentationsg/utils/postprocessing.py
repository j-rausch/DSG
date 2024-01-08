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
    
    for class_index in (raw_tensor["instances"].pred_classes).tolist():
        if class_index == root_index:
            has_root = True
        if class_index == article_index:
            has_article = True
        if class_index == meta_index:
            has_meta = True
        if class_index == toc_index:
            has_toc = True
            
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
    num_instances = parentof_matrix[0].size
    # only leave most likely parent
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


        else: # j no parent
            #class of the current column in question
            classx = class_mapping_list[int(raw_tensor['instances'].pred_classes[j])]
            
            #documentroot always doesn't have a parent
            if classx == "documentroot":
                continue
            if classx == "table" and has_toc:
                tensor_toc_index = ((raw_tensor["instances"].pred_classes).tolist()).index(class_mapping_list.index("tableofcontent"))
                parentof_matrix[tensor_toc_index, j] = 1.0
                parentof_matrix_full[tensor_toc_index, j] = 1.0
                continue
                

            found_good_parent = False
            good_parent_id = None
            good_parent_value = 0.0

            max_iterations = 1000
            iterations = 1
            while(not found_good_parent):
                found_good_parent = True
                maxidx = np.argmax(parentof_matrix_full[:,j]) #index of max value of all parents (even if background would be stronger)
                maxvalue = parentof_matrix_full[maxidx,j]
                # we already established that meta, toc, and article are kids of docroot so if another node has docroot as its parent we won't accept it
                if class_mapping_list[int(raw_tensor['instances'].pred_classes[maxidx])] == "documentroot":
                    parentof_matrix[maxidx, j] = 0.0
                    parentof_matrix_full[maxidx, j] = 0.0
                    found_good_parent = False
                
                # if the relation would build and antisymmetric we won't take it
                if parentof_matrix[j, maxidx] != 0:
                    parentof_matrix[maxidx, j] = 0.0
                    parentof_matrix_full[maxidx, j] = 0.0
                    found_good_parent = False

                if check_for_cycles(parentof_matrix, followedby_matrix, parentof_matrix_full, followedby_matrix_full):
                    parentof_matrix[maxidx, j] = 0.0
                    parentof_matrix_full[maxidx, j] = 0.0
                    found_good_parent = False
                
                
                good_parent_id = maxidx
                good_parent_value = maxvalue

                # if article doesn't have a child we set good parent_parent_value == 0.0 and we didnt find a good fit of the first iteration

                try:
                    tensor_article_index = ((raw_tensor["instances"].pred_classes).tolist()).index(class_mapping_list.index("article"))
                except:
                    tensor_article_index = ((raw_tensor["instances"].pred_classes).tolist()).index(class_mapping_list.index("tableofcontent"))
                if not (any(parentof_matrix[tensor_article_index,:])) and iterations >= 2:
                    good_parent_value = 0.0
                iterations += 1
                if iterations > max_iterations:
                    break

            if good_parent_value == 0.0:
                article_kids_list, meta_kids_list = create_article_and_meta_kids_list(class_mapping_list)
                has_toc_and_no_article = False
                try:
                    tensor_article_index = ((raw_tensor["instances"].pred_classes).tolist()).index(class_mapping_list.index("article"))
                except:
                    tensor_article_index = ((raw_tensor["instances"].pred_classes).tolist()).index(class_mapping_list.index("tableofcontent"))
                    has_toc_and_no_article = True
                
                if classx in meta_kids_list:
                    tensor_meta_index = ((raw_tensor["instances"].pred_classes).tolist()).index(class_mapping_list.index("meta"))
                    parentof_matrix[tensor_meta_index, j] = 1.0
                    parentof_matrix_full[tensor_meta_index, j] = 1.0
                    print(f"fallback heuristics used: instance {j} appended to meta")
                elif classx in article_kids_list:
                    if has_toc_and_no_article:
                        tensor_article_index = ((raw_tensor["instances"].pred_classes).tolist()).index(class_mapping_list.index("tableofcontent"))
                    else:
                        tensor_article_index = ((raw_tensor["instances"].pred_classes).tolist()).index(class_mapping_list.index("article"))
                    parentof_matrix[tensor_article_index, j] = 1.0
                    parentof_matrix_full[tensor_article_index, j] = 1.0
                    if has_toc_and_no_article:
                        print(f"fallback heuristics used: instance {j} appended to tableofcontent")
                    else:
                        print(f"fallback heuristics used: instance {j} appended to article")

                
            parentof_matrix[good_parent_id, j] = good_parent_value
            parentof_matrix_full[good_parent_id, j] = good_parent_value
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
    # we first make sure that there is a documentroot, article and meta
    # no relations added this just adds the instance with scores to the raw_tensor
    article_kids_list, meta_kids_list = create_article_and_meta_kids_list(class_mapping_list)
    
    has_root, has_article, has_meta, has_toc = has_root_article_meta_toc(raw_tensor, class_mapping_list)
    if not has_root:
        raw_tensor = create_root(raw_tensor, class_mapping_list)
        warnings.warn("no root instance found, this may lead to undefined behavior in postprocessing and hocr file creation", category=UserWarning)
    # only create new article when there's no article or tableofcontent present
    if not has_article and not has_toc:
        raw_tensor = create_article(raw_tensor, class_mapping_list)
        warnings.warn("no article or tableofcontent instance found, this may lead to undefined behavior in postprocessing and hocr file creation", category=UserWarning)
    if not has_meta:
        raw_tensor = create_meta(raw_tensor, class_mapping_list)
        warnings.warn("no meta instance found, this may lead to undefined behavior in postprocessing and hocr file creation", category=UserWarning)
    
    # create 4 matrices for parentof and followedby for easier checking and handling
    parentof_matrix, followedby_matrix, parentof_matrix_full, followedby_matrix_full= create_parentof_and_followedby_matrices(raw_tensor)

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





    return postprocessed_tensor
    
