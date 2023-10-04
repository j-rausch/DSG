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

def followedby(dgg_class1_path, dgg_class2_path, root_hocr):
    ret = []
    for node in root_hocr.xpath(dgg_class1_path):
        flwdby = node.attrib["dgg_followedby"]
        for n in root_hocr.xpath(dgg_class2_path):
            if flwdby == n.attrib['dgg_id']:
                ret.append(n)
    return ret

def create_graphs(postprocessed_tensor, class_mapping_list):
    #graph with only parentof edges
    graph_parentof = nx.DiGraph()
    #graph with all edges after postprocessing
    graph_after = nx.DiGraph()
    for i in range(0, len(postprocessed_tensor['instances'].pred_boxes)):
        graph_parentof.add_node(i, label = class_mapping_list[postprocessed_tensor['instances'].pred_classes[i]])
        graph_after.add_node(i, label = class_mapping_list[postprocessed_tensor['instances'].pred_classes[i]])
    
    for i, pair in enumerate(postprocessed_tensor["rel_pair_idxs"]):
        rel = torch.argmax(postprocessed_tensor["pred_rel_scores"][i])
        if rel == 0: #followedby
            graph_after.add_edge(int(pair[0]), int(pair[1]), label="followedby")
            
        elif rel == 1: #parentof
            graph_after.add_edge(int(pair[0]), int(pair[1]), label="parentof")
            graph_parentof.add_edge(int(pair[0]), int(pair[1]), label="parentof")
            
    return graph_parentof, graph_after

def scale_ocr_x(x, dimensions_scenegraph, dimensions_ocr):
    return x * dimensions_scenegraph[0] / dimensions_ocr[0]
def scale_ocr_y(y, dimensions_scenegraph, dimensions_ocr):
    return y * dimensions_scenegraph[1] / dimensions_ocr[1]

def convert_to_xywh(bbox):
    return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

def convert_to_2_cornerpoints(bbox):
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

def iou_xywh(bbox1, bbox2):
    from shapely.geometry import Polygon
    bbox1_a = [[bbox1[0], bbox1[1]], [bbox1[0] + bbox1[2], bbox1[1]], [bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]], [bbox1[0], bbox1[1] + bbox1[3]]]
    bbox2_a = [[bbox2[0], bbox2[1]], [bbox2[0] + bbox2[2], bbox2[1]], [bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]], [bbox2[0], bbox2[1] + bbox2[3]]]
    poly_1 = Polygon(bbox1_a)
    poly_2 = Polygon(bbox2_a)
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou

def iou_x1y1x2y2(bbox1, bbox2):
    from shapely.geometry import Polygon
    bbox1_a = [ [bbox1[0], bbox1[1]], [bbox1[2], bbox1[1]], [bbox1[2], bbox1[3]], [bbox1[0], bbox1[3]]]
    bbox2_a = [ [bbox2[0], bbox2[1]], [bbox2[2], bbox2[1]], [bbox2[2], bbox2[3]], [bbox2[0], bbox2[3]]]
    poly_1 = Polygon(bbox1_a)
    poly_2 = Polygon(bbox2_a)
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou


def attach_head_and_body(root):
    head = ET.Element("head")
    body = ET.Element("body")
    root.append(head)
    root.append(body)
    meta1 = ET.Element("meta")
    meta1.set('name', 'ocr-system')
    meta1.set('content', 'eperiodica_fulltext')
    
    meta2 = ET.Element("meta")
    meta2.set('name', 'ocr-capabilities')
    meta2.set('content', 'ocr_page ocr_author ocr_carea ocr_photo ocr_caption ocr_linear ocr_footer ocr_header ocr_pageno ocr_table ocr_section ocrx_block ocrx_word')
    
    head.append(meta1)
    head.append(meta2)
    return head, body

#returns the node index that i is followedby if it exists, otherwise "None"
def get_followedby(postprocessed_tensor, i):
    followedby = "None"
    followedby_score = -1.0
    for index, rel in enumerate(postprocessed_tensor['rel_pair_idxs']):
        if int(rel[0]) == i:
            if torch.argmax(postprocessed_tensor['pred_rel_scores'][index]) == 0:
                followedby=str(int(rel[1]))

    return followedby

#def create_dgg_node():
#
#def create_hocr_node():
    
def create_xml_node(ids, dgg_id, bbox, dgg_class, dgg_score, dgg_children, dgg_followedby):
    bbox[0], bbox[1], bbox[2], bbox[3], ids, dgg_children, dgg_id = str(int(bbox[0])), str(int(bbox[1])), str(int(bbox[2])), str(int(bbox[3])), str(ids), str(dgg_children), str(dgg_id)
    dgg_hocr_mapping = {
    "documentroot": "ocr_page",
    "meta": "None",
    "author": "ocr_author",
    "backgroundfigure": "ocr_float",
    "contentblock": "ocrx_block",
    "figure": "ocr_float",
    "figuregraphic": "ocr_photo",
    "figurecaption": "ocr_caption",
    "foot": "ocr_footer",
    "footnote": "ocr_footer",
    "head": "ocr_header",
    "header": "ocr_header",
    "item": "ocr_carea",
    "itemize": "ocr_float",
    "orderedgroup": "ocr_carea",
    "pagenr": "ocr_pageno",
    "tabular": "ocr_table",
    "table": "ocr_table",
    "unorderedgroup": "ocr_float",
    "article": "None",
    "tableofcontent": "ocr_table",
    "col": "ocr_carea",
    "row": "ocr_carea"}
    
    span_div_mapping = {
    "ocr_page":"div",
    "ocr_author":"div",  
    "ocr_float":"span",    
    "ocrx_block":"span",
    "ocr_photo":"span",
    "ocr_caption":"span",
    "ocr_footer":"span",
    "ocr_header":"span",
    "ocr_carea":"div",
    "ocr_pageno":"span",
    "ocr_table":"span",
    "None":"div"}
    
    #since article and meta are supposed to not have an hocr node we create a special case for them
    if dgg_class == "meta" or dgg_class == "article":
        dgg_node = ET.Element("div")
        dgg_node.set("dgg_class", dgg_class)
        dgg_node.set("id", ids)
        ids = str(int(ids) + 1)
        dgg_node.set("dgg_id", dgg_id)
        dgg_node.set("dgg_score", dgg_score)
        dgg_node.set("dgg_children", dgg_children)
        dgg_node.set("dgg_followedby", dgg_followedby)
        
        #parse bbox from string to int again for later applications
        bbox[0], bbox[1], bbox[2], bbox[3] = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        return dgg_node, int(ids)
    
    else:
        hocr_class = dgg_hocr_mapping[dgg_class]
    
        hocr_node = ET.Element(span_div_mapping[hocr_class])
        hocr_node.set("class", hocr_class)
        hocr_node.set("id", ids)
        ids = str(int(ids) + 1)
        hocr_node.set('title', 'bbox '+bbox[0]+" "+bbox[1]+" "+bbox[2]+" "+bbox[3])
        
        dgg_node = ET.Element("div")
        dgg_node.set("dgg_class", dgg_class)
        dgg_node.set("id", ids)
        ids = str(int(ids) + 1)
        dgg_node.set("dgg_id", dgg_id)
        dgg_node.set("dgg_score", dgg_score)
        dgg_node.set("dgg_children", dgg_children)
        dgg_node.set("dgg_followedby", dgg_followedby)
        
        hocr_node.append(dgg_node)
        
        #parse bbox from string to int again for later applications
        bbox[0], bbox[1], bbox[2], bbox[3] = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        return hocr_node, int(ids)
    

def create_hocr_skeleton(bbox_list, graph_parentof, graph_after, postprocessed_tensor, class_mapping_list, ids):
    #create root node and attach head and body
    root = ET.Element("html")
    head, body = attach_head_and_body(root) 
    
    #create list of xml nodes, each representing a detected bounding box
    amount_of_bboxes = len(postprocessed_tensor['instances'].pred_boxes)
    xml_node_list = []
    
    
    #create xml node list
    for i in range(0, amount_of_bboxes):
        xml_node_list.append([])
        
    #create each xml node 
    for i in range(0, amount_of_bboxes):
        #bbox
        bbox = bbox_list[i]
        #class
        dgg_class = graph_parentof.nodes[i]['label']
        dgg_score = str(float(postprocessed_tensor['instances'].scores[i]))
        
        #parentof
        dgg_children = list(graph_parentof.neighbors(i))
        if dgg_children == []:
            dgg_children = "None"
        
        #followedby
        dgg_followedby = get_followedby(postprocessed_tensor, i)
        
        xml_node_list[i], ids = create_xml_node(ids, i, bbox, dgg_class, dgg_score, dgg_children, dgg_followedby)
    
    #append xml_nodes in correct order to each other
    for i in range(0, amount_of_bboxes):
        #class of i-th node
        classx = class_mapping_list[postprocessed_tensor['instances'].pred_classes[i]]
        
        dgg_children = list(graph_parentof.neighbors(i))
        
        #set order of children
        dgg_children_sorted = []
        
        #subgraph induced on children of i-th node
        subgraph = graph_after.subgraph(dgg_children).copy()
        
        #if a node is an isolate in the subgraph it means that it isn't part of any followedby relations. These nodes are added last
        subgraph_isolates = list(nx.isolates(subgraph))
        subgraph.remove_nodes_from(subgraph_isolates)
        
        #nodes with followedby relations left, sort them inside of connected components
        subgraph_connected_component_list = []
        
        #for each component run a dfs and add nodes in the order of dfs
        for component in nx.connected_components(subgraph.to_undirected(as_view=True)):
            #find root of each connected component 
            local_source = -1
            for node in component:
                if list(subgraph.predecessors(node)) == []:
                    local_source = node
                    
            # add the order of the connected component to the connected component list      
            subgraph_connected_component_list.append(list(nx.dfs_preorder_nodes(subgraph.subgraph(component).copy(), source=local_source)))
        
        #correct order children
        connected_components_flattened = [item for sublist in subgraph_connected_component_list for item in sublist]
        dgg_children_sorted = connected_components_flattened + subgraph_isolates
        
        
        
        for child in dgg_children_sorted:
            if classx == "article" or classx == "meta":
                xml_node_list[i].append(xml_node_list[child])
            else:
                xml_node_list[i][0].append(xml_node_list[child])
    
    dgg_documentroot_index = class_mapping_list.index("documentroot")
    tensor_documentroot_index = ((postprocessed_tensor["instances"].pred_classes).tolist()).index(dgg_documentroot_index)
    
    body.append(xml_node_list[tensor_documentroot_index])
    
    
    return root, xml_node_list, ids

def create_hocr_word(ids, word, bbox):
    bbox[0], bbox[1], bbox[2], bbox[3] = str(int(bbox[0])), str(int(bbox[1])), str(int(bbox[2])), str(int(bbox[3]))
    
    hocr_word = ET.Element("span")
    hocr_word.set('class', 'ocrx_word')
    hocr_word.set('id', str(ids))
    ids = str(int(ids) + 1)
    hocr_word.set('title', 'bbox'+bbox[0]+" "+bbox[1]+" "+bbox[2]+" "+bbox[3])
    hocr_word.text=word
    
    return hocr_word, int(ids)
    
    
    
    
def create_hocr(postprocessed_tensor, ocr_fulltext_path, class_mapping_list, output_folder=None, filename=None):
    #create graphs
    graph_parentof, graph_after = create_graphs(postprocessed_tensor, class_mapping_list)
    
    #create list of boundingboxes of postprocessed_tensor
    bbox_list = []
    for bbox in postprocessed_tensor['instances'].pred_boxes:
        bbox_list.append([bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()])
    
    # load ocr fulltext
    f = open(ocr_fulltext_path)
    
    #dimensions of ocr text
    dimensions_ocr = f.readline()
    dimensions_ocr = dimensions_ocr.split(",", 2)
    dimensions_ocr[0], dimensions_ocr[1] = float(dimensions_ocr[0]), float(dimensions_ocr[1])
    
    #dimensions of scenegraph
    height, width = postprocessed_tensor['instances'].image_size
    dimensions_scenegraph = [width, height]
    
    #we want an id for each xml node created
    ids = 0
    
    #create hocr skeleton
    root_hocr, xml_node_list, ids = create_hocr_skeleton(bbox_list, graph_parentof, graph_after, postprocessed_tensor, class_mapping_list, ids)
    
    #ocr text
    lines = f.readlines()
    
    #leaf nodes of parentof graph
    leaf_nodes = [node for node in graph_parentof.nodes() if graph_parentof.in_degree(node)!=0 and graph_parentof.out_degree(node)==0]
    
    #append each word in the ocr text to correct xml node in hocr skeleton
    for l in lines:
        #line format is: Word x, y, w, h or <EOP>/<EOS> (end of paragraph, end of sentence)
        if l != "<EOP>\n" and l != "<EOS>\n":
            l = l.split(" ", 1)
            #bounding box of word in l[1]
            l[1] = l[1].split(",", 4)
            
            #word coordinates scaled to scenegraph dimensions
            x, y, w, h = scale_ocr_x(float(l[1][0]), dimensions_scenegraph, dimensions_ocr), scale_ocr_y(float(l[1][1]), dimensions_scenegraph, dimensions_ocr), scale_ocr_x(float(l[1][2]), dimensions_scenegraph, dimensions_ocr), scale_ocr_y(float(l[1][3]), dimensions_scenegraph, dimensions_ocr)
            word_bbox = [x, y, w, h]
            
            word_bbox = convert_to_2_cornerpoints(word_bbox)
            #intersection over union list with all leaf nodes
            iou_list = []
            for bbox in postprocessed_tensor['instances'].pred_boxes:
                iou_list.append(0.0)
                
            for leaf_node in leaf_nodes:
                bbox = bbox_list[leaf_node]
                #these conversions are needed since the dgg bbox coordinates are in x1y1x2y2 format and the eperiodica ocr is in xywh
                iou_list[leaf_node] = iou_x1y1x2y2(word_bbox, bbox)
            
            #best fitting xml node for word, !TODO: heuristic for word that isn't in any bounding box (attach to same as last node)
            maxiou_index = np.argmax(iou_list)
            
            word = l[0]
            
            #create hocr word xml node
            hocr_word, ids = create_hocr_word(ids, word, word_bbox)
            
            #append to correct entity, in very rare cases we might create a meta or article that is a leaf node therefore except
            #is there, could add a heuristic here potentially
            try:
                xml_node_list[maxiou_index][0].append(hocr_word)
            except IndexError:
                xml_node_list[maxiou_index].append(hocr_word)
            
    
    if output_folder:
        tree = ET.ElementTree(root_hocr)
        ET.indent(tree, space=" ", level=0)
        tree.write(output_folder+filename, encoding='utf-8')
        
        print(filename+" saved at "+output_folder+filename)
    
    return root_hocr