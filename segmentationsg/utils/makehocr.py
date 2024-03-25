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
import warnings

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
import shapely

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

def find_root_node(graph):
    root_node = 0
    for node in graph.nodes:
        if graph.nodes[node]['label'] == "documentroot":    
            root_node = node
    return root_node

import xml.dom.minidom

def create_xml_skeleton(graph_parentof, graph_full):
    def get_ordered_children(node, graph_full, graph_parentof):
        dgg_children = list(graph_parentof.neighbors(node))

        #set order of children
        dgg_children_sorted = []
        #subgraph induced on children of node
        subgraph = graph_full.subgraph(dgg_children).copy()
        #if a node is an isolate in the subgraph it means that it isn't part of any followedby relations. These nodes are added last
        subgraph_isolates = list(nx.isolates(subgraph))

        subgraph.remove_nodes_from(subgraph_isolates)
        #nodes with followedby relations left, sort them inside of connected components
        subgraph_connected_component_list = []
        
        #for each component run a dfs and add nodes in the order of dfs
        for component in nx.connected_components(subgraph.to_undirected(as_view=True)):
            #find root of each connected component 
            local_source = -1
            for node1 in component:
                if list(subgraph.predecessors(node1)) == []:
                    local_source = node1
            # add the order of the connected component to the connected component list      
            subgraph_connected_component_list.append(list(nx.dfs_preorder_nodes(subgraph.subgraph(component).copy(), source=local_source)))
        
        #correct order children
        connected_components_flattened = [item for sublist in subgraph_connected_component_list for item in sublist]
        dgg_children_sorted = connected_components_flattened + subgraph_isolates
            

        return dgg_children_sorted
    
    def add_children_to_xml(parent_xml_node, parent_graph_node):
        # Iterate over all children of the current node in the graph
        for child in get_ordered_children(parent_graph_node, graph_full, graph_parentof):
            child_label = graph_parentof.nodes[child]['label']
            # Create an XML element for the child node
            child_xml_node = ET.SubElement(parent_xml_node, child_label, id=str(child))
            # Recursively add children of this child node to the XML
            add_children_to_xml(child_xml_node, child)
    
    root_node = find_root_node(graph_parentof)

    # Create the root element of the XML document
    root_label = graph_parentof.nodes[root_node]['label']
    root_xml_node = ET.Element(root_label)

    # Start the recursive process to add all children to the XML root
    add_children_to_xml(root_xml_node, root_node)

    # Convert the XML structure to a string representation
    xml_str = ET.tostring(root_xml_node, encoding='utf8', method='xml').decode()

    # Pretty print the XML string
    dom = xml.dom.minidom.parseString(xml_str)  # Parse the string
    pretty_xml_str = dom.toprettyxml(indent="  ")  # Pretty print with indentation

    # Optionally, you can write the XML string to a file
    with open('graph.xml', 'w') as file:
        file.write(pretty_xml_str)

    return root_xml_node

def adjust_xml_skeleton(root_xml_node, postprocessed_tensor, graph_full, graph_parentof):
    def get_followedby_id(node_id):
        for source, target, data in graph_full.edges(data=True):
            if source == node_id and data['label'] == 'followedby':
                return target
        return None

    def adjust_node(node, node_id):
        node.set('dgg_class', node.tag)
        node.set('dgg_id', str(node_id))
        node.set('dgg_score', str(postprocessed_tensor['instances'].scores[node_id].item()))
        
        children_ids = [child for child in graph_parentof.successors(node_id)]
        node.set('dgg_children', str(children_ids))
        
        followedby_id = get_followedby_id(node_id)
        node.set('dgg_followedby', str(followedby_id) if followedby_id is not None else 'None')

        # Remove the original 'id' attribute
        if 'id' in node.attrib:
            del node.attrib['id']

        node.tag = 'div'

        for child in node:
            child_id = int(child.get('id'))
            adjust_node(child, child_id)

    root_node = find_root_node(graph_parentof)

    adjust_node(root_xml_node, root_node)

    # Convert the XML structure to a string representation
    xml_str = ET.tostring(root_xml_node, encoding='utf8', method='xml').decode()

    # Pretty print the XML string
    dom = xml.dom.minidom.parseString(xml_str)  # Parse the string
    pretty_xml_str = dom.toprettyxml(indent="  ")  # Pretty print with indentation

    # Optionally, you can write the XML string to a file
    with open('graph.xml', 'w') as file:
        file.write(pretty_xml_str)
    return root_xml_node


def transform_to_hocr(root_xml_node, postprocessed_tensor):

    def create_hocr_node(dgg_node):
        #dgg node
        dgg_node_copy = ET.Element(dgg_node.tag)
        dgg_node_copy.attrib = dgg_node.attrib.copy()

        #hocr node
        dgg_class = dgg_node_copy.get('dgg_class')
        hocr_class = dgg_hocr_mapping.get(dgg_class, 'None')
        hocr_node = ET.Element('div')
        hocr_node.set('class', hocr_class)

        dgg_id = int(dgg_node.get('dgg_id'))
        #create list of boundingboxes of postprocessed_tensor
        bbox_list = []
        for bbox in postprocessed_tensor['instances'].pred_boxes:
            bbox_list.append([bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()])
        bbox = bbox_list[dgg_id]
        
        hocr_node.set('title', f"bbox {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}")

        if hocr_class == 'None':
            return dgg_node_copy
        
        hocr_node.append(dgg_node_copy)

        return hocr_node
    
    def append_to_children(dgg_node):
        hocr_node = create_hocr_node(dgg_node)
        for child in dgg_node:
            if dgg_node.get('dgg_class') == "meta" or dgg_node.get('dgg_class') == "article":
                hocr_node.append(append_to_children(child))
            else:
                hocr_node[0].append(append_to_children(child))
        return hocr_node
    
    def adjust_tags(node):
        for child in node:
            hocr_class = child.get('class', 'None')
            new_tag = span_div_mapping.get(hocr_class, 'div')
            if child.tag != new_tag:
                child.tag = new_tag
            adjust_tags(child)


    
    root_hocr_node = append_to_children(root_xml_node)
    
    # Adjust tags according to span_div_mapping
    adjust_tags(root_hocr_node)
    
    

    return root_hocr_node

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

def create_hocr_word(word, bbox):
    bbox[0], bbox[1], bbox[2], bbox[3] = str(int(bbox[0])), str(int(bbox[1])), str(int(bbox[2])), str(int(bbox[3]))
    
    hocr_word = ET.Element("span")
    hocr_word.set('class', 'ocrx_word')
    hocr_word.set('title', 'bbox'+bbox[0]+" "+bbox[1]+" "+bbox[2]+" "+bbox[3])
    hocr_word.text=word
    
    return hocr_word

           
def find_maxiou_node(maxiou_index, root_hocr):
    for child in root_hocr.iter():
        dgg_id = child.get('dgg_id', 'None')
        if dgg_id != 'None':
            dgg_id = int(dgg_id)
        
        if dgg_id == maxiou_index:
            return child
        else:
             return None

def find_closest_leaf_node(word_bbox, root_hocr, leaf_nodes):
    pass
                

def create_hocr(postprocessed_tensor, ocr_fulltext_path, class_mapping_list, output_folder=None, filename=None):
    graph_parentof, graph_full = create_graphs(postprocessed_tensor, class_mapping_list)
    if not nx.is_tree(graph_parentof):
        warnings.warn("postprocessing failed to generate correct tree structure, hocr file may be incomplete", category=UserWarning)
    root_xml_node = create_xml_skeleton(graph_parentof, graph_full)
    root_xml_node = adjust_xml_skeleton(root_xml_node, postprocessed_tensor, graph_full, graph_parentof)
    root_hocr_node = transform_to_hocr(root_xml_node, postprocessed_tensor)
    

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

    #ocr text
    lines = f.readlines()

    #leaf nodes of parentof graph
    leaf_nodes = [node for node in graph_parentof.nodes() if graph_parentof.in_degree(node)!=0 and graph_parentof.out_degree(node)==0]

    prob = False
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
            hocr_word = create_hocr_word(word, word_bbox)
            #find correspond hocr node for maxiou
            maxiou_index = str(maxiou_index)
            maxiou_node = None
            for child in root_hocr_node.iter():
                dgg_id = child.get('dgg_id', 'None')
                if dgg_id == maxiou_index:
                    maxiou_node = child
            #maxiou_node = find_maxiou_node(maxiou_index, root_hocr_node)
            if maxiou_node != None:
                maxiou_node.append(hocr_word)
            else:
                pass
                prob = True
    




      
    idx = 0
    for child in root_hocr_node.iter():
        child.set('id', str(idx))
        idx += 1        
    
    root = ET.Element("html")
    _, body = attach_head_and_body(root)

    body.append(root_hocr_node)        

    xml_str = ET.tostring(root, encoding='utf8', method='xml').decode()
    # Pretty print the XML string
    dom = xml.dom.minidom.parseString(xml_str)  # Parse the string
    pretty_xml_str = dom.toprettyxml(indent="  ")  # Pretty print with indentation

    # Optionally, you can write the XML string to a file
    with open('graph.xml', 'w') as file:
        file.write(pretty_xml_str)
    
    if output_folder:
        tree = ET.ElementTree(root)
        ET.indent(tree, space=" ", level=0)
        tree.write(output_folder+filename, encoding='utf-8')
        
        print(filename+" saved at "+output_folder+filename)

    return root





