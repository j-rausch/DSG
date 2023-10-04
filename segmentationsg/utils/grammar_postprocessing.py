from types import new_class
from xmlrpc.client import MAXINT
import torch
import os
import glob
from detectron2.data import MetadataCatalog
import yaml
import numpy as np
#import matplotlib

#matplotlib.use('Agg')  # turn off gui
from segmentationsg.utils.visualizer import SGVisualizer
from detectron2.utils.visualizer import ColorMode  # , Visualizer
from detectron2.utils.file_io import PathManager
from PIL import Image
from detectron2.data.detection_utils import _apply_exif_orientation
from detectron2.data.detection_utils import convert_PIL_to_numpy
from detectron2.structures.instances import Instances
from detectron2.structures.boxes import Boxes
import sys


def postprocess_prediction_with_grammar(prediction, thing_classes=None, has_documentroot=False):
    """
    predictions is a dict with keys "instances", "rel_pair_idxs", "pred_rel_scores"
    and within instances we have a tuple "instances" containing "num_instances", "image_height", "image_width",
    "fields"=[pred_boxes: Boxes()]" these fields are 7x4 tensors, where i think the 4 doubles refer to a box
    and "device" as well as a tensor "pred_classes" which are the ids of the predicted classes
    a tensor scores with the scores referring to the pred classes i think
    for the rest check out the images "whatispredictions"
    image_name = path.split("/")[-1]
    and now also includes a "rel_logit_matrix_tailcentric" key.
    """

    # since we simply set the third column to 1 if no other relations had a argmax, we just need to check this
    # labels are no longer present it seems?
    pred_rel_scores = prediction["pred_rel_scores"]
    rel_pair_idxs = prediction["rel_pair_idxs"]
    background_removal_mask = torch.where(pred_rel_scores[:, 2] == 0)
    prediction["pred_rel_scores"] = pred_rel_scores[background_removal_mask]
    prediction["rel_pair_idxs"] = rel_pair_idxs[background_removal_mask]
    orig_device = prediction['instances'].scores.device

    # print(predictions["instances"])
    # print( predictions["rel_pair_idxs"])
    # print( predictions["pred_rel_scores"])

    # TODO get this from the path in the config file.
    if thing_classes is not None:
        class_mapping_list = thing_classes
    else:
        class_mapping_list = ['article', 'author', 'backgroundfigure', 'col', 'contentblock',
                              'documentroot', 'figure', 'figurecaption', 'figuregraphic', 'foot',
                              'footnote', 'head', 'header', 'introduction', 'item', 'itemize', 'logo',
                              'meta', 'pagenr', 'row', 'table', 'tableofcontent', 'tabular', 'unk']
    # class_mapping_list = ["article", "author", "backgroundfigure", "col", "contentblock", "documentroot", "figure", "figurecaption", "figuregraphic", "foot", "footnote", "head", "header", "introduction", "item", "itemize", "logo", "meta", "orderedgroup", "pagenr", "row", "table", "tableofcontent","tabular","unorderedgroup"]

    if has_documentroot:
        #### check if there is a documentroot, if not create one #######
        global_idxdocumentroot = class_mapping_list.index("documentroot")

        # get the local and global idx of direct children of docroot
        documentrootkids_mappings = {"toc": {"global": -1, "local": -1},
                                     "meta": {"global": -1, "local": -1},
                                     "article": {"global": -1, "local": -1},
                                     "orderedgroup": {"global": -1, "local": -1},
                                     "unorderedgroup": {"global": -1, "local": -1}}
        if ("article" in class_mapping_list):
            documentrootkids_mappings["article"]["global"] = class_mapping_list.index("article")
            if (documentrootkids_mappings["article"]["global"] in prediction[
                "instances"].pred_classes):  # i know youre tempted to write a function for this. dont. its fine.
                documentrootkids_mappings["article"]["local"] = (
                    prediction["instances"].pred_classes.tolist()).index(
                    documentrootkids_mappings["article"]["global"])

        if ("toc" in class_mapping_list):
            documentrootkids_mappings["toc"]["global"] = class_mapping_list.index("toc")
            if (documentrootkids_mappings["toc"]["global"] in prediction[
                "instances"].pred_classes):  # i know youre tempted to write a function for this. dont. its fine.
                documentrootkids_mappings["toc"]["local"] = (
                    prediction["instances"].pred_classes.tolist()).index(
                    documentrootkids_mappings["toc"]["global"])

        if ("meta" in class_mapping_list):
            documentrootkids_mappings["meta"]["global"] = class_mapping_list.index("meta")
            if (documentrootkids_mappings["meta"]["global"] in prediction[
                "instances"].pred_classes):  # i know youre tempted to write a function for this. dont. its fine.
                documentrootkids_mappings["meta"]["local"] = (
                    prediction["instances"].pred_classes.tolist()).index(
                    documentrootkids_mappings["meta"]["global"])

        if ("unorderedgroup" in class_mapping_list):
            documentrootkids_mappings["unorderedgroup"]["global"] = class_mapping_list.index(
                'unorderedgroup')
            if (documentrootkids_mappings["unorderedgroup"]["global"] in prediction[
                "instances"].pred_classes):  # i know youre tempted to write a function for this. dont. its fine.
                documentrootkids_mappings["unorderedgroup"]["local"] = (
                    prediction["instances"].pred_classes.tolist()).index(
                    documentrootkids_mappings["unorderedgroup"]["global"])

        if ("orderedgroup" in class_mapping_list):
            documentrootkids_mappings["orderedgroup"]["global"] = class_mapping_list.index(
                "orderedgroup")
            if (documentrootkids_mappings["orderedgroup"]["global"] in prediction[
                "instances"].pred_classes):  # i know youre tempted to write a function for this. dont. its fine.
                documentrootkids_mappings["orderedgroup"]["local"] = (
                    prediction["instances"].pred_classes.tolist()).index(
                    documentrootkids_mappings["orderedgroup"]["global"])

        try:  # T!ODO change this to if else
            local_idxdocumentroot = ((prediction["instances"].pred_classes).tolist()).index(
                global_idxdocumentroot)
        except ValueError:
            # print("halt")
            # continue
            # in that case, documentroot must have had the same bbox as another object and thus was rejected
            # we add the documentroot back.
            # UPDATE actually, it is not possible to add another instance into the "Instances" object of detectron2 after the fact
            # so we have to skip these images.

            prediction["instances"].remove("pred_class_prob")
            # adding the object documentroot
            new_instance = Instances(prediction["instances"].image_size)
            documentrootbox = torch.tensor([0, 0, prediction["instances"].image_size[0],
                                            prediction["instances"].image_size[1]]).unsqueeze(0).to(orig_device)
            new_instance.pred_boxes = Boxes(documentrootbox)

            new_instance.pred_classes = torch.tensor([global_idxdocumentroot]).to(orig_device)
            local_idxdocumentroot = len(prediction["instances"].pred_classes)
            new_instance.scores = torch.tensor([1]).to(orig_device)

            # new_instance.pred_class_prob = torch.tensor([1])

            # now concatenate the old instances with the new instances:
            instanceslist = [prediction["instances"], new_instance]
            prediction["instances"] = Instances.cat(instanceslist)

            # adding the relations to documentroot
            # possiblechildrenofdocroot = ["article", "toc", "meta"]

            if (documentrootkids_mappings["meta"]["local"] != -1):
                prediction["rel_pair_idxs"] = torch.cat((prediction["rel_pair_idxs"], torch.tensor(
                    [local_idxdocumentroot, documentrootkids_mappings["meta"]["local"]]).unsqueeze(0).to(orig_device)))
                prediction["pred_rel_scores"] = torch.cat(
                    (prediction["pred_rel_scores"], torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0).to(orig_device)))

            if (documentrootkids_mappings["article"]["local"] != -1):
                prediction["rel_pair_idxs"] = torch.cat((prediction["rel_pair_idxs"], torch.tensor(
                    [local_idxdocumentroot, documentrootkids_mappings["article"]["local"]]).unsqueeze(
                    0).to(orig_device)))
                prediction["pred_rel_scores"] = torch.cat(
                    (prediction["pred_rel_scores"], torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0).to(orig_device)))

            if (documentrootkids_mappings["toc"]["local"] != -1):
                prediction["rel_pair_idxs"] = torch.cat((prediction["rel_pair_idxs"], torch.tensor(
                    [local_idxdocumentroot, documentrootkids_mappings["toc"]["local"]]).unsqueeze(0).to(orig_device)))
                prediction["pred_rel_scores"] = torch.cat(
                    (prediction["pred_rel_scores"], torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0).to(orig_device)))
    else:
        global_idxdocument = class_mapping_list.index("document")
        global_idxmeta = class_mapping_list.index("meta")
        local_idxdocument = ((prediction["instances"].pred_classes).tolist()).index(
            global_idxdocument)
        local_idxmeta = ((prediction["instances"].pred_classes).tolist()).index(
            global_idxmeta)

    # print(prediction["rel_logit_matrix_tailcentric"])
    # print(prediction["rel_logit_matrix_tailcentric"].shape)
    # import torch.nn.functional as F
    # rel_class_prob_matrix_tailcentric = F.softmax(prediction["rel_logit_matrix_tailcentric"], dim=1)
    # print(rel_class_prob_matrix_tailcentric)
    # print(rel_class_prob_matrix_tailcentric.shape)
    # tailcentric_argmaxes_parentof_per_tail_entity = torch.argmax(rel_class_prob_matrix_tailcentric[:, :-1, 1], dim=1)
    # print(tailcentric_argmaxes_parentof_per_tail_entity)
    # print(tailcentric_argmaxes_parentof_per_tail_entity.shape)
    # sys.exit()

    # create 2 matrices for parentof and followedby for easier checking and handling:
    num_instances = len(prediction["instances"])
    sg_parentof = np.zeros([num_instances, num_instances])
    sg_followedby = np.zeros([num_instances, num_instances])

    for i, pair in enumerate(prediction["rel_pair_idxs"]):
        rel = torch.argmax(prediction["pred_rel_scores"][i])
        if rel == 0:  # followedby
            sg_followedby[pair[0]][pair[1]] = prediction["pred_rel_scores"][i][rel]
        elif rel == 1:  # parentof
            sg_parentof[pair[0]][pair[1]] = prediction["pred_rel_scores"][i][rel]
        else:
            raise Exception("This is not a valid relation")

    import torch.nn.functional as F

    rel_class_prob_matrix_tailcentric = F.softmax(prediction["rel_logit_matrix_tailcentric"], dim=1)

    ####### parentof #######
    # we do not want documentroot to have a parent, nor a followedby relation:
    if has_documentroot:
        for i in range(num_instances):
            sg_parentof[i, local_idxdocumentroot] = 0.0
            sg_followedby[i, local_idxdocumentroot] = 0.0
            sg_followedby[local_idxdocumentroot, i] = 0.0
    else:
        for i in range(num_instances):
            sg_parentof[i, local_idxdocument] = 0.0
            sg_parentof[i, local_idxmeta] = 0.0


    # every node has exactly one parent, except for documentroot which has no parent
    for j in range(num_instances):
        if (any(sg_parentof[:, j])):
            maxidx = np.argmax(sg_parentof[:, j])  # index of max value
            maxvalue = sg_parentof[maxidx, j]
            sg_parentof[:, j] = np.zeros([num_instances])
            sg_parentof[maxidx, j] = maxvalue
        else:  # we add a parentof to documentroot
            if has_documentroot:
                sg_parentof[local_idxdocumentroot][j] = rel_class_prob_matrix_tailcentric[
                    j, local_idxdocumentroot, 1]
            else:
                sg_parentof[local_idxdocument][j] = rel_class_prob_matrix_tailcentric[
                    j, local_idxdocument, 1]
                sg_parentof[local_idxmeta][j] = rel_class_prob_matrix_tailcentric[
                    j, local_idxmeta, 1]

    # parentof is antisymmetric
    # if (a, parentof, b) and (b, parentof, a) we remove the relation with the lower score
    for i in range(num_instances):
        for j in range(num_instances):
            if (sg_parentof[i][j] != 0.0 and sg_parentof[j][i] != 0.0):
                if (sg_parentof[i][j] > sg_parentof[j][i]):
                    sg_parentof[j][i] = 0.0
                else:
                    sg_parentof[i][j] = 0.0

    #####followedby######

    # all children of a node should have a followedby relation
    for i in range(num_instances):
        children = []
        for j in range(num_instances):
            if (sg_parentof[i][j] != 0.0):
                children.append(j)

        assert (i not in children)  # no self relations
        # throw out followedby relation inbetween children of different parents
        # this also enforces that k1 can be the head of *only one* followedby relation
        # NOTE this is highly dependent on the order in which we process the children.
        # import random
        # random.shuffle(children)
        for k1 in children:
            children_sorted_by_score = torch.topk(
                rel_class_prob_matrix_tailcentric[children, k1, 0], len(children))
            for k2 in children_sorted_by_score.indices:
                nextmaxindex = children[k2]
                # check if this node is already tail of a "better" relation:
                # T!ODO turn this ifelse around
                if (sg_followedby[np.argmax(sg_followedby[:, nextmaxindex]), nextmaxindex] >
                        rel_class_prob_matrix_tailcentric[nextmaxindex, k1, 0]):
                    continue
                else:
                    sg_followedby[k1, :] = np.zeros(num_instances)
                    sg_followedby[k1][nextmaxindex] = rel_class_prob_matrix_tailcentric[
                        nextmaxindex, k1, 0]
                    break

    # followedby is antisymmetric
    for i in range(num_instances):
        for j in range(num_instances):
            if (sg_followedby[i][j] != 0.0 and sg_followedby[j][i] != 0.0):
                if (sg_followedby[i][j] > sg_followedby[j][i]):
                    sg_followedby[j][i] = 0.0
                else:
                    sg_followedby[i][j] = 0.0

    # no two followedby end in the same node:
    # T!ODO this actually never happens, unless I personally changed it in line 236 but build in an assertion just in case
    for i in range(num_instances):
        maxidx = np.argmax(sg_followedby[:, i])  # index of max value
        maxvalue = sg_followedby[maxidx, i]
        sg_followedby[:, i] = np.zeros([num_instances])
        sg_followedby[maxidx, i] = maxvalue

    # no two followedby start in the same node
    for i in range(num_instances):
        maxidx = np.argmax(sg_followedby[i, :])  # index of max value
        maxvalue = sg_followedby[i, maxidx]
        sg_followedby[i, :] = np.zeros([num_instances])
        sg_followedby[i, maxidx] = maxvalue

    if has_documentroot:
        # unorderedGroup nodes have no followedby relations
        if ("unorderedgroup" in class_mapping_list):
            unorderedGroup_local_idx = documentrootkids_mappings["unorderedgroup"]["local"]
            sg_followedby[:, unorderedGroup_local_idx] = np.zeros([num_instances])
            sg_followedby[unorderedGroup_local_idx, :] = np.zeros([num_instances])

    ###### parentof and followedby #########
    assert (((sg_parentof != 0.0) == (sg_followedby != 0.0)).any())
    # never both, in opposite directions
    for i in range(num_instances):
        for j in range(num_instances):
            if (sg_parentof[i][j] != 0.0 and sg_followedby[j][i] != 0.0):
                if (sg_parentof[i][j] > sg_followedby[j][i]):
                    sg_followedby[j][i] = 0.0
                else:
                    sg_parentof[i][j] = 0.0

    # no cycles
    sg_combined = np.zeros([num_instances, num_instances])
    for i in range(num_instances):
        for j in range(num_instances):
            if (sg_parentof[i][j] != 0.0 or sg_followedby[i][j] != 0.0):
                sg_combined[i][j] = 1

    import networkx as nx

    pred_graph = nx.OrderedDiGraph()
    for i in range(num_instances):
        pred_graph.add_node(i)
        for j in range(num_instances):
            pred_graph.add_node(j)
            if sg_parentof[i][j] != 0.0:
                pred_graph.add_edge(i, j, label="parentof", weight=sg_parentof[i][j])
            if sg_followedby[i][j] != 0.0:
                pred_graph.add_edge(i, j, label="followedby", weight=sg_followedby[i][j])

    cycles = nx.simple_cycles(pred_graph)
    cycles_list = list(cycles)
    MAX_ITERATIONS = 1000
    counter = 0
    while (cycles_list != [] and counter < MAX_ITERATIONS):
        counter = counter + 1
        for cycle in cycles_list:
            min_cycle_score = MAXINT
            min_cycle_edge = [None, None]
            min_cycle_rel = None

            for v in range(len(cycle)):
                # an edge is v and v+1
                v1 = cycle[v]
                v2 = cycle[(v + 1) % (len(cycle))]
                if (pred_graph[v1][v2]["weight"] < min_cycle_score):
                    min_cycle_score = pred_graph[v1][v2]["weight"]
                    min_cycle_pair = [v1, v2]
                    min_cycle_rel = pred_graph[v1][v2]["label"]

            assert (
                    min_cycle_score != MAXINT)  # If this happens we must have overflowed or something. do not keep working.

            if (min_cycle_rel == "parentof"):
                sg_parentof[min_cycle_pair[0]][min_cycle_pair[1]] = 0
            if (min_cycle_rel == "followedby"):
                sg_followedby[min_cycle_pair[0]][min_cycle_pair[1]] = 0

            sg_combined[min_cycle_pair[0], min_cycle_pair[1]] = 0
            pred_graph.remove_edge(min_cycle_pair[0], min_cycle_pair[1])
            break  # now the cycles list will be different, as this edge might have been in several cycles
            # this is kind of stupid, T!ODO make this smarter
        cycles = nx.simple_cycles(pred_graph)
        cycles_list = list(cycles)

    #### back to tensors ##########
    rel_pair_idxs_list = []
    pred_rel_scores_list = []

    sum_sg_parentof = np.sum(sg_parentof, axis=1)
    sum_sg_followedby = np.sum(sg_parentof, axis=1)
    sum_sg_parentof_dim0 = np.sum(sg_parentof, axis=0)
    sum_sg_followedby_dim0 = np.sum(sg_parentof, axis=0)


    for i in range(len(sg_followedby)):
        for j in range(len(sg_followedby)):
            if sg_followedby[i][j] != 0.0:
                rel_pair_idxs_list.append([i, j])
                pred_rel_scores_list.append([sg_followedby[i][j], 0, 0])

    for i in range(len(sg_parentof)):
        for j in range(len(sg_parentof)):
            if sg_parentof[i][j] != 0.0:
                rel_pair_idxs_list.append([i, j])
                pred_rel_scores_list.append([0, sg_parentof[i][j], 0])

    assert (len(rel_pair_idxs_list) == len(pred_rel_scores_list))
    prediction["pred_rel_scores"] = torch.tensor(pred_rel_scores_list)
    prediction["rel_pair_idxs"] = torch.tensor(rel_pair_idxs_list)

    return prediction

