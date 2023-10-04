from detectron2.utils.visualizer import ColorMode, Visualizer, GenericMask
#from segmentationsg.utils.mapping_dictionaries import class_names_to_new_names, class_names_to_colors
from .mapping_dictionaries import class_names_to_new_names, class_names_to_colors
import logging
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
_SMALL_OBJECT_AREA_THRESH = 1000
_LARGE_MASK_AREA_THRESH = 120000
_OFF_WHITE = (1.0, 1.0, 240.0 / 255)
_BLACK = (0, 0, 0)
_RED = (1.0, 0, 0)

_KEYPOINT_THRESHOLD = 0.05
from detectron2.utils.colormap import random_color


logger = logging.getLogger(__name__)

class SGVisualizer(Visualizer):
    def __init__(self, img_rgb, metadata=None, scale=1.0, instance_mode=ColorMode.IMAGE):
        super().__init__(img_rgb, metadata=metadata, scale=scale, instance_mode=instance_mode)

    def generate_relation_graph(self, instances, rel_pair_idx, pred_rel_scores, pred_classes, ax=None):
        #instances = predictions["instances"].to(self.cpu_device)

        rel_score_threshold = 0.01
        # rel_pair_id
        #instance_id_to_label = {i: label for i, label in enumerate(['background'] + self.metadata.thing_classes)}

        #classes = instances.pred_classes if instances.has("pred_classes") else None
        #scores = instances.scores if instances.has("scores") else None
        labels = _create_numbered_text_labels(pred_classes, None, self.metadata.get("thing_classes", None), add_numbers=True)
        instance_id_to_label = {i:label for i,label in enumerate(labels)}

        predicate_id_to_label = {i: label for i, label in enumerate(self.metadata.predicate_classes + ['background'] )}

#        pred_rel_pair = predictions['rel_pair_idxs'].to(self.cpu_device).tolist()
#        pred_rel_scores = predictions['pred_rel_scores'].to(self.cpu_device)
        pred_rel_pair = rel_pair_idx
        #pred_rel_
        pred_rel_scores[:, -1] = 0
        if len(pred_rel_scores) == 0:
            pred_rel_score, pred_rel_label = [], []
            print('warning: no FG relations found!')
        else:
            pred_rel_score, pred_rel_label = pred_rel_scores.max(-1)
            print('mean relation score: {}'.format(np.mean(pred_rel_score.cpu().numpy())))

#        pred_rels = np.column_stack((pred_rel_pair, pred_rel_scores[:, :-1].argmax(1)))  # Backround index at the end
#        pred_scores = pred_rel_scores[:, :-1].max(1)  # Backround index at the end

        # print('relation scores: {}'.format(pred_rel_score.cpu().numpy()))
        # print('total of {} predicted relation pairs: {}'.format(len(pred_rel_pair), pred_rel_pair))

#
#        mask = pred_rel_score > rel_score_threshold
#        pred_rel_score = pred_rel_score[mask]
#        print('mean relation score after thresholding: {}'.format(np.mean(pred_rel_score.cpu().numpy())))
#        pred_rel_label = pred_rel_label[mask]
#
#        mask_list = mask.tolist()
#
#        pred_rel_pair = [x for i, x in enumerate(pred_rel_pair) if mask_list[
#            i] is True]  #
#        print('after thresholding: total of {} predicted relation pairs'.format(len(pred_rel_pair)))



        if ax is None:
            fig, ax = plt.subplots(1,1,figsize=(16,16))
            #ax[0].axis('off')
            #fig, ax = plt.subplots(1, 2, figsize=(30, 15))
            ax.axis('off')
        else:
            fig = None #in case we just want to fill the ax object, e.g. in a jupyter notebook

        #print("Predicted Graph:")
        pred_graph = nx.OrderedDiGraph()

        if len(pred_rel_scores) > 0:
            pred_rels = [(instance_id_to_label[i[0]], predicate_id_to_label[j], instance_id_to_label[i[1]]) for i, j in
                         zip(pred_rel_pair, pred_rel_label.tolist())]
            for pred_rel in pred_rels:
                subj, pred, obj = pred_rel
                subj_id = subj.split(' -')[0]
                obj_id = obj.split(' -')[0]
                pred_graph.add_node(subj_id, label=subj)
                pred_graph.add_node(obj_id, label=obj)
                pred_graph.add_edge(subj_id, obj_id, label=pred)

        #also add any nodes that are not connected anywhere
        for i, label in instance_id_to_label.items():
            pred_graph.add_node('#{}'.format(i), label=label)

        pos = nx.spring_layout(pred_graph, k=1, iterations=100)
        nx.draw(pred_graph, pos, ax=ax, arrows=True)
        node_labels = nx.get_node_attributes(pred_graph, 'label')
        nx.draw_networkx_labels(pred_graph, pos, node_labels, font_size=9, ax=ax)
        edge_labels = nx.get_edge_attributes(pred_graph, 'label')
        nx.draw_networkx_edge_labels(pred_graph, pos, edge_labels, font_size=7, ax=ax)
        #plt.show()
        if fig is not None:
            return fig
        else:
            return fig

        #return pred_rels

    def draw_instance_predictions(self, predictions, use_gt=False, fake_bboxes=False):
        """
        Draw instance-level prediction results on an image.

        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

        Returns:
            output (VisImage): image object with visualizations.
        """
        if use_gt is False:
            boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
            scores = predictions.scores if predictions.has("scores") else None
            classes = predictions.pred_classes if predictions.has("pred_classes") else None
        else:
            boxes = predictions.gt_boxes if predictions.has("gt_boxes") else None
            scores = None
            classes = predictions.gt_classes if predictions.has("gt_classes") else None
        #labels = _create_numbered_text_labels(classes, scores, self.metadata.get("thing_classes", None), )
        #TODO: intermediate ID to orig class name mapping
        thing_classes = self.metadata.get("thing_classes", None)
        #original_class_names = [thing_classes[c.item()] for c in classes]
        updated_thing_classes = [ class_names_to_new_names[x] for x in thing_classes]
        updated_thing_colors = [ [int(y * 255) for y in class_names_to_colors[x]] for x in thing_classes]
        

        labels = _create_numbered_text_labels(classes, None, updated_thing_classes, True)
        #labels = _create_numbered_text_labels(classes, None, self.metadata.get("thing_classes", None), True)
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        else:
            if fake_bboxes is False:
                masks = None
            else:
                raise NotImplementedError
#            else:
#                #convert instance bounding boxes to masks
#                masks = []
#                for box in boxes:
#                    x0, y0, x1, y1 = box.cpu().numpy().tolist()
#                    mask = GenericMask([x0,y0,x0,y1,x1,y1,x1,y0], self.output.height, self.output.width)
#                    masks.append(mask)

        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[c.item()]]) for c in classes
            ]
            alpha = 0.8
        else:
            colors = None
#            colors = [
#                self._jitter([min(0.999,x / 255) for x in updated_thing_colors[c.item()]]) for c in classes #cap at 0.999 to avoid error when 'lighter_colors' are generated by detectron2
#                #self._jitter([x  for x in updated_thing_colors[c.item()]]) for c in classes 
#            ]
            alpha = 0.5

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.img = self._create_grayscale_image(
                (predictions.pred_masks.any(dim=0) > 0).numpy()
                if predictions.has("pred_masks")
                else None
            )
            alpha = 0.3

        self.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output
    
   
    #re-implement overlay_instances to fix error with ligher_color range 
    def overlay_instances(
        self,
        *,
        boxes=None,
        labels=None,
        masks=None,
        keypoints=None,
        assigned_colors=None,
        alpha=0.5,
    ):
        """
        Args:
            boxes (Boxes, RotatedBoxes or ndarray): either a :class:`Boxes`,
                or an Nx4 numpy array of XYXY_ABS format for the N objects in a single image,
                or a :class:`RotatedBoxes`,
                or an Nx5 numpy array of (x_center, y_center, width, height, angle_degrees) format
                for the N objects in a single image,
            labels (list[str]): the text to be displayed for each instance.
            masks (masks-like object): Supported types are:

                * :class:`detectron2.structures.PolygonMasks`,
                  :class:`detectron2.structures.BitMasks`.
                * list[list[ndarray]]: contains the segmentation masks for all objects in one image.
                  The first level of the list corresponds to individual instances. The second
                  level to all the polygon that compose the instance, and the third level
                  to the polygon coordinates. The third level should have the format of
                  [x0, y0, x1, y1, ..., xn, yn] (n >= 3).
                * list[ndarray]: each ndarray is a binary mask of shape (H, W).
                * list[dict]: each dict is a COCO-style RLE.
            keypoints (Keypoint or array like): an array-like object of shape (N, K, 3),
                where the N is the number of instances and K is the number of keypoints.
                The last dimension corresponds to (x, y, visibility or score).
            assigned_colors (list[matplotlib.colors]): a list of colors, where each color
                corresponds to each mask or box in the image. Refer to 'matplotlib.colors'
                for full list of formats that the colors are accepted in.
        Returns:
            output (VisImage): image object with visualizations.
        """
        num_instances = 0
        if boxes is not None:
            boxes = self._convert_boxes(boxes)
            num_instances = len(boxes)
        if masks is not None:
            masks = self._convert_masks(masks)
            if num_instances:
                assert len(masks) == num_instances
            else:
                num_instances = len(masks)
        if keypoints is not None:
            if num_instances:
                assert len(keypoints) == num_instances
            else:
                num_instances = len(keypoints)
            keypoints = self._convert_keypoints(keypoints)
        if labels is not None:
            assert len(labels) == num_instances
        if assigned_colors is None:
            assigned_colors = [random_color(rgb=True, maximum=1) for _ in range(num_instances)]
        if num_instances == 0:
            return self.output
        if boxes is not None and boxes.shape[1] == 5:
            return self.overlay_rotated_instances(
                boxes=boxes, labels=labels, assigned_colors=assigned_colors
            )

        # Display in largest to smallest order to reduce occlusion.
        areas = None
        if boxes is not None:
            areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        elif masks is not None:
            areas = np.asarray([x.area() for x in masks])

        if areas is not None:
            sorted_idxs = np.argsort(-areas).tolist()
            # Re-order overlapped instances in descending order.
            boxes = boxes[sorted_idxs] if boxes is not None else None
            labels = [labels[k] for k in sorted_idxs] if labels is not None else None
            masks = [masks[idx] for idx in sorted_idxs] if masks is not None else None
            assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]
            keypoints = keypoints[sorted_idxs] if keypoints is not None else None

        for i in range(num_instances):
            color = assigned_colors[i]
            if boxes is not None:
                self.draw_box(boxes[i], edge_color=color)

            if masks is not None:
                for segment in masks[i].polygons:
                    self.draw_polygon(segment.reshape(-1, 2), color, alpha=alpha)

            if labels is not None:
                # first get a box
                if boxes is not None:
                    x0, y0, x1, y1 = boxes[i]
                    text_pos = (x0, y0)  # if drawing boxes, put text on the box corner.
                    horiz_align = "left"
                elif masks is not None:
                    # skip small mask without polygon
                    if len(masks[i].polygons) == 0:
                        continue

                    x0, y0, x1, y1 = masks[i].bbox()

                    # draw text in the center (defined by median) when box is not drawn
                    # median is less sensitive to outliers.
                    text_pos = np.median(masks[i].mask.nonzero(), axis=1)[::-1]
                    horiz_align = "center"
                else:
                    continue  # drawing the box confidence for keypoints isn't very useful.
                # for small objects, draw text at the side to avoid occlusion
                instance_area = (y1 - y0) * (x1 - x0)
                if (
                    instance_area < _SMALL_OBJECT_AREA_THRESH * self.output.scale
                    or y1 - y0 < 40 * self.output.scale
                ):
                    if y1 >= self.output.height - 5:
                        text_pos = (x1, y0)
                    else:
                        text_pos = (x0, y1)

                height_ratio = (y1 - y0) / np.sqrt(self.output.height * self.output.width)
                lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
                font_size = (
                    np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
                    * 0.5
                    * self._default_font_size
                )
                #TODO: fix and reimplement in segmentationsg
                lighter_color = tuple([min(x,1.0) for x in lighter_color])
                self.draw_text(
                    labels[i],
                    text_pos,
                    color=lighter_color,
                    horizontal_alignment=horiz_align,
                    font_size=font_size,
                )

        # draw keypoints
        if keypoints is not None:
            for keypoints_per_instance in keypoints:
                self.draw_and_connect_keypoints(keypoints_per_instance)

        return self.output

    def overlay_rotated_instances(self, boxes=None, labels=None, assigned_colors=None):
        """
        Args:
            boxes (ndarray): an Nx5 numpy array of
                (x_center, y_center, width, height, angle_degrees) format
                for the N objects in a single image.
            labels (list[str]): the text to be displayed for each instance.
            assigned_colors (list[matplotlib.colors]): a list of colors, where each color
                corresponds to each mask or box in the image. Refer to 'matplotlib.colors'
                for full list of formats that the colors are accepted in.

        Returns:
            output (VisImage): image object with visualizations.
        """
        num_instances = len(boxes)

        if assigned_colors is None:
            assigned_colors = [random_color(rgb=True, maximum=1) for _ in range(num_instances)]
        if num_instances == 0:
            return self.output

        # Display in largest to smallest order to reduce occlusion.
        if boxes is not None:
            areas = boxes[:, 2] * boxes[:, 3]

        sorted_idxs = np.argsort(-areas).tolist()
        # Re-order overlapped instances in descending order.
        boxes = boxes[sorted_idxs]
        labels = [labels[k] for k in sorted_idxs] if labels is not None else None
        colors = [assigned_colors[idx] for idx in sorted_idxs]

        for i in range(num_instances):
            self.draw_rotated_box_with_label(
                boxes[i], edge_color=colors[i], label=labels[i] if labels is not None else None
            )

        return self.output
    

def _create_numbered_text_labels(classes, scores, class_names, add_numbers = False):
    """
    Args:
        classes (list[int] or None):
        scores (list[float] or None):
        class_names (list[str] or None):

    Returns:
        list[str] or None
    """
    labels = None
    if classes is not None and class_names is not None and len(class_names) > 0:
        labels = [class_names[i] for i in classes]
    if scores is not None:
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
    if add_numbers is True:
        labels = [f"#{i} - " + s for i,s in enumerate(labels)]
    return labels
