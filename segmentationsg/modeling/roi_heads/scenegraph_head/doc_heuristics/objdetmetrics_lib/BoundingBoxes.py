import logging
import os

import glob

from segmentationsg.modeling.roi_heads.scenegraph_head.doc_heuristics.objdetmetrics_lib.BoundingBox import *
from segmentationsg.modeling.roi_heads.scenegraph_head.doc_heuristics.objdetmetrics_lib.utils import *

logger = logging.getLogger(__name__)


class BoundingBoxes:
    def __init__(self):
        self._boundingBoxes = []

    def addBoundingBox(self, bb):
        self._boundingBoxes.append(bb)

    def removeBoundingBox(self, _boundingBox):
        for d in self._boundingBoxes:
            if BoundingBox.compare(d, _boundingBox):
                del self._boundingBoxes[d]
                return

    def removeAllBoundingBoxes(self):
        self._boundingBoxes = []

    def getBoundingBoxes(self):
        return self._boundingBoxes

    def getBoundingBoxImageNames(self):
        return set(d.getImageName() for d in self._boundingBoxes)

    def getBoundingBoxByClass(self, classId):
        boundingBoxes = []
        for d in self._boundingBoxes:
            if d.getClassId() == classId:  # get only specified bounding box type
                boundingBoxes.append(d)
        return boundingBoxes

    def getClasses(self):
        classes = []
        for d in self._boundingBoxes:
            c = d.getClassId()
            if c not in classes:
                classes.append(c)
        return classes

    def getBoundingBoxesByType(self, bbType):
        # get only specified bb type
        return [d for d in self._boundingBoxes if d.getBBType() == bbType]

    def getBoundingBoxesByImageName(self, imageName):
        # get only specified bb type
        return [d for d in self._boundingBoxes if d.getImageName() == imageName]

    def count(self, bbType=None):
        if bbType is None:  # Return all bounding boxes
            return len(self._boundingBoxes)
        count = 0
        for d in self._boundingBoxes:
            if d.getBBType() == bbType:  # get only specified bb type
                count += 1
        return count

    def clone(self):
        newBoundingBoxes = BoundingBoxes()
        for d in self._boundingBoxes:
            det = BoundingBox.clone(d)
            newBoundingBoxes.addBoundingBox(det)
        return newBoundingBoxes

    def drawAllBoundingBoxes(self, image, imageName):
        bbxes = self.getBoundingBoxesByImageName(imageName)
        for bb in bbxes:
            if bb.getBBType() == BBType.GroundTruth:  # if ground truth
                image = add_bb_into_image(image, bb, color=(0, 255, 0), label=bb.getClassId())  # green
            else:  # if detection
                image = add_bb_into_image(image, bb, color=(255, 0, 0), label=bb.getClassId())  # red
        return image


def getBoundingBoxes(directory,
                     bbFormat,
                     coordType,
                     isGT=False,
                     allBoundingBoxes=None,
                     allClasses=None,
                     imgSize=(0, 0),
                     header=True):
    """Read txt files containing bounding boxes (ground truth and detections)."""
    if allBoundingBoxes is None:
        allBoundingBoxes = BoundingBoxes()
    if allClasses is None:
        allClasses = []
    # Read ground truths
    os.chdir(directory)
    files = glob.glob("*.txt")
    files.sort()
    # Read GT detections from txt file
    # Each line of the files in the groundtruths folder represents a ground truth bounding box
    # (bounding boxes that a detector should detect)
    # Each value of each line is  "class_id, x, y, width, height" respectively
    # Class_id represents the class of the bounding box
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    for f in files:
        nameOfImage = f.replace(".txt", "")
        fh1 = open(f, "r")

        if header is True:
            header_line = fh1.readline()
            header_split = header_line.split(';')
            assert len(header_split) > 0
            try:
                height = header_split[0].split(':')[-1]
                width = header_split[1].split(':')[-1]
            except IndexError as e:
                logger.error('error for header split of {} in {}: {}'.format(f, directory, header_split))
                raise
            imgSize = (width, height)

        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            if isGT:
                # idClass = int(splitLine[0]) #class
                ann_id = int(splitLine[0])
                idClass = (splitLine[1])  # class
                x = float(splitLine[2])
                y = float(splitLine[3])
                w = float(splitLine[4])
                h = float(splitLine[5])
                bb = BoundingBox(
                    nameOfImage,
                    idClass,
                    x,
                    y,
                    w,
                    h,
                    coordType,
                    imgSize,
                    BBType.GroundTruth,
                    format=bbFormat,
                    bbox_id=ann_id)
            else:
                # idClass = int(splitLine[0]) #class
                ann_id = int(splitLine[0])
                idClass = (splitLine[1])  # class
                confidence = float(splitLine[2])
                x = float(splitLine[3])
                y = float(splitLine[4])
                w = float(splitLine[5])
                h = float(splitLine[6])
                bb = BoundingBox(
                    nameOfImage,
                    idClass,
                    x,
                    y,
                    w,
                    h,
                    coordType,
                    imgSize,
                    BBType.Detected,
                    confidence,
                    format=bbFormat,
                    bbox_id=ann_id)
            allBoundingBoxes.addBoundingBox(bb)
            if idClass not in allClasses:
                allClasses.append(idClass)
        fh1.close()
    return allBoundingBoxes, allClasses


def getBoundingBoxesForFile(filepath,
                            isGT,
                            bbFormat,
                            coordType,
                            allBoundingBoxes=None,
                            allClasses=None,
                            imgSize=(0, 0),
                            header=True):
    """Read txt files containing bounding boxes (ground truth and detections)."""
    if allBoundingBoxes is None:
        allBoundingBoxes = BoundingBoxes()
    if allClasses is None:
        allClasses = []
    nameOfImage = filepath.replace(".txt", "")
    fh1 = open(filepath, "r")
    if header is True:
        header_line = fh1.readline()
        header_split = header_line.split(';')
        assert len(header_split) > 0
        height = header_split[0].split(':')[-1]
        width = header_split[1].split(':')[-1]
        if height is None or width is None:
            logger.warning("Height or width None!")
        else:
            imgSize = (width, height)

    for line in fh1:
        line = line.replace("\n", "")
        if line.replace(' ', '') == '':
            continue

        splitLine = line.split(" ")
        if isGT:
            # idClass = int(splitLine[0]) #class
            ann_id = int(splitLine[0])
            idClass = (splitLine[1])  # class
            x = float(splitLine[2])
            y = float(splitLine[3])
            w = float(splitLine[4])
            h = float(splitLine[5])
            bb = BoundingBox(
                nameOfImage,
                idClass,
                x,
                y,
                w,
                h,
                coordType,
                imgSize,
                BBType.GroundTruth,
                format=bbFormat,
                bbox_id=ann_id)
        else:
            # idClass = int(splitLine[0]) #class
            ann_id = int(splitLine[0])
            idClass = (splitLine[1])  # class
            confidence = float(splitLine[2])
            x = float(splitLine[3])
            y = float(splitLine[4])
            w = float(splitLine[5])
            h = float(splitLine[6])
            bb = BoundingBox(
                nameOfImage,
                idClass,
                x,
                y,
                w,
                h,
                coordType,
                imgSize,
                BBType.Detected,
                confidence,
                format=bbFormat,
                bbox_id=ann_id)
        allBoundingBoxes.addBoundingBox(bb)
        if idClass not in allClasses:
            allClasses.append(idClass)
    fh1.close()
    return allBoundingBoxes, allClasses


def getBoundingBoxesFromDetectron2(detectron2_instances_for_imgs,
                            isGT,
                            coordType,
                            allBoundingBoxes=None,
                            allClasses=None,
                            imgSize=(0, 0),
                            class_mapping=None,
                            first_img_id=None ):
    """Read txt files containing bounding boxes (ground truth and detections)."""
    assert class_mapping is not None
    if allBoundingBoxes is None:
        allBoundingBoxes = BoundingBoxes()


    if allClasses is None:
        allClasses = []

    #print('number of imgs: {}'.format(len(detectron2_instances_for_imgs)))
    ann_id_counter = 0
    cur_img_id = first_img_id if first_img_id is not None else -1
    for detectron2_instances in detectron2_instances_for_imgs:
        pred_boxes = detectron2_instances.pred_boxes
        pred_scores = detectron2_instances.scores.cpu().numpy().tolist()
        pred_classes = detectron2_instances.pred_classes.cpu().numpy().tolist()
        img_size = detectron2_instances.image_size
        height = img_size[0]
        width = img_size[1]
        wh = pred_boxes.tensor[:, 2:] - pred_boxes.tensor[:, :2] + 1.0
        xy = pred_boxes.tensor[:, :2] + 0.5 * wh
        w, h = wh.split([1, 1], dim=-1)
        x, y = xy.split([1, 1], dim=-1)
        x1, y1, x2, y2 = pred_boxes.tensor.split([1, 1, 1, 1], dim=-1)
        w = w.cpu().numpy()
        h = h.cpu().numpy()
        x1_list = x1.squeeze().cpu().numpy().tolist()
        y1_list = y1.squeeze().cpu().numpy().tolist()
        x2_list = x2.squeeze().cpu().numpy().tolist()
        y2_list = y2.squeeze().cpu().numpy().tolist()

        imgSize = (width, height)
        for i in range(len(pred_boxes)):
            if isGT:
                confidence = BBType.GroundTruth,
            else:
                confidence = float(pred_scores[i])
            box_x1 = x1_list[i]
            box_y1 = y1_list[i]
            box_x2 = x2_list[i]
            box_y2 = y2_list[i]

            ann_id = ann_id_counter
            ann_id_counter += 1
            idClass = pred_classes[i]
            class_name = class_mapping[idClass]

            bb = BoundingBox(
                         imageName=cur_img_id,
                         classId=class_name,
                         x=box_x1,
                         y=box_y1,
                         w=box_x2,
                         h=box_y2,
                         typeCoordinates=coordType,
                         imgSize=imgSize,
                         bbType=BBType.Detected,
                         classConfidence=confidence,
                         format=BBFormat.XYX2Y2,
                         bbox_id=ann_id,
                         column=None)
#            bb = BoundingBox(
#                cur_img_id,
#                class_name,
#                box_x1,
#                box_y1,
#                box_x2,
#                box_y2,
#                coordType,
#                imgSize,
#                confidence,
#                format=BBFormat.XYX2Y2,
#                bbox_id=ann_id)
            allBoundingBoxes.addBoundingBox(bb)
            if idClass not in allClasses:
                allClasses.append(idClass)
        cur_img_id += 1
    return allBoundingBoxes, allClasses
