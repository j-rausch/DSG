import contextlib
import copy
import io
from collections import defaultdict
import itertools
import json
import logging
import numpy as np
import os
import pickle
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate
from detectron2.utils.events import get_event_storage

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.evaluation.fast_eval_api import COCOeval_opt
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.logger import create_small_table
#from detectron2.evaluation import COCOEvaluator
from detectron2.evaluation.coco_evaluation import COCOEvaluator, _evaluate_predictions_on_coco

class COCOEvaluatorWeakSegmentation(COCOEvaluator):
    def _tasks_from_config(self, cfg):
        """
        Returns:
            tuple[str]: tasks that can be evaluated under the given configuration.
        """
        tasks = ("bbox",)
        # if cfg.MODEL.MASK_ON:
        #     tasks = tasks + ("segm",)
        # if cfg.MODEL.KEYPOINT_ON:
        #     tasks = tasks + ("keypoints",)
        return tasks


class COCOEvaluatorForExactMatching(COCOEvaluator):
    def __init__(
            self,
            dataset_name,
            tasks,
            distributed,
            output_dir=None,
            *,
            use_fast_impl=False, #use default COCOeval to access pred matches
            kpt_oks_sigmas=(),
    ):
        super(COCOEvaluatorForExactMatching, self).__init__(dataset_name,tasks,distributed,output_dir,use_fast_impl=use_fast_impl,kpt_oks_sigmas=kpt_oks_sigmas)
        self.coco_evals = dict()


    def evaluate(self, img_ids=None):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._results = OrderedDict()
        if "proposals" in predictions[0]:
            self._eval_box_proposals(predictions)
        if "instances" in predictions[0]:
            self._eval_predictions(set(self._tasks), predictions, img_ids=img_ids)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)


    def get_pred_to_gt_mapping(self):
        #  dtIds      - [1xD] id for each of the D detections (dt)
        #  gtIds      - [1xG] id for each of the G ground truths (gt)
        #  dtMatches  - [TxD] matching gt id at each IoU or 0
        #  gtMatches  - [TxG] matching dt id at each IoU or 0
        #  dtScores   - [1xD] confidence of each dt
        #  gtIgnore   - [1xG] ignore flag for each gt
        #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
        if len(self.coco_evals) == 0:
            return
        coco_eval = self.coco_evals['bbox']
        iou_thrs = coco_eval.params.iouThrs
        category_ids = coco_eval.params.catIds
        area_ranges = coco_eval.params.areaRng
        max_det_vals = coco_eval.params.maxDets
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        selected_area_range_filter = area_ranges[0]
        selected_max_det_filter = max_det_vals[-1]

        mapping_from_coco_to_instances_gt_per_img, mapping_from_coco_to_instances_dt_per_img = self.get_mappings_from_coco_ann_ids_to_instance_ids(coco_eval)
        pred_to_gt_maches_per_iou = dict()
        for iou_index, iou_thr in enumerate(iou_thrs):
            eval_imgs = coco_eval.evalImgs
            pred_to_gt_maches_per_img_for_all_categories = dict()
            for eval_img in eval_imgs:
                if eval_img is not None:
                    area_range = eval_img['aRng']
                    max_det = eval_img['maxDet']
                    category = eval_img['category_id']
                    if max_det != selected_max_det_filter or area_range != selected_area_range_filter:
                        continue
                    img_id = eval_img['image_id']
                    if img_id not in pred_to_gt_maches_per_img_for_all_categories:
                        pred_to_gt_maches_per_img_for_all_categories[img_id] = dict()

                    dt_matches = eval_img['dtMatches'][iou_index]
                    gt_matches = eval_img['gtMatches'][iou_index]
                    dt_ignore = eval_img['dtIgnore'][iou_index]
                    gt_ignore = eval_img['gtIgnore']
                    dt_ids = eval_img['dtIds']
                    for dt_index, gt_id in enumerate(dt_matches):
                        if dt_ignore[dt_index] is True:
                            continue
                        if int(gt_id) == 0:
                            continue  # a zero in the dtm list indicates that there was no matched gt for this prediction
                        if int(gt_id) == -1:
                            continue  # a -1 in the dtm list means that the detection match should be ignored
                        dt_id = int(dt_ids[dt_index])
                        if dt_id in pred_to_gt_maches_per_img_for_all_categories[img_id]:
                            raise AssertionError("detection id was already matched with a GT id!. only unique matches possible")
                        pred_to_gt_maches_per_img_for_all_categories[img_id][dt_id] = int(gt_id)

            pred_to_gt_maches_per_iou[iou_thr] = pred_to_gt_maches_per_img_for_all_categories

        for iou_thr, matches_per_img in pred_to_gt_maches_per_iou.items():
            for img_id, coco_pred_to_gt_matches in matches_per_img.items():
                instance_pred_to_gt_matches = dict()
                dt_mappings_for_cur_img =mapping_from_coco_to_instances_dt_per_img[img_id]
                gt_mappings_for_cur_img =mapping_from_coco_to_instances_gt_per_img[img_id]
                for coco_dt_id, coco_gt_id in coco_pred_to_gt_matches.items():
#                    instance_dt_id = mapping_from_coco_to_instances_dt_per_img[img_id][coco_dt_id]
#                    instance_gt_id = mapping_from_coco_to_instances_dt_per_img[img_id][coco_gt_id]
                    instance_dt_id = dt_mappings_for_cur_img[coco_dt_id]
                    instance_gt_id = gt_mappings_for_cur_img[coco_gt_id]
                    instance_pred_to_gt_matches[instance_dt_id] = instance_gt_id
                matches_per_img[img_id] = instance_pred_to_gt_matches

        return pred_to_gt_maches_per_iou

    def get_mappings_from_coco_ann_ids_to_instance_ids(self, coco_eval):
        coco_gt_ann_mappings = coco_eval.cocoGt.imgToAnns
        coco_dt_ann_mappings = coco_eval.cocoDt.imgToAnns
        mapping_from_coco_to_instances_dt_per_img = dict()
        mapping_from_coco_to_instances_gt_per_img = dict()
        img_ids = coco_eval.params.imgIds
        for img_id in img_ids:
            mapping_from_coco_to_instances_dt = dict()
            mapping_from_coco_to_instances_gt = dict()
            gt_anns_for_img = coco_gt_ann_mappings[img_id]
            dt_anns_for_img = coco_dt_ann_mappings[img_id]
            for gt_instance_index, coco_gt_ann in enumerate(gt_anns_for_img):
                coco_gt_ann_id = coco_gt_ann['id']
                mapping_from_coco_to_instances_gt[coco_gt_ann_id] = gt_instance_index
            for dt_instance_index, coco_dt_ann in enumerate(dt_anns_for_img):
                coco_dt_ann_id = coco_dt_ann['id']
                mapping_from_coco_to_instances_dt[coco_dt_ann_id] = dt_instance_index
            mapping_from_coco_to_instances_gt_per_img[img_id] = mapping_from_coco_to_instances_gt
            mapping_from_coco_to_instances_dt_per_img[img_id] = mapping_from_coco_to_instances_dt
        return mapping_from_coco_to_instances_gt_per_img, mapping_from_coco_to_instances_dt_per_img

    def _eval_predictions(self, tasks, predictions, img_ids=None):
            """
            Evaluate predictions on the given tasks.
            Fill self._results with the metrics of the tasks.
            """
            self._logger.info("Preparing results for COCO format ...")
            coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))

            # unmap the category ids for COCO
            if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
                reverse_id_mapping = {
                    v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
                }
                for result in coco_results:
                    category_id = result["category_id"]
                    assert (
                            category_id in reverse_id_mapping
                    ), "A prediction has category_id={}, which is not available in the dataset.".format(
                        category_id
                    )
                    result["category_id"] = reverse_id_mapping[category_id]

            try:
                storage = get_event_storage()
                current_iter_string = '_{}'.format(storage.iter)
            except AssertionError as e:
                current_iter_string = ''

            if self._output_dir:
                file_path = os.path.join(self._output_dir, "coco_instances_results{}.json".format(current_iter_string))
                self._logger.info("Saving results to {}".format(file_path))
                with PathManager.open(file_path, "w") as f:
                    f.write(json.dumps(coco_results))
                    f.flush()

            if not self._do_evaluation:
                self._logger.info("Annotations are not available for evaluation.")
                return

            self._logger.info(
                "Evaluating predictions with {} COCO API...".format(
                    "unofficial" if self._use_fast_impl else "official"
                )
            )
            for task in sorted(tasks):
                coco_eval = (
                    _evaluate_predictions_on_coco(
                        self._coco_api,
                        coco_results,
                        task,
                        kpt_oks_sigmas=self._kpt_oks_sigmas,
                        use_fast_impl=self._use_fast_impl,
                        img_ids=img_ids,
                    )
                    if len(coco_results) > 0
                    else None  # cocoapi does not handle empty results very well
                )

                res = self._derive_coco_results_with_per_iou_scores(
                    coco_eval, task, class_names=self._metadata.get("thing_classes")
                )
                self.coco_evals[task] = coco_eval
                self._results[task] = res

    def _derive_coco_results_with_per_iou_scores(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category_per_iou = dict()
        iou_thrs = coco_eval.params.iouThrs
        for iou_index, iou_thr in enumerate(iou_thrs):
            results_per_category = []
            for idx, name in enumerate(class_names):
                # area range index 0: all area ranges
                # max dets index -1: typically 100 per image
                precision = precisions[iou_index, :, idx, 0, -1]
                precision = precision[precision > -1]
                ap = np.mean(precision) if precision.size else float("nan")
                results_per_category.append(("{}".format(name), float(ap * 100)))
            results_per_category_per_iou[iou_thr] = results_per_category

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))
        results_per_category_per_iou['all'] = results_per_category

        # tabulate it
        for iou_key in [0.5, 'all']:
            N_COLS = min(6, len(results_per_category_per_iou[iou_key]) * 2)
            results_flatten = list(itertools.chain(*results_per_category_per_iou[iou_key]))
            results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
            table = tabulate(
                results_2d,
                tablefmt="pipe",
                floatfmt=".3f",
                headers=["category", "AP"] * (N_COLS // 2),
                numalign="left",
            )
            self._logger.info("Per-category {} AP @ IoU[{}]: \n".format(iou_type, iou_key) + table)

        all_ious = list(results_per_category_per_iou.keys())
        for iou in all_ious: # [0.5, 0.65, 0.8, 'all']:
            results.update({"AP(@{}-".format(iou) + name: ap for name, ap in results_per_category_per_iou[iou]})
        return results


    def get_gt_category_id_to_label_mapping(self):
        class_names = self._metadata.get("thing_classes")
        id_to_name_mapping = {i:class_name for i,class_name in enumerate(class_names)}
        return id_to_name_mapping


