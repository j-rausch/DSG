import os
import torch
import logging
import detectron2.utils.comm as comm
import time 
import datetime
from collections import OrderedDict
from detectron2.utils.logger import log_every_n_seconds
from detectron2.engine import DefaultTrainer
from segmentationsg.engine.defaults import ModifiedDefaultTrainer, get_bn_modules
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
    build_batch_data_loader
)
from detectron2.evaluation import DatasetEvaluators, DatasetEvaluator, print_csv_format, inference_context

from detectron2.engine import HookBase
from segmentationsg.data import SceneGraphDatasetMapper
from detectron2.evaluation import (
    COCOEvaluator
)
from ..checkpoint import PeriodicCheckpointerWithEval
from segmentationsg.evaluation import COCOEvaluatorWeakSegmentation, scenegraph_inference_on_dataset, DocSceneGraphEvaluator
from detectron2.engine import hooks
from detectron2.data.samplers import InferenceSampler, RepeatFactorTrainingSampler, TrainingSampler
from detectron2.data.common import MapDataset, DatasetFromList
from detectron2.data.build import trivial_batch_collator
from detectron2.utils.comm import get_world_size


class DocSceneGraphTrainer(ModifiedDefaultTrainer):
    def __init__(self, cfg):
        super(DocSceneGraphTrainer, self).__init__(cfg)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=SceneGraphDatasetMapper(cfg, True))

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=SceneGraphDatasetMapper(cfg, False))

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_to_keep=100))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.
        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)

        
        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # import ipdb; ipdb.set_trace()
            
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            # detection_evaluator = COCOEvaluator(dataset_name, cfg, True, output_folder)

            #eval_metrics = ('SGRecall', 'SGNoGraphConstraintRecall', 'SGPairAccuracy', 'SGMeanRecall', 'SGExactMatches')
            eval_metrics = set(['SGRecall', 'SGExactMatches'])
            evaluator = DocSceneGraphEvaluator(dataset_name, cfg, True, output_folder, metrics=eval_metrics)
            results_i = scenegraph_inference_on_dataset(cfg, model, data_loader, evaluator)



            # print("Out of sg inference")
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)
        comm.synchronize()
        if len(results) == 1:
            results = list(results.values())[0]
        return results
