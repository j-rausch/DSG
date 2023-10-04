import os
import torch
import logging
import detectron2.utils.comm as comm
import time 
import datetime
from collections import OrderedDict
from detectron2.utils.logger import log_every_n_seconds
from detectron2.engine import DefaultTrainer
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
from ..evaluation import COCOEvaluatorWeakSegmentation, scenegraph_inference_on_dataset, SceneGraphEvaluator
from detectron2.engine import hooks
from detectron2.data.samplers import InferenceSampler, RepeatFactorTrainingSampler, TrainingSampler
from detectron2.data.common import MapDataset, DatasetFromList
from detectron2.data.build import trivial_batch_collator
from detectron2.utils.comm import get_world_size


class SceneGraphTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super(SceneGraphTrainer, self).__init__(cfg)

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
            evaluator = SceneGraphEvaluator(dataset_name, cfg, True, output_folder)
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

class SceneGraphSegmentationTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super(SceneGraphSegmentationTrainer, self).__init__(cfg)
        self.mask_train_loader = iter(self.build_mask_loader(cfg, is_train=True))

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=SceneGraphDatasetMapper(cfg, True))

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=SceneGraphDatasetMapper(cfg, False))

    @classmethod
    def build_mask_loader(cls, cfg, is_train=True):
        dataset_name = cfg.DATASETS.MASK_TRAIN if is_train else cfg.DATASETS.MASK_TEST
        if is_train:
            dataset = get_detection_dataset_dicts(
                dataset_name,
                filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
                min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
                if cfg.MODEL.KEYPOINT_ON
                else 0,
                proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
            )
            sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
            if sampler_name == "TrainingSampler":
                sampler = TrainingSampler(len(dataset))
            elif sampler_name == "RepeatFactorTrainingSampler":
                repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                    dataset, cfg.DATALOADER.REPEAT_THRESHOLD
                )
                sampler = RepeatFactorTrainingSampler(repeat_factors)
            else:
                raise ValueError("Unknown training sampler: {}".format(sampler_name))
            if isinstance(dataset, list):
                dataset = DatasetFromList(dataset, copy=False)
            mapper = SceneGraphDatasetMapper(cfg, is_train)
            dataset = MapDataset(dataset, mapper)
            if sampler is None:
                sampler = TrainingSampler(len(dataset))
            assert isinstance(sampler, torch.utils.data.sampler.Sampler)
            return build_batch_data_loader(
                    dataset,
                    sampler,
                    cfg.SOLVER.IMS_PER_BATCH//cfg.DATASETS.SEG_DATA_DIVISOR,
                    aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
                    num_workers=cfg.DATALOADER.NUM_WORKERS,
                )
        else:
            dataset = get_detection_dataset_dicts(
                dataset_name,
                filter_empty=False,
                proposal_files=[
                    cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(dataset_name)]
                ]
                if cfg.MODEL.LOAD_PROPOSALS
                else None,
            )
            mapper = SceneGraphDatasetMapper(cfg, is_train)
            if isinstance(dataset, list):
                dataset = DatasetFromList(dataset, copy=False)
            dataset = MapDataset(dataset, mapper)
            sampler = InferenceSampler(len(dataset))
            batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)
            data_loader = torch.utils.data.DataLoader(
                dataset,
                num_workers=cfg.DATALOADER.NUM_WORKERS,
                batch_sampler=batch_sampler,
                collate_fn=trivial_batch_collator,
            )
            return data_loader

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
        # if comm.is_main_process():
        #     ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_to_keep=1))
        
        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results
        ret.append(PeriodicCheckpointerWithEval(cfg.TEST.EVAL_PERIOD, test_and_save_results, self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_to_keep=100))
        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        # ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._trainer._data_loader_iter)
        mask_data = next(self.mask_train_loader)
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.model(data, mask_batched_inputs=mask_data)
        losses = sum(loss_dict.values())

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        losses.backward()

        # for name, param in self.model.named_parameters():
        #     try:
        #         print (name, param.grad.norm())
        #     except:
        #         print (name, param.grad)
        # import ipdb; ipdb.set_trace()
        # use a new stream so the ops don't wait for DDP
        with torch.cuda.stream(
            torch.cuda.Stream()
        ) if losses.device.type == "cuda" else _nullcontext():
            self._trainer._write_metrics(loss_dict, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()
    
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
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            if 'coco' in dataset_name:
                data_loader = cls.build_mask_loader(cfg, is_train=False)
                evaluator = cls.build_evaluator(cfg, dataset_name)
                results_i = inference_on_dataset_with_segmentation(model, data_loader, evaluator, segmentation_step=True)
            else:
                data_loader = cls.build_test_loader(cfg, dataset_name)
                evaluator = SceneGraphEvaluator(dataset_name, cfg, True, output_folder)
                results_i = scenegraph_inference_on_dataset(cfg, model, data_loader, evaluator)
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

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        evaluator_list = []
        if 'coco' in dataset_name:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_dir=cfg.OUTPUT_DIR))
        else:
            evaluator_list.append(COCOEvaluatorWeakSegmentation(dataset_name, cfg, True, output_dir=cfg.OUTPUT_DIR))
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

class TestTrainer(SceneGraphTrainer):
    def run_step(self):
            """
            Implement the standard training logic described above.
            """
            assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
            start = time.perf_counter()
            """
            If you want to do something with the data, you can wrap the dataloader.
            """
            data = next(self._trainer._data_loader_iter)
            data_time = time.perf_counter() - start

            """
            If you want to do something with the losses, you can wrap the model.
            """
            import ipdb; ipdb.set_trace()
            loss_dict = self.model(data)
            losses = sum(loss_dict.values())

            """
            If you need to accumulate gradients or do something similar, you can
            wrap the optimizer with your custom `zero_grad()` method.
            """
            self.optimizer.zero_grad()
            losses.backward()
            for name, param in self.model.named_parameters():
                try:
                    print (name, param.grad.norm())
                except:
                    print (name, param.grad)
            import ipdb; ipdb.set_trace()
            # for name, param in self.model.named_parameters():
            #     try:
            #         print (name, param.grad.norm())
            #     except:
            #         print (name, param.grad)
            # import ipdb; ipdb.set_trace()
            # use a new stream so the ops don't wait for DDP
            with torch.cuda.stream(
                torch.cuda.Stream()
            ) if losses.device.type == "cuda" else _nullcontext():
                self._trainer._write_metrics(loss_dict, data_time)

            """
            If you need gradient clipping/scaling or other processing, you can
            wrap the optimizer with your custom `step()` method. But it is
            suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
            """
            self.optimizer.step()

def inference_on_dataset_with_segmentation(model, data_loader, evaluator, segmentation_step=False):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.forward` accurately.
    The model will be used in eval mode.
    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use `None` if you only want
            to benchmark, but don't want to do any evaluation.
    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs, segmentation_step=segmentation_step)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            evaluator.process(inputs, outputs)

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results