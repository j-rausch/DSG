import sys
import os
import numpy as np
import torch
import random

#for reproducibility
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

from pathlib import Path

import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger
from detectron2.engine import default_setup, launch #default_argument_parser
from segmentationsg.engine.defaults import sg_argument_parser
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from detectron2.evaluation.testing import flatten_results_dict


from detectron2.utils.events import EventStorage, get_event_storage

from segmentationsg.engine import DocSceneGraphTrainer
from segmentationsg.data import add_dataset_config, register_datasets
from segmentationsg.modeling.roi_heads.scenegraph_head import add_scenegraph_config
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from segmentationsg.modeling import *

parser = sg_argument_parser()


def setup(args):
    cfg = get_cfg()
    add_dataset_config(cfg)
    add_scenegraph_config(cfg)
    assert(cfg.MODEL.ROI_SCENEGRAPH_HEAD.MODE in ['predcls', 'sgls', 'sgdet']) , "Mode {} not supported".format(cfg.MODEL.ROI_SCENEGRaGraph.MODE)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    register_datasets(cfg)
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="LSDA")
    return cfg

def main(args):
    cfg = setup(args)
    if args.eval_only is True or args.eval_only_all_checkpoints is True:
        if args.eval_only_all_checkpoints is True:
            with torch.no_grad():
                #trainer = DocSceneGraphTrainer(cfg)
                model = DocSceneGraphTrainer.build_model(cfg)
                model.eval()
                weights_path_debug = cfg.MODEL.WEIGHTS
                if not os.path.isdir(weights_path_debug):
                    weights_dir = Path(weights_path_debug).parent.absolute()
                else:
                    weights_dir = cfg.MODEL.WEIGHTS
                checkpointer = DetectionCheckpointer(model, save_dir=weights_dir)
                all_weights = checkpointer.get_all_checkpoint_files()
                sorted_weights = sorted([x for x in all_weights if not 'model_final' in x  and 'model' in x], key = lambda x: int(x.split('model_')[-1].split('.pth')[0]))
                with EventStorage(0) as storage:
                    # metric_printer = CommonMetricPrinter(cfg.SOLVER.MAX_ITER)
                    tb_writer = TensorboardXWriter(cfg.OUTPUT_DIR)
                    #json_writer = JSONWriter(os.path.join(cfg.OUTPUT_DIR, "debug_metrics_{}.json".format(weight_iter)))
                    json_writer = JSONWriter(os.path.join(cfg.OUTPUT_DIR, "debug_metrics.json"))
                    for weight_path in sorted_weights:
                        weight_iter = int(weight_path.split('model_')[-1].split('.pth')[0])
                        storage._iter = weight_iter
                        storage.iter = weight_iter
                        #for sorted_weight in sorted_weights:

    #                    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
    #                        cfg.MODEL.WEIGHTS, resume=args.resume
    #                    )
                        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                            weight_path, resume=args.resume
                        )

                        res = DocSceneGraphTrainer.test(cfg, model)
                        #this is basically replicating the behaviour of EvalHook
                        flattened_results = flatten_results_dict(res)
                        for k, v in flattened_results.items():
                            try:
                                v = float(v)
                            except Exception as e:
                                raise ValueError(
                                    "[EvalHook] eval_function should return a nested dict of float. "
                                    "Got '{}: {}' instead.".format(k, v)
                                ) from e
                        storage.put_scalars(**flattened_results, smoothing_hint=False)
                        #TODO: use event_storage inside writers to add checkpoint iteration number to outputs
                        tb_writer.write()
                        json_writer.write()

                return
        elif args.eval_only is True and args.eval_only_all_checkpoints is False:
            with torch.no_grad():
                model = DocSceneGraphTrainer.build_model(cfg)
                model.eval()
                DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                    cfg.MODEL.WEIGHTS, resume=args.resume
                )
                res = DocSceneGraphTrainer.test(cfg, model)
                # if comm.is_main_process():
                #     verify_results(cfg, res)
                return res

    trainer = DocSceneGraphTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

if __name__ == '__main__':
    args = parser.parse_args()
    try:
        # use the last 4 numbers in the job id as the id
        default_port = os.environ['LSB_JOBID']
        default_port = default_port[-4:]

        # all ports should be in the 10k+ range
        default_port = int(default_port) + 15000

    except Exception:
        default_port = 59482 + random.randint(1,500)

    
    args.dist_url = 'tcp://127.0.0.1:'+str(default_port)
    print (args)
    
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
