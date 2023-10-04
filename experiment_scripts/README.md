To train a model from scratch, download a model checkpoint of your choice from the detectron2 model zoo and edit the MODEL.WEIGHTS inside the training script according to its path.
For instance, a [pretrained detectron2 model](https://dl.fbaipublicfiles.com/detectron2/new_baselines/mask_rcnn_R_50_FPN_400ep_LSJ/42019571/model_final_14d201.pkl)
E.g. `MODEL.WEIGHTS ./data/checkpoints/model_final_14d201.pkl`
