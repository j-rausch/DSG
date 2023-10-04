CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train_doc_SG_head.py \
  --config-file ./configs/sgg_end2end_EP.yaml \
  --num-gpus 4  \
  --resume \
  MODEL.ROI_SCENEGRAPH_HEAD.PREDICT_USE_VISION  True \
  OUTPUT_DIR ./data/checkpoints/03_213_sgg_end2end_EP_WSFT_unionfeat \
  MODEL.WEIGHTS ./data/checkpoints/03_012_sgg_end2end_ADwk_unionfeat/model_0199999.pth