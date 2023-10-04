import os
from detectron2.config import CfgNode as CN

def add_dataset_config(cfg):
  _C = cfg

  _C.MODEL.ROI_HEADS.NUM_OUTPUT_CLASSES = 80
  _C.MODEL.ROI_HEADS.EMBEDDINGS_PATH = ""
  _C.MODEL.ROI_HEADS.EMBEDDINGS_PATH_COCO = ""
  _C.MODEL.ROI_HEADS.LINGUAL_MATRIX_THRESHOLD = 0.05
  _C.MODEL.ROI_HEADS.MASK_NUM_CLASSES = 80

  _C.MODEL.FREEZE_LAYERS = CN()
  _C.MODEL.FREEZE_LAYERS.META_ARCH = []
  _C.MODEL.FREEZE_LAYERS.ROI_HEADS = []

  _C.DATASETS.TYPE = ""
  _C.DATASETS.VISUAL_GENOME = CN()
  _C.DATASETS.VISUAL_GENOME.IMAGES = './datasets/vg/VG_100K/'
  _C.DATASETS.VISUAL_GENOME.MAPPING_DICTIONARY = './datasets/vg/orig_and_generated_files/VG-SGG-dicts-with-attri.json'
  _C.DATASETS.VISUAL_GENOME.IMAGE_DATA = './datasets/vg/orig_and_generated_files/image_data.json'
  _C.DATASETS.VISUAL_GENOME.VG_ATTRIBUTE_H5 = './datasets/vg/orig_and_generated_files/VG-SGG-with-attri.h5'
  _C.DATASETS.VISUAL_GENOME.TRAIN_MASKS = ""
  _C.DATASETS.VISUAL_GENOME.TEST_MASKS = ""
  _C.DATASETS.VISUAL_GENOME.VAL_MASKS = ""
  _C.DATASETS.VISUAL_GENOME.CLIPPED = False

  _C.DATASETS.ARXIVDOCS_TARGET = CN()
  _C.DATASETS.ARXIVDOCS_TARGET.CLIPPED = False
  _C.DATASETS.ARXIVDOCS_TARGET.FILTER_EMPTY_RELATIONS = True
  _C.DATASETS.ARXIVDOCS_TARGET.FILTER_DUPLICATE_RELATIONS = True
  _C.DATASETS.ARXIVDOCS_TARGET.FILTER_NON_OVERLAP = False
  _C.DATASETS.ARXIVDOCS_TARGET.BOX_SCALE = 1024
  _C.DATASETS.ARXIVDOCS_TARGET.TRAIN_MASKS = "./datasets/ADtgt_VGv2/additional_processed_anns/train/attribute_files/arxivdocs_target_layout_v2_train_scene_graph_segmentations.json"
  _C.DATASETS.ARXIVDOCS_TARGET.TEST_MASKS = "./datasets/ADtgt_VGv2/additional_processed_anns/test/attribute_files/arxivdocs_target_layout_v2_test_scene_graph_segmentations.json"
  _C.DATASETS.ARXIVDOCS_TARGET.VAL_MASKS = "./datasets/ADtgt_VGv2/additional_processed_anns/dev/attribute_files/arxivdocs_target_layout_v2_dev_scene_graph_segmentations.json"

  _C.DATASETS.ARXIVDOCS_TARGET.TRAIN_IMAGES = "./datasets/ADtgt_VGv2/imgs/train"
  _C.DATASETS.ARXIVDOCS_TARGET.TRAIN_MAPPING_DICTIONARY = "./datasets/ADtgt_VGv2/additional_processed_anns/train/attribute_files/arxivdocs_target_layout_v2_train_scene_graph_dicts_with_attri.json"
  _C.DATASETS.ARXIVDOCS_TARGET.TRAIN_IMAGE_DATA = "./datasets/ADtgt_VGv2/anns/train/arxivdocs_target_layout_v2_train_scene_graph_image_data.json"
  _C.DATASETS.ARXIVDOCS_TARGET.TRAIN_ARXIVDOCS_TARGET_ATTRIBUTE_H5 = "./datasets/ADtgt_VGv2/additional_processed_anns/train/attribute_files/arxivdocs_target_layout_v2_train_scene_graph_with_attri.h5"

  _C.DATASETS.ARXIVDOCS_TARGET.VAL_IMAGES = "./datasets/ADtgt_VGv2/imgs/dev"
  _C.DATASETS.ARXIVDOCS_TARGET.VAL_MAPPING_DICTIONARY = "./datasets/ADtgt_VGv2/additional_processed_anns/dev/attribute_files/arxivdocs_target_layout_v2_dev_scene_graph_dicts_with_attri.json"
  _C.DATASETS.ARXIVDOCS_TARGET.VAL_IMAGE_DATA = "./datasets/ADtgt_VGv2/anns/dev/arxivdocs_target_layout_v2_dev_scene_graph_image_data.json"
  _C.DATASETS.ARXIVDOCS_TARGET.VAL_ARXIVDOCS_TARGET_ATTRIBUTE_H5 = "./datasets/ADtgt_VGv2/additional_processed_anns/dev/attribute_files/arxivdocs_target_layout_v2_dev_scene_graph_with_attri.h5"


  _C.DATASETS.ARXIVDOCS_TARGET.TEST_IMAGES = "./datasets/ADtgt_VGv2/imgs/test"
  _C.DATASETS.ARXIVDOCS_TARGET.TEST_MAPPING_DICTIONARY = "./datasets/ADtgt_VGv2/additional_processed_anns/test/attribute_files/arxivdocs_target_layout_v2_test_scene_graph_dicts_with_attri.json"
  _C.DATASETS.ARXIVDOCS_TARGET.TEST_IMAGE_DATA = "./datasets/ADtgt_VGv2/anns/test/arxivdocs_target_layout_v2_test_scene_graph_image_data.json"
  _C.DATASETS.ARXIVDOCS_TARGET.TEST_ARXIVDOCS_TARGET_ATTRIBUTE_H5 = "./datasets/ADtgt_VGv2/additional_processed_anns/test/attribute_files/arxivdocs_target_layout_v2_test_scene_graph_with_attri.h5"

  _C.DATASETS.ARXIVDOCS_WEAK = CN()
  _C.DATASETS.ARXIVDOCS_WEAK.CLIPPED = False
  _C.DATASETS.ARXIVDOCS_WEAK.FILTER_EMPTY_RELATIONS = True
  _C.DATASETS.ARXIVDOCS_WEAK.FILTER_DUPLICATE_RELATIONS = True
  _C.DATASETS.ARXIVDOCS_WEAK.FILTER_NON_OVERLAP = False
  _C.DATASETS.ARXIVDOCS_WEAK.BOX_SCALE = 1024

  _C.DATASETS.ARXIVDOCS_WEAK.TRAIN_MASKS = "./datasets/ADwk_VGv2/additional_processed_anns/train/attribute_files/arxivdocs_weak_layout_v2_train_scene_graph_segmentations.json"
  _C.DATASETS.ARXIVDOCS_WEAK.TEST_MASKS = "./datasets/ADwk_VGv2/additional_processed_anns/test/attribute_files/arxivdocs_weak_layout_v2_test_scene_graph_segmentations.json"
  _C.DATASETS.ARXIVDOCS_WEAK.VAL_MASKS = "./datasets/ADwk_VGv2/additional_processed_anns/dev/attribute_files/arxivdocs_weak_layout_v2_dev_scene_graph_segmentations.json"


  _C.DATASETS.ARXIVDOCS_WEAK.TRAIN_IMAGES = "./datasets/ADwk_VGv2/imgs/train"
  _C.DATASETS.ARXIVDOCS_WEAK.TRAIN_MAPPING_DICTIONARY = "./datasets/ADwk_VGv2/additional_processed_anns/train/attribute_files/arxivdocs_weak_layout_v2_train_scene_graph_dicts_with_attri.json"
  _C.DATASETS.ARXIVDOCS_WEAK.TRAIN_IMAGE_DATA = "./datasets/ADwk_VGv2/anns/train/arxivdocs_weak_layout_v2_train_scene_graph_image_data.json"
  _C.DATASETS.ARXIVDOCS_WEAK.TRAIN_ARXIVDOCS_WEAK_ATTRIBUTE_H5 = "./datasets/ADwk_VGv2/additional_processed_anns/train/attribute_files/arxivdocs_weak_layout_v2_train_scene_graph_with_attri.h5"


  _C.DATASETS.EPERIODICA = CN()
  _C.DATASETS.EPERIODICA.CLIPPED = False
  _C.DATASETS.EPERIODICA.FILTER_EMPTY_RELATIONS = True
  _C.DATASETS.EPERIODICA.FILTER_DUPLICATE_RELATIONS = True
  _C.DATASETS.EPERIODICA.FILTER_NON_OVERLAP = False
  _C.DATASETS.EPERIODICA.BOX_SCALE = 1024
  _C.DATASETS.EPERIODICA.TRAIN_MASKS = ''
  _C.DATASETS.EPERIODICA.VAL_MASKS = ''
  _C.DATASETS.EPERIODICA.TEST_MASKS = ''

  _C.DATASETS.EPERIODICA.TRAIN_MASKS = "./datasets/eperiodica_v3/additional_processed_anns/train/eperiodica_minitrain/attribute_files/eperiodica_minitrain_VG_scene_graph_segmentations.json"
  _C.DATASETS.EPERIODICA.VAL_MASKS = "./datasets/eperiodica_v3/additional_processed_anns/val/eperiodica_minival/attribute_files/eperiodica_minival_VG_scene_graph_segmentations.json"
  _C.DATASETS.EPERIODICA.TEST_MASKS = "./datasets/eperiodica_v3/additional_processed_anns/test/eperiodica_minitest/attribute_files/eperiodica_minitest_VG_scene_graph_segmentations.json"

  _C.DATASETS.EPERIODICA.TRAIN_IMAGES = "./datasets/eperiodica_v3/imgs/train"
  _C.DATASETS.EPERIODICA.TRAIN_MAPPING_DICTIONARY = "./datasets/eperiodica_v3/additional_processed_anns/train/eperiodica_minitrain/attribute_files/eperiodica_minitrain_VG_scene_graph_dicts_with_attri.json"
  _C.DATASETS.EPERIODICA.TRAIN_IMAGE_DATA = "./datasets/eperiodica_v3/anns/train/eperiodica_minitrain/eperiodica_minitrain_VG_scene_graph_image_data.json"
  _C.DATASETS.EPERIODICA.TRAIN_EPERIODICA_TARGET_ATTRIBUTE_H5 = "./datasets/eperiodica_v3/additional_processed_anns/train/eperiodica_minitrain/attribute_files/eperiodica_minitrain_VG_scene_graph_with_attri.h5"

  _C.DATASETS.EPERIODICA.VAL_IMAGES = "./datasets/eperiodica_v3/imgs/val"
  _C.DATASETS.EPERIODICA.VAL_MAPPING_DICTIONARY = "./datasets/eperiodica_v3/additional_processed_anns/val/eperiodica_minival/attribute_files/eperiodica_minival_VG_scene_graph_dicts_with_attri.json"
  _C.DATASETS.EPERIODICA.VAL_IMAGE_DATA = "./datasets/eperiodica_v3/anns/val/eperiodica_minival/eperiodica_minival_VG_scene_graph_image_data.json"
  _C.DATASETS.EPERIODICA.VAL_EPERIODICA_TARGET_ATTRIBUTE_H5 = "./datasets/eperiodica_v3/additional_processed_anns/val/eperiodica_minival/attribute_files/eperiodica_minival_VG_scene_graph_with_attri.h5"

  _C.DATASETS.EPERIODICA.TEST_IMAGES = "./datasets/eperiodica_v3/imgs/test"
  _C.DATASETS.EPERIODICA.TEST_MAPPING_DICTIONARY = "./datasets/eperiodica_v3/additional_processed_anns/test/eperiodica_minitest/attribute_files/eperiodica_minitest_VG_scene_graph_dicts_with_attri.json"
  _C.DATASETS.EPERIODICA.TEST_IMAGE_DATA = "./datasets/eperiodica_v3/anns/test/eperiodica_minitest/eperiodica_minitest_VG_scene_graph_image_data.json"
  _C.DATASETS.EPERIODICA.TEST_EPERIODICA_TARGET_ATTRIBUTE_H5 = "./datasets/eperiodica_v3/additional_processed_anns/test/eperiodica_minitest/attribute_files/eperiodica_minitest_VG_scene_graph_with_attri.h5"

  _C.DATASETS.EPERIODICA2 = CN()
  _C.DATASETS.EPERIODICA2.CLIPPED = False
  _C.DATASETS.EPERIODICA2.FILTER_EMPTY_RELATIONS = True
  _C.DATASETS.EPERIODICA2.FILTER_DUPLICATE_RELATIONS = True
  _C.DATASETS.EPERIODICA2.FILTER_NON_OVERLAP = False
  _C.DATASETS.EPERIODICA2.BOX_SCALE = 1024
  _C.DATASETS.EPERIODICA2.TRAIN_MASKS = ""
  _C.DATASETS.EPERIODICA2.TEST_MASKS = ""
  _C.DATASETS.EPERIODICA2.VAL_MASKS = ""

  _C.DATASETS.EPERIODICA2.TRAIN_IMAGES = "./datasets/eperiodica2/imgs/train"
  _C.DATASETS.EPERIODICA2.TRAIN_MAPPING_DICTIONARY = "./datasets/eperiodica2/additional_processed_anns/train/eperiodica_minitrain/attribute_files/eperiodica_minitrain_VG_scene_graph_dicts_with_attri.json"
  _C.DATASETS.EPERIODICA2.TRAIN_IMAGE_DATA = "./datasets/eperiodica2/anns/train/eperiodica_minitrain/eperiodica_minitrain_VG_scene_graph_image_data.json"
  _C.DATASETS.EPERIODICA2.TRAIN_EPERIODICA_TARGET_ATTRIBUTE_H5 = "./datasets/eperiodica2/additional_processed_anns/train/eperiodica_minitrain/attribute_files/eperiodica_minitrain_VG_scene_graph_with_attri.h5"

  _C.DATASETS.EPERIODICA2.VAL_IMAGES = "./datasets/eperiodica2/imgs/val"
  _C.DATASETS.EPERIODICA2.VAL_MAPPING_DICTIONARY = "./datasets/eperiodica2/additional_processed_anns/val/eperiodica_minival/attribute_files/eperiodica_minival_VG_scene_graph_dicts_with_attri.json"
  _C.DATASETS.EPERIODICA2.VAL_IMAGE_DATA = "./datasets/eperiodica2/anns/val/eperiodica_minival/eperiodica_minival_VG_scene_graph_image_data.json"
  _C.DATASETS.EPERIODICA2.VAL_EPERIODICA_TARGET_ATTRIBUTE_H5 = "./datasets/eperiodica2/additional_processed_anns/val/eperiodica_minival/attribute_files/eperiodica_minival_VG_scene_graph_with_attri.h5"

  _C.DATASETS.EPERIODICA2.TEST_IMAGES = "./datasets/eperiodica2/imgs/test"
  _C.DATASETS.EPERIODICA2.TEST_MAPPING_DICTIONARY = "./datasets/eperiodica2/additional_processed_anns/test/eperiodica_minitest/attribute_files/eperiodica_minitest_VG_scene_graph_dicts_with_attri.json"
  _C.DATASETS.EPERIODICA2.TEST_IMAGE_DATA = "./datasets/eperiodica2/anns/test/eperiodica_minitest/eperiodica_minitest_VG_scene_graph_image_data.json"
  _C.DATASETS.EPERIODICA2.TEST_EPERIODICA_TARGET_ATTRIBUTE_H5 = "./datasets/eperiodica2/additional_processed_anns/test/eperiodica_minitest/attribute_files/eperiodica_minitest_VG_scene_graph_with_attri.h5"

  #eperiodica3 contains ordered and unordered groups, but a reduced set of classes

  _C.DATASETS.EPERIODICA3 = CN()
  _C.DATASETS.EPERIODICA3.CLIPPED = False
  _C.DATASETS.EPERIODICA3.FILTER_EMPTY_RELATIONS = True
  _C.DATASETS.EPERIODICA3.FILTER_DUPLICATE_RELATIONS = True
  _C.DATASETS.EPERIODICA3.FILTER_NON_OVERLAP = False
  _C.DATASETS.EPERIODICA3.BOX_SCALE = 1024
  _C.DATASETS.EPERIODICA3.TRAIN_MASKS = "./datasets/eperiodica3/additional_processed_anns/train/eperiodica_minitrain/attribute_files/eperiodica_minitrain_VG_scene_graph_segmentations.json"
  _C.DATASETS.EPERIODICA3.VAL_MASKS = "./datasets/eperiodica3/additional_processed_anns/val/eperiodica_minival/attribute_files/eperiodica_minival_VG_scene_graph_segmentations.json"
  _C.DATASETS.EPERIODICA3.TEST_MASKS = "./datasets/eperiodica3/additional_processed_anns/test/eperiodica_minitest/attribute_files/eperiodica_minitest_VG_scene_graph_segmentations.json"


  _C.DATASETS.EPERIODICA3.TRAIN_IMAGES = "./datasets/eperiodica3/imgs/train"
  _C.DATASETS.EPERIODICA3.TRAIN_MAPPING_DICTIONARY = "./datasets/eperiodica3/additional_processed_anns/train/eperiodica_minitrain/attribute_files/eperiodica_minitrain_VG_scene_graph_dicts_with_attri.json"
  _C.DATASETS.EPERIODICA3.TRAIN_IMAGE_DATA = "./datasets/eperiodica3/anns/train/eperiodica_minitrain/eperiodica_minitrain_VG_scene_graph_image_data.json"
  _C.DATASETS.EPERIODICA3.TRAIN_EPERIODICA_TARGET_ATTRIBUTE_H5 = "./datasets/eperiodica3/additional_processed_anns/train/eperiodica_minitrain/attribute_files/eperiodica_minitrain_VG_scene_graph_with_attri.h5"

  _C.DATASETS.EPERIODICA3.VAL_IMAGES = "./datasets/eperiodica3/imgs/val"
  _C.DATASETS.EPERIODICA3.VAL_MAPPING_DICTIONARY = "./datasets/eperiodica3/additional_processed_anns/val/eperiodica_minival/attribute_files/eperiodica_minival_VG_scene_graph_dicts_with_attri.json"
  _C.DATASETS.EPERIODICA3.VAL_IMAGE_DATA = "./datasets/eperiodica3/anns/val/eperiodica_minival/eperiodica_minival_VG_scene_graph_image_data.json"
  _C.DATASETS.EPERIODICA3.VAL_EPERIODICA_TARGET_ATTRIBUTE_H5 = "./datasets/eperiodica3/additional_processed_anns/val/eperiodica_minival/attribute_files/eperiodica_minival_VG_scene_graph_with_attri.h5"

  _C.DATASETS.EPERIODICA3.TEST_IMAGES = "./datasets/eperiodica3/imgs/test"
  _C.DATASETS.EPERIODICA3.TEST_MAPPING_DICTIONARY = "./datasets/eperiodica3/additional_processed_anns/test/eperiodica_minitest/attribute_files/eperiodica_minitest_VG_scene_graph_dicts_with_attri.json"
  _C.DATASETS.EPERIODICA3.TEST_IMAGE_DATA = "./datasets/eperiodica3/anns/test/eperiodica_minitest/eperiodica_minitest_VG_scene_graph_image_data.json"
  _C.DATASETS.EPERIODICA3.TEST_EPERIODICA_TARGET_ATTRIBUTE_H5 = "./datasets/eperiodica3/additional_processed_anns/test/eperiodica_minitest/attribute_files/eperiodica_minitest_VG_scene_graph_with_attri.h5"


  _C.DATASETS.MSCOCO = CN()
  _C.DATASETS.MSCOCO.ANNOTATIONS = ''
  _C.DATASETS.MSCOCO.DATAROOT = ''

  _C.DATASETS.VISUAL_GENOME.FILTER_EMPTY_RELATIONS = True
  _C.DATASETS.VISUAL_GENOME.FILTER_DUPLICATE_RELATIONS = True
  _C.DATASETS.VISUAL_GENOME.FILTER_NON_OVERLAP = True
  _C.DATASETS.VISUAL_GENOME.NUMBER_OF_VALIDATION_IMAGES = 5000
  _C.DATASETS.VISUAL_GENOME.BOX_SCALE = 1024

  _C.DATASETS.SEG_DATA_DIVISOR = 1

  _C.DATASETS.TRANSFER = ('coco_train_2014',)
  _C.DATASETS.MASK_TRAIN = ('coco_train_2017',)
  _C.DATASETS.MASK_TEST = ('coco_val_2017',)
