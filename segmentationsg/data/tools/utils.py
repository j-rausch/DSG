import os
import sys

from ..datasets import VisualGenomeTrainData, SggDataset

def register_datasets(cfg):
    if cfg.DATASETS.TYPE == 'VISUAL GENOME':
        for split in ['train', 'val', 'test']:
            dataset_instance = VisualGenomeTrainData(cfg, split=split)
    elif cfg.DATASETS.TYPE == 'ADtgt':
        for split in ['train', 'val', 'test']:
            mask_location = cfg.DATASETS.ARXIVDOCS_TARGET.get(split.upper() + '_MASKS')
            dataset_name = 'ADtgt_{}'.format(split)
            dataset_image_dir = cfg.DATASETS.ARXIVDOCS_TARGET.get(split.upper()+'_IMAGES')
            scenegraph_dataset_h5_path = cfg.DATASETS.ARXIVDOCS_TARGET.get(split.upper()+'_ARXIVDOCS_TARGET_ATTRIBUTE_H5')
            dataset_image_data_json_path = cfg.DATASETS.ARXIVDOCS_TARGET.get(split.upper()+'_IMAGE_DATA')
            mapping_dictionary_json_path = cfg.DATASETS.ARXIVDOCS_TARGET.get(split.upper()+'_MAPPING_DICTIONARY')
            scenegraph_dictionary_cache_file = None

            dataset_instance = SggDataset(cfg, dataset_name, dataset_image_dir,
                                          mapping_dictionary_json_path,
                                          scenegraph_dataset_h5_path, dataset_image_data_json_path,
                                          scenegraph_dictionary_cache_file, dataset_split=split,
                                          is_clipped=False,
                                          filter_empty_relations=True, filter_duplicate_relations=True,
                                          filter_non_overlapping_boxes=False, box_scale=1024,
                                          mask_location=mask_location, use_basename_for_file_paths=False)
    elif cfg.DATASETS.TYPE == 'EP' or cfg.DATASETS.TYPE == 'EP2':
        raise NotImplementedError
#        for split in ['train', 'val', 'test']:
#            dataset_instance = EperiodicaTrainData(cfg, split=split)
    elif cfg.DATASETS.TYPE == 'ADwk':
        for split in ['train']:
            mask_location = cfg.DATASETS.ARXIVDOCS_WEAK.get(split.upper() + '_MASKS')
            dataset_name = 'ADwk_{}'.format(split)
            dataset_image_dir = cfg.DATASETS.ARXIVDOCS_WEAK.get(split.upper() + '_IMAGES')
            scenegraph_dataset_h5_path = cfg.DATASETS.ARXIVDOCS_WEAK.get(
                split.upper() + '_ARXIVDOCS_WEAK_ATTRIBUTE_H5')
            dataset_image_data_json_path = cfg.DATASETS.ARXIVDOCS_WEAK.get(
                split.upper() + '_IMAGE_DATA')
            mapping_dictionary_json_path = cfg.DATASETS.ARXIVDOCS_WEAK.get(
                split.upper() + '_MAPPING_DICTIONARY')
            scenegraph_dictionary_cache_file = None
            dataset_instance = SggDataset(cfg, dataset_name, dataset_image_dir,
                                          mapping_dictionary_json_path,
                                          scenegraph_dataset_h5_path, dataset_image_data_json_path,
                                          scenegraph_dictionary_cache_file, dataset_split=split,
                                          is_clipped=False,
                                          filter_empty_relations=True,
                                          filter_duplicate_relations=True,
                                          filter_non_overlapping_boxes=False, box_scale=1024,
                                          mask_location=mask_location,
                                          use_basename_for_file_paths=False,
                                          remove_subdir_from_file_name=True)
        for split in ['val', 'test']:
            mask_location = cfg.DATASETS.ARXIVDOCS_TARGET.get(split.upper() + '_MASKS')
            dataset_name = 'ADtgt_{}'.format(split)
            dataset_image_dir = cfg.DATASETS.ARXIVDOCS_TARGET.get(split.upper() + '_IMAGES')
            scenegraph_dataset_h5_path = cfg.DATASETS.ARXIVDOCS_TARGET.get(
                split.upper() + '_ARXIVDOCS_TARGET_ATTRIBUTE_H5')
            dataset_image_data_json_path = cfg.DATASETS.ARXIVDOCS_TARGET.get(
                split.upper() + '_IMAGE_DATA')
            mapping_dictionary_json_path = cfg.DATASETS.ARXIVDOCS_TARGET.get(
                split.upper() + '_MAPPING_DICTIONARY')
            scenegraph_dictionary_cache_file = None
            dataset_instance = SggDataset(cfg, dataset_name, dataset_image_dir,
                                          mapping_dictionary_json_path,
                                          scenegraph_dataset_h5_path, dataset_image_data_json_path,
                                          scenegraph_dictionary_cache_file, dataset_split=split,
                                          is_clipped=False,
                                          filter_empty_relations=True,
                                          filter_duplicate_relations=True,
                                          filter_non_overlapping_boxes=False, box_scale=1024,
                                          mask_location=mask_location,
                                          use_basename_for_file_paths=False
                                          )
    elif cfg.DATASETS.TYPE == 'EP3':
        for split in ['train', 'val', 'test']:
            mask_location = cfg.DATASETS.EPERIODICA3.get(split.upper()+'_MASKS')
            dataset_name = 'EP3_{}'.format(split)
            dataset_image_dir = cfg.DATASETS.EPERIODICA3.get(split.upper()+'_IMAGES')
            scenegraph_dataset_h5_path = cfg.DATASETS.EPERIODICA3.get(split.upper()+'_EPERIODICA_TARGET_ATTRIBUTE_H5')
            dataset_image_data_json_path = cfg.DATASETS.EPERIODICA3.get(split.upper()+'_IMAGE_DATA')
            mapping_dictionary_json_path = cfg.DATASETS.EPERIODICA3.get(split.upper()+'_MAPPING_DICTIONARY')
            scenegraph_dictionary_cache_file = None

            dataset_instance = SggDataset(cfg, dataset_name, dataset_image_dir, mapping_dictionary_json_path,
                         scenegraph_dataset_h5_path, dataset_image_data_json_path,
                         scenegraph_dictionary_cache_file, dataset_split=split, is_clipped=False,
                         filter_empty_relations=True, filter_duplicate_relations=True,
                         filter_non_overlapping_boxes=False, box_scale=1024, mask_location=mask_location, use_basename_for_file_paths=True)
