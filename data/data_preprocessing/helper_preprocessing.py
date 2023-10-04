from pathlib import Path
from step1.vg_to_imdb import build_imdb_from_vg
from step1.vg_to_roidb import create_roidb
from step2.generate_attribute_labels import create_attribute_files
from step3.generate_segmentations import create_segmentation_file
import os

def step1_imdb(splits, anns_root, imgs_root, processed_output_root, dataset_descriptor_template, file_exists_checks=True, use_img_filename_instead_of_relpath=False):
    for split in splits:
        img_subdir = imgs_root.format(split=split)
        anns_subdir = anns_root.format(split=split)
        output_subdir = processed_output_root.format(split=split)
        dataset_descriptor = dataset_descriptor_template.format(split=split)
        print('output subdir: {}'.format(output_subdir))
        Path(output_subdir).mkdir(parents=True, exist_ok=True)
        scenegraph_image_data_json_filename =  dataset_descriptor + '_image_data.json'
        scenegraph_image_data_json_filepath = os.path.join(anns_subdir, scenegraph_image_data_json_filename)

        output_h5_imdb_filename = dataset_descriptor + '_imdb_1024.h5'
        build_imdb_from_vg(img_subdir, output_h5_imdb_filename, output_subdir, scenegraph_image_data_json_filepath, file_exists_checks=file_exists_checks, use_img_filename_instead_of_relpath=use_img_filename_instead_of_relpath) 

def step1_roidb(splits, anns_root, imgs_root, processed_output_root,dataset_descriptor_template):

    for split in splits:
        img_subdir = imgs_root.format(split=split)
        anns_subdir = anns_root.format(split=split)
        output_subdir = processed_output_root.format(split=split)
        dataset_descriptor = dataset_descriptor_template.format(split=split)
        object_input = os.path.join(anns_subdir, dataset_descriptor + '_objects.json')
        relationship_input = os.path.join(anns_subdir, dataset_descriptor + '_relationships.json')
        metadata_input = os.path.join(anns_subdir, dataset_descriptor + '_image_data.json')
        object_alias = os.path.join(anns_subdir, dataset_descriptor + '_object_alias.txt')
        pred_alias = os.path.join(anns_subdir, dataset_descriptor + '_predicate_alias.txt')
        object_list = os.path.join(anns_subdir, dataset_descriptor + '_object_list.txt')
        pred_list = os.path.join(anns_subdir, dataset_descriptor + '_predicate_list.txt')
        imdb = os.path.join(output_subdir, dataset_descriptor + '_imdb_1024.h5')
        json_file = os.path.join(output_subdir, dataset_descriptor + '_dicts.json')
        h5_file = os.path.join(output_subdir, dataset_descriptor + '.h5')
        if split == 'train':
            train_frac, val_frac = 1.0, 1.0 
        elif split == 'dev' or split == 'val':
            train_frac, val_frac = 0.0, 1.0 
        elif split == 'test':
            train_frac, val_frac = 0.0, 0.0 
        else:
            raise NotImplementedError

        create_roidb(imdb, object_input, relationship_input, metadata_input, object_alias, pred_alias, object_list, pred_list, json_file, h5_file, train_frac, val_frac)

def step2_attributes(splits, anns_root, imgs_root, processed_output_root,dataset_descriptor_template):

    for split in splits:
        img_subdir = imgs_root.format(split=split)
        anns_subdir = anns_root.format(split=split)
        output_subdir = processed_output_root.format(split=split)
        dataset_descriptor = dataset_descriptor_template.format(split=split)
        metadata_input = os.path.join(anns_subdir, dataset_descriptor + '_image_data.json')
        json_file = os.path.join(output_subdir, dataset_descriptor + '_dicts.json')
        h5_file = os.path.join(output_subdir, dataset_descriptor + '.h5')
        object_input = os.path.join(anns_subdir, dataset_descriptor + '_objects.json')
        attribute_input = os.path.join(anns_subdir, dataset_descriptor + '_attributes.json')
        attribute_synsets_input = os.path.join(anns_subdir, dataset_descriptor + '_attributes_synsets.json')
        attribute_dict_file_dir = os.path.join(output_subdir, 'attribute_files')
        Path(attribute_dict_file_dir).mkdir(parents=True, exist_ok=True)


        output_attribute_dict_file_path = os.path.join(attribute_dict_file_dir, dataset_descriptor + '_dicts_with_attri.json')
        output_attribute_h5_file_path = os.path.join(attribute_dict_file_dir, dataset_descriptor + '_with_attri.h5')


#        output_attribute_dict_file_path  = f'arxivdocs-{dataset_type}-SGG-{split}-dicts-with-attri.json'
#        output_attribute_h5_file_path = f'arxivdocs-{dataset_type}-SGG-{split}-with-attri.h5'

        print('starting attribute data generation')
        create_attribute_files(metadata_input, json_file, h5_file, object_input, attribute_input, attribute_synsets_input, output_subdir, output_attribute_dict_file_path, output_attribute_h5_file_path)

def step3_segmentation(splits, anns_root, imgs_root, processed_output_root, dataset_descriptor_template):
    for split in splits:
        img_subdir = imgs_root.format(split=split)
        anns_subdir = anns_root.format(split=split)
        output_subdir = processed_output_root.format(split=split)
        dataset_descriptor = dataset_descriptor_template.format(split=split)
        metadata_input = os.path.join(anns_subdir, dataset_descriptor + '_image_data.json')
        object_input = os.path.join(anns_subdir, dataset_descriptor + '_objects.json')
        relationship_input = os.path.join(anns_subdir, dataset_descriptor + '_relationships.json')
        attribute_synsets_input = os.path.join(anns_subdir, dataset_descriptor + '_attributes_synsets.json')

        attribute_dict_file_dir = os.path.join(output_subdir, 'attribute_files')
        Path(attribute_dict_file_dir).mkdir(parents=True, exist_ok=True)

        attribute_dict_file_path = os.path.join(attribute_dict_file_dir,
                                                       dataset_descriptor + '_dicts_with_attri.json')
        output_json_path = os.path.join(attribute_dict_file_dir, dataset_descriptor + '_segmentations.json')

        print('starting segmentation data generation')
        create_segmentation_file(img_subdir, anns_subdir, output_subdir, dataset_descriptor, metadata_input,
                                 object_input, relationship_input, attribute_synsets_input, attribute_dict_file_dir,
                                 attribute_dict_file_path, output_json_path)


def process_arxivdocs():
    directory_mappings = {'target': 'ADtgt_VGv2', 'weak': 'ADwk_VGv2'}
    dataset_types = ['target', 'weak']
    #dataset_types = ['target']
    splits = ['train', 'dev', 'test']
    for dataset_type in dataset_types:
        dir_mapping = directory_mappings[dataset_type] 
        anns_root = f'../../datasets/{dir_mapping}/anns' + '/{split}/'
        imgs_root = f'../../datasets/{dir_mapping}/imgs' + '/{split}/'
        processed_output_root = f'../../datasets/{dir_mapping}/additional_processed_anns' + '/{split}/'
        dataset_descriptor_template = f'arxivdocs_{dataset_type}_layout_v2' + '_{split}_scene_graph'
        step1_imdb(splits, anns_root, imgs_root, processed_output_root, dataset_descriptor_template, file_exists_checks=False)  
        step1_roidb(splits, anns_root, imgs_root, processed_output_root, dataset_descriptor_template)
        step2_attributes(splits, anns_root, imgs_root, processed_output_root,dataset_descriptor_template)
#        step3_segmentation(splits, anns_root, imgs_root, processed_output_root, dataset_descriptor_template)

def process_eperiodica():
    anns_root = '../../datasets/eperiodica3/anns/{split}/eperiodica_mini{split}/'
    imgs_root = '../../datasets/eperiodica3/imgs/{split}/'
    processed_output_root = '../../datasets/eperiodica3/additional_processed_anns/{split}/eperiodica_mini{split}/'

    dataset_descriptor_template = 'eperiodica_mini{split}_VG_scene_graph'
    #splits = ['train', 'val']#, 'test']
    splits = ['train', 'val', 'test']
    step1_imdb(splits, anns_root, imgs_root, processed_output_root, dataset_descriptor_template, file_exists_checks=True, use_img_filename_instead_of_relpath=True)
    step1_roidb(splits, anns_root, imgs_root, processed_output_root, dataset_descriptor_template)
    step2_attributes(splits, anns_root, imgs_root, processed_output_root,dataset_descriptor_template)
#    step3_segmentation(splits, anns_root, imgs_root, processed_output_root, dataset_descriptor_template)

if __name__ == '__main__':
    #process_arxivdocs()
    process_eperiodica()

