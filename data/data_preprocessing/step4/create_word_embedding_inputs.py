import os
import shutil, errno
import subprocess
import pdf2json
import json
from collections import defaultdict


def copyanything(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc:  # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else:
            raise


def fetch_pdfs(file_id_path, source_dir, target_dir):
    with open(file_id_path, 'r') as in_file:
        file_ids = [x.strip() for x in in_file.readlines()]

    for file_id in file_ids:
        print(file_id)
        doc_id = file_id.rsplit('_')[0]
        print(doc_id)
        doc_source_path = os.path.join(source_dir, doc_id)
        dest_path = os.path.join(target_dir, doc_id)
        print('copy from {} to {}'.format(doc_source_path, dest_path))
        copyanything(doc_source_path, dest_path)


def fetch_data():
    for split in ['train', 'dev', 'test']:
        dataset_file_id_path = '../datasets/arxiv_mixed_full/manual3/splits/by_page/{}_ids.txt'.format(
            split)
        source_dir = '/mnt/ds3lab-scratch/arxiv/mixedset/mixed_v2_full_set_with_imgs_splits/splits/dev/'
        target_dir = '../datasets/arxiv_mixed_full/manual3/fulldocs/{}'.format(split)
        fetch_pdfs(dataset_file_id_path, source_dir, target_dir)


def run_pdfminer_on_docs(dataset_file_id_path, target_dir):
    with open(dataset_file_id_path, 'r') as in_file:
        file_ids = [x.strip() for x in in_file.readlines()]

    for file_id in file_ids:
        doc_id = file_id.rsplit('_')[0]
        print(doc_id)
        doc_path = os.path.join(target_dir, doc_id, doc_id + '.pdf')
        print(os.path.isfile(doc_path))
        pdfminer_output = os.path.join(target_dir, doc_id, doc_id + '_pdfminer.xml')
        subprocess.run(["pdf2txt.py", "-t", "xml", doc_path, '-o', pdfminer_output])


def run_pdfminer_on_data():
    for split in ['train', 'dev', 'test']:
        dataset_file_id_path = '../datasets/arxiv_mixed_full/manual3/splits/by_page/{}_ids.txt'.format(
            split)
        target_dir = '../datasets/arxiv_mixed_full/manual3/fulldocs/{}'.format(split)
        run_pdfminer_on_docs(dataset_file_id_path, target_dir)


def convert_pdfminer_to_json(dataset_file_id_path, source_dir):
    with open(dataset_file_id_path, 'r') as in_file:
        file_ids = [x.strip() for x in in_file.readlines()]

    for file_id in file_ids:
        doc_id = file_id.rsplit('_')[0]
        print(file_id)
        page_nr = file_id.rsplit('_')[-1]
        print('page: {}'.format(page_nr))
        pdfminer_input = os.path.join(source_dir, doc_id, doc_id + '_pdfminer.xml')
        converted_pdfminer_json_output = os.path.join(source_dir, doc_id, doc_id + '-pdfm.json')
        with open(pdfminer_input, 'r') as in_file:
            xml_content = in_file.readlines()
            xml_string = "".join(xml_content)
            pdfminer_dict = pdf2json.convert_xml_to_dict(xml_string)
            # input(pdfminer_dict)
        print(converted_pdfminer_json_output)
        with open(converted_pdfminer_json_output, 'w') as out_file:
            json.dump(pdfminer_dict, out_file)


def extract_pdfminer_json_page(dataset_file_id_path, source_dir, target_dir):
    with open(dataset_file_id_path, 'r') as in_file:
        file_ids = [x.strip() for x in in_file.readlines()]

    for file_id in file_ids:
        doc_id = file_id.rsplit('_')[0]
        print(file_id)
        page_nr = file_id.rsplit('_')[-1]
        print('page: {}'.format(page_nr))
        converted_pdfminer_json = os.path.join(source_dir, doc_id, doc_id + '-pdfm.json')
        with open(converted_pdfminer_json, 'r') as in_file:
            anns = json.load(in_file)

        ann_by_id = dict()
        filtered_children_by_parent = defaultdict(list)
        filtered_anns = []
        for ann in anns:
            if 'page' in ann:
                if int(ann['page']) != int(page_nr):
                    continue
            ann_by_id[ann['id']] = ann
            filtered_anns.append(ann)
            if ann['parent'] is not None:
                filtered_children_by_parent[ann['parent']].append(ann)

        result_annotation_list = []
        for ann in filtered_anns:
            if ann['category'] in ['meta', 'document']:
                result_annotation_list.append(ann)
            elif len(filtered_children_by_parent[ann['id']]) > 0:
                result_annotation_list.append(ann)
            elif 'bbox' in ann:
                result_annotation_list.append(ann)
        # input(result_annotation_list )

        # input(pdfminer_dict)
        #        doc_path = os.path.join(target_dir, doc_id, doc_id + '.pdf')
        #        print(os.path.isfile(doc_path))
        #        pdfminer_output =  os.path.join(target_dir, doc_id, doc_id + '_pdfminer.xml')

        # set all annotation pages to dummy 0
        for ann in result_annotation_list:
            if 'page' in ann:
                ann['page'] = 0

        target_path = os.path.join(target_dir, file_id, file_id + '-pdfm.json')
        print(target_path)
        with open(target_path, 'w') as out_file:
            json.dump(result_annotation_list, out_file)


def convert_pdfminer2json():
    for split in ['train', 'dev', 'test']:
        dataset_file_id_path = '../datasets/arxiv_mixed_full/manual3/splits/by_page/{}_ids.txt'.format(
            split)
        source_dir = '../datasets/arxiv_mixed_full/manual3/fulldocs/{}'.format(split)
        convert_pdfminer_to_json(dataset_file_id_path, source_dir)


def extract_pdfminer_page():
    for split in ['train', 'dev', 'test']:
        dataset_file_id_path = '../datasets/arxiv_mixed_full/manual3/splits/by_page/{}_ids.txt'.format(
            split)
        source_dir = '../datasets/arxiv_mixed_full/manual3/fulldocs/{}'.format(split)
        target_dir = '../datasets/arxiv_mixed_full/manual3/splits/by_page/{}/'.format(split)
        extract_pdfminer_json_page(dataset_file_id_path, source_dir, target_dir)


#if __name__ == '__main__':
#    # fetch_data()
#    # get pdfminer annotations
#    # run_pdfminer_on_data()
##    convert_pdfminer2json()
##    extract_pdfminer_page()