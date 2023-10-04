#!/usr/bin/env python3
"""
PDFMiner XML to Annotation JSON Converter
"""

import argparse
from bs4 import BeautifulSoup
import json
import sys
import os
from pdf2image import convert_from_path
import subprocess

id = 0



def get_id():
    global id
    id += 1
    return id


def new_structure(id, category, parent):
    return {
        'id': id,
        'category': category,
        'parent': parent
    }


def new_box(id, category, page, bbox, parent, text=None):
    return {
        'id': id,
        'category': category,
        'page': page,
        'bbox': bbox,
        'parent': parent,
        'text': text
    }


def parse_bbox(bbox):
    coords = [float(f) for f in bbox.split(',')]
    return coords


def cover_bbox(bboxes):
    return [
        min([b[0] for b in bboxes]),
        min([b[1] for b in bboxes]),
        max([b[2] for b in bboxes]),
        max([b[3] for b in bboxes])
    ]


def convert_bbox(bbox, page_bbox):
    coords = bbox.copy()
    coords[1] = page_bbox[3] - bbox[3]
    coords[3] = page_bbox[3] - bbox[1]
    coords[2] -= coords[0]
    coords[3] -= coords[1]
    return [round(c, 3) for c in coords]

#
#parser = argparse.ArgumentParser(description=__doc__)
#parser.add_argument('-p', '--paragraph', action='store_true',
#                    help='create paragraph structure node for annotations')
#parser.add_argument('-f', '--figure', action='store_true',
#                    help='create figure structure/annotation nodes')
#parser.add_argument('-l', '--line', action='store_true',
#                    help='create line structure/annotation nodes')
#parser.add_argument('-w', '--word', action='store_true',
#                    help='create annotations for words')
#parser.add_argument('-c', '--character', action='store_true',
#                    help='create annotations for characters')
#args = parser.parse_args()


def convert_xml_to_dict(xml_string, use_paragraphs=True, use_figures=True, use_content_lines=True, use_lines=False, use_words=True, use_characters=False):

    output = []
    #soup = BeautifulSoup(sys.stdin.read(), 'xml')
    soup = BeautifulSoup(xml_string, 'xml')
    pages = soup.pages.find_all('page')
    unk = new_structure(get_id(), 'unk', None)
    output.append(unk)
    meta = new_structure(get_id(), 'meta', None)
    output.append(meta)
    root = new_structure(get_id(), 'document', None)
    output.append(root)
    for page in pages:
        page_id = int(page['id']) - 1
        page_bbox = parse_bbox(page['bbox'])
        textboxes = page.find_all('textbox')
        for textbox in textboxes:
            textlines = textbox.find_all('textline')
            parent = root['id']
            if use_paragraphs and len(textlines) > 0:
                paragraph = new_structure(get_id(), 'paragraph', root['id'])
                output.append(paragraph)
                parent = paragraph['id']
            else:
                parent = root['id']
            #TODO: fix character parent
            for textline in textlines:
                if use_words:
                    if use_content_lines:
                        content_line = new_structure(get_id(), 'content_line', parent)
                        output.append(content_line)
                        word_parent = content_line['id']
                    else:
                        word_parent = parent
                    texts = textline.find_all('text')
                    bboxes = []
                    word = ''
                    for text in texts:
                        if text.has_attr('bbox'):
                            bboxes.append(parse_bbox(text['bbox']))
                            try:
                                word += text.string
                            except TypeError:
                                continue
                        else:
                            box = new_box(get_id(), 'box', page_id, convert_bbox(
                                cover_bbox(bboxes), page_bbox), word_parent, word)
                            output.append(box)
                            bboxes = []
                            word = ''
                elif use_characters:
                    content_line = new_structure(get_id(), 'content_line', root['id'])
                    output.append(content_line)
                    texts = textline.find_all('text')
                    word = new_structure(get_id(), 'word', content_line['id'])
                    output.append(word)
                    for text in texts:
                        if text.has_attr('bbox'):
                            box = new_box(get_id(), 'box', page_id, convert_bbox(parse_bbox(text['bbox']), page_bbox), word['id'], text.string)
                            output.append(box)
                        else:
                            word = new_structure(get_id(), 'word', content_line['id'])
                            output.append(word)
                else:
                    box = new_box(get_id(), 'box', page_id, convert_bbox(parse_bbox(textline['bbox']), page_bbox), box_ann_id)
                    output.append(box)
        if use_figures:
            for figure in page.find_all('figure'):
                f = new_structure(get_id(), 'image', root['id'])
                output.append(f)
                output.append(new_box(get_id(), 'box', page_id, convert_bbox(parse_bbox(figure['bbox']), page_bbox), f['id']))
        if use_lines:
            for line in page.find_all('line'):
                l = new_structure(get_id(), 'line', root['id'])
                output.append(l)
                output.append(new_box(get_id(), 'box', page_id, convert_bbox(parse_bbox(line['bbox']), page_bbox), l['id']))

    return output
    #print(json.dumps(output))


def convert_pdfminer_output_to_docparser_json(pdfminer_input, docparser_output, use_paragraphs=True, use_content_lines=True):
    with open(pdfminer_input, 'r') as in_file:
        xml_content = in_file.readlines()
        xml_string = "".join(xml_content)
        pdfminer_dict = convert_xml_to_dict(xml_string, use_paragraphs=use_paragraphs, use_figures=True,
                                            use_content_lines=use_content_lines, use_lines=False,
                                                     use_words=True, use_characters=False)
        with open(docparser_output, 'w') as out_file:
            json.dump(pdfminer_dict, out_file)


def run_pdfminer(pdf_path, output_xml_path):
    #pdfminer_output = os.path.join(target_dir, doc_id, doc_id + '_pdfminer.xml')
    subprocess.run(["pdf2txt.py", "-t", "xml", pdf_path, '-o', output_xml_path])

def create_pdfminer_annotations(document_dir, pdf_filename, use_paragraphs=True, use_content_lines=True):
    pdf_path = os.path.join(document_dir, pdf_filename)
    document_id = os.path.basename(pdf_filename).lower().replace('.pdf','')
    output_xml_path = os.path.join(document_dir, document_id + '_pdfminer.xml')
    run_pdfminer(pdf_path, output_xml_path)
    docparser_annotation_path = os.path.join(document_dir, document_id + '-pdfm.json')
    convert_pdfminer_output_to_docparser_json(output_xml_path, docparser_annotation_path, use_paragraphs=use_paragraphs, use_content_lines=use_content_lines)

def create_meta_files_from_images(document_dir, num_pages, doc_id):
    meta_dict = {'id' : doc_id, 'title': doc_id, 'pages': num_pages}
    json_output_path = os.path.join(document_dir, doc_id + '.json')

    with open(json_output_path, 'w') as out_file:
        #print('creating meta json file at {}'.format(json_output_path))
        json.dump(meta_dict, out_file, indent=1, sort_keys=True)


def create_images_and_meta_file_for_pdf(document_dir, pdf_filename, output_basename=None, dpi=72):
    pdf_path = os.path.join(document_dir, pdf_filename)
    pages = convert_from_path(pdf_path, dpi=dpi)
    if output_basename is None:
        output_basename = os.path.basename(document_dir)
    for i, page in enumerate(pages):
        output_filepath = os.path.join(document_dir, output_basename + '-{}.png'.format(i))
        #width, height = page.size
        #print('page {} of {}, height: {}, width: {}'.format(i, pdf_filename, height, width))
        page.save(output_filepath, 'PNG')
    num_pages = len(pages)
    create_meta_files_from_images(document_dir, num_pages, output_basename)
