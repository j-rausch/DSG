# coding=utf8

import argparse, os, json, string
from queue import Queue
from threading import Thread, Lock

import h5py
import numpy as np
# from scipy.misc import imresize, imread
import cv2


def add_images(im_data, h5_file, image_dir, image_size=1024, num_workers=20, file_exists_checks=True, use_img_filename_instead_of_relpath=True):
    fns = []; ids = []; idx = []
    for i, img in enumerate(im_data):
        if i % 10000 == 0:
            print('img {} of {}'.format(i, len(im_data)))
        if use_img_filename_instead_of_relpath is True:
            img_rel_path = os.path.basename(img['file_name'])
        else:
            img_rel_path = img['file_name']
        filename = os.path.join(image_dir, img_rel_path)
        if file_exists_checks is True and not os.path.exists(filename):
            print('skipping not existing file: {}'.format(filename))
            continue
        fns.append(filename)
        ids.append(img['image_id'])
        idx.append(i)

    ids = np.array(ids, dtype=np.int32)
    idx = np.array(idx, dtype=np.int32)
    h5_file.create_dataset('image_ids', data=ids)
    h5_file.create_dataset('valid_idx', data=idx)

    num_images = len(fns)

    shape = (num_images, 3, image_size, image_size)
    image_dset = h5_file.create_dataset('images', shape, dtype=np.uint8)
    original_heights = np.zeros(num_images, dtype=np.int32)
    original_widths = np.zeros(num_images, dtype=np.int32)
    image_heights = np.zeros(num_images, dtype=np.int32)
    image_widths = np.zeros(num_images, dtype=np.int32)

    lock = Lock()
    q = Queue()
    for i, fn in enumerate(fns):
        q.put((i, fn))

    def worker():
        while True:
            i, filename = q.get()

            if i % 50 == 0:
                print('processed %i images...' % i)
            img = cv2.imread(filename)
            # handle grayscale
            if img.ndim == 2:
                img = img[:, :, None][:, :, [0, 0, 0]]
            H0, W0 = img.shape[0], img.shape[1]
            # img = imresize(img, float(image_size) / max(H0, W0))
            img = cv2.resize(img, dsize=(image_size, image_size), interpolation=cv2.INTER_LINEAR)
            H, W = img.shape[0], img.shape[1]
            # swap rgb to bgr. This can't be the best way right? #fail
            r = img[:,:,0].copy()
            img[:,:,0] = img[:,:,2]
            img[:,:,2] = r

            lock.acquire()
            original_heights[i] = H0
            original_widths[i] = W0
            image_heights[i] = H
            image_widths[i] = W
            image_dset[i, :, :H, :W] = img.transpose(2, 0, 1)
            lock.release()
            q.task_done()

    for i in range(num_workers):
        t = Thread(target=worker)
        t.daemon = True
        t.start()

    q.join()

    h5_file.create_dataset('image_heights', data=image_heights)
    h5_file.create_dataset('image_widths', data=image_widths)
    h5_file.create_dataset('original_heights', data=original_heights)
    h5_file.create_dataset('original_widths', data=original_widths)

    return fns


def build_imdb_from_vg(image_dir, h5_output_filename, h5_output_dir, metadata_input_json_path, image_size=1024, num_workers=20, file_exists_checks=True, use_img_filename_instead_of_relpath=False):
    im_metadata = json.load(open(metadata_input_json_path))
    # write the h5 file
    if not os.path.exists(h5_output_dir):
        print('Creating output directory: {}'.format(h5_output_dir))
        os.makedirs(h5_output_dir, exist_ok=True) 
    h5_file_path = os.path.join(h5_output_dir, h5_output_filename)
    h5_file = h5py.File(h5_file_path, 'w')
    # load images
    im_fns = add_images(im_metadata, h5_file, image_dir, image_size, num_workers, file_exists_checks=file_exists_checks, use_img_filename_instead_of_relpath=use_img_filename_instead_of_relpath)
    print('finished generation')
#
#
#
#
#if __name__ == '__main__':
#    #TODO: check if we have to rescale images to have consistent size
##    run_generation(is_weak_dataset=False, split_str='train')
##    run_generation(is_weak_dataset=False, split_str='dev')
##    run_generation(is_weak_dataset=False, split_str='test')
#    #weak
#    run_generation(is_weak_dataset=True, split_str='train')
#    run_generation(is_weak_dataset=True, split_str='dev')
#    run_generation(is_weak_dataset=True, split_str='test')
#
