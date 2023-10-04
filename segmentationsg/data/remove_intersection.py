import pickle
import json

coco_train = json.load(open('/h/skhandel/FewshotDetection/WSASOD/data/data_utils/data/MSCOCO2017/annotations/instances_train2017.json', 'r'))
coco_test = json.load(open('/h/skhandel/FewshotDetection/WSASOD/data/data_utils/data/MSCOCO2017/annotations/instances_val2017.json', 'r'))

vg_test_ids = pickle.load(open('/h/skhandel/SceneGraph/scripts/test_coco_ids.pkl', 'rb'))
vg_train_ids = pickle.load(open('/h/skhandel/SceneGraph/scripts/VG_COCO_ids_train', 'rb'))
coco_test_ids = [x['id'] for x in coco_test['images']]
coco_train_ids = [x['id'] for x in coco_train['images']]
num_overlap = 0
for id in coco_test_ids:
    if id in vg_train_ids:
        num_overlap += 1

num_overlap_1 = 0
ids_to_remove = []
for id in vg_test_ids:
    if id in coco_train_ids:
        ids_to_remove.append(id)
        num_overlap_1 += 1

num_overlap_2 = 0
for id in vg_test_ids:
    if id in coco_test_ids:
        num_overlap_2 += 1

num_overlap_3 = 0
for id in coco_test_ids:
    if id in vg_test_ids:
        num_overlap_3 += 1

new_coco_train = {}
new_coco_train['info'] = coco_train['info']
new_coco_train['categories'] = coco_train['categories']
new_coco_train['licenses'] = coco_train['licenses']
new_coco_train['images'] = []
new_coco_train['annotations'] = []

new_coco_test = {}
new_coco_test['info'] = coco_test['info']
new_coco_test['categories'] = coco_test['categories']
new_coco_test['licenses'] = coco_test['licenses']
new_coco_test['images'] = []
new_coco_test['annotations'] = []


for idx, data in enumerate(coco_train['images']):
    if data['id'] not in vg_test_ids:
        new_coco_train['images'].append(coco_train['images'][idx])

for idx, data in enumerate(coco_train['annotations']):
    if data['image_id'] not in vg_test_ids:
        new_coco_train['annotations'].append(coco_train['annotations'][idx])   

for idx, data in enumerate(coco_test['images']):
    if data['id'] not in vg_train_ids:
        new_coco_test['images'].append(coco_test['images'][idx])

for idx, data in enumerate(coco_test['annotations']):
    if data['image_id'] not in vg_train_ids:
        new_coco_test['annotations'].append(coco_test['annotations'][idx]) 

with open('/h/skhandel/FewshotDetection/WSASOD/data/data_utils/data/MSCOCO2017/annotations/instances_train2017_clipped.json', 'w') as f:
    json.dump(new_coco_train, f)

with open('/h/skhandel/FewshotDetection/WSASOD/data/data_utils/data/MSCOCO2017/annotations/instances_val2017_clipped.json', 'w') as f:
    json.dump(new_coco_test, f)

with open('/h/skhandel/SceneGraph/scripts/coco_ids_to_remove.pkl', 'wb') as f:
    pickle.dump(coco_test_ids, f)

import ipdb; ipdb.set_trace()
a = 1