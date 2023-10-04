import json
from queue import Queue
from threading import Thread, Lock


class Graph:
    def __init__(self, parent_map, children_map, id_list):
        self.parent_map = parent_map
        self.children_map = children_map
        self.id_list = id_list

    # return child-nodes first --> start with nodes that have no children
    def topoligical_sort(self):
        order = []
        next = []
        # start at root nodes, go down in the tree, then reverse the ordering
        for id in self.id_list:
            if len(self.parent_map[id]) == 0 and id not in order:
                order.append(id)
                for child_id in self.children_map[id]:
                    if child_id not in next and child_id not in order:
                        next.append(child_id)
        while len(next) != 0:
            id = next.pop(0)
            order.append(id)
            for child_id in self.children_map[id]:
                if child_id not in next and child_id not in order:
                    next.append(child_id)
        order.reverse()
        return order


def to_x1y1x2y2(obj):
    x1 = obj['x']
    y1 = obj['y']
    x2 = obj['x'] + obj['w']
    y2 = obj['y'] + obj['h']
    return [x1, y1, x2, y2]


def to_segmentation(obj):
    l = to_x1y1x2y2(obj)
    x1 = l[0]
    y1 = l[1]
    x2 = l[2]
    y2 = l[3]
    return [x1, y1, x2, y1, x2, y2, x1, y2]  # order from arxiv-docs dataset file
    # return [x1, y1, x1, y2, x2, y2, x2, y1]  # order from coco.py file


def preprocessRelations(object_data, relations_data):
    # Return two dicts (one with parent-of, one with child-of maps)
    parent_map = {}
    children_map = {}
    bbox_map = {}
    id_list = []
    image_ids = []
    for object in object_data["objects"]:
        obj_id = object["object_id"]
        parent_map[obj_id] = []
        children_map[obj_id] = []
        id_list.append(obj_id)
        object['image_id'] = object_data['image_id']
        bbox_map[obj_id] = object

        if object_data['image_id'] not in image_ids:
            image_ids.append(object_data['image_id'])

    # NOTE: Adapt this for-loop to extend relationships or adapt to hierarchical relationships in other datasets.

    for relation in relations_data['relationships']:
        if relation["predicate"] == "parent_of":
            parent_id = relation['subject']["object_id"]
            child_id = relation['object']["object_id"]
            parent_map[child_id].append(parent_id)
            children_map[parent_id].append(child_id)

            if parent_id not in id_list:
                id_list.append(parent_id)
            if child_id not in id_list:
                id_list.append(child_id)
        else:
            continue

    # construct orders within the images
    obj_ids_in_image = [x['object_id'] for x in object_data['objects']]
    G = Graph(parent_map, children_map, obj_ids_in_image)
    id_order = G.topoligical_sort()

    return children_map, parent_map, bbox_map, id_order, image_ids


def create_segmentation_file(img_subdir, anns_subdir, output_subdir, dataset_descriptor, metadata_input,
                             object_input, relationship_input, attribute_synsets_input, attribute_dict_file_dir,
                             attribute_dict_file_path, output_json_path, num_workers=20):
    obj_data = json.load(open(object_input))
    rel_data = json.load(open(relationship_input))
    # img_data = json.load(open(metadata_input))
    attr_data = json.load(open(attribute_dict_file_path))

    # Note: python dictionary should be thread-safe for atomic operations
    #   Worker only writes to one location at a time (the image-id they are working on)
    #   Workers should be safely able to read from the shared dictionaries

    assert len(obj_data) == len(
        rel_data), f"Expected #objects = #relations, got: {len(obj_data)} objects, {len(rel_data)} relations"
    n_images = len(obj_data)

    # Map[image_id] -> Map[object_id] -> object{'segmentation', 'orig_object_id', 'object_id', 'image_id', 'names'}
    segm_map = {}

    idxs = list(range(n_images))
    lock = Lock()
    q = Queue()
    for i, idx in enumerate(idxs):
        q.put((i, idx))

    def worker():
        while True:
            i, idx = q.get()

            if i % 100 == 0:
                print('[Segmentation-Generation] processed %i images...' % i)

            image_id = obj_data[idx]["image_id"]
            assert image_id == rel_data[idx][
                "image_id"], f"Expected ordered objects and relationships, mismatch at index {idx}"

            segm_map_entry = {}
            children_map, parent_map, bbox_map, id_list, _ = preprocessRelations(obj_data[idx], rel_data[idx])

            for id in id_list:
                image_id = bbox_map[id]['image_id']
                name = bbox_map[id]['names'][0]
                category_id = attr_data['label_to_idx'][name.replace("_", "").lower()]
                if len(bbox_map[id]['names']) != 1:
                    print(f"Object {id} has multiple names: {bbox_map[id]}")
                segm_map_entry[id] = {
                    'segmentation': [],
                    'object_id': id,
                    'category_id': category_id,
                    'image_id': bbox_map[id]['image_id'],
                    'names': bbox_map[id]['names']
                }

                if len(children_map[id]) == 0:
                    bboxSegm = to_segmentation(bbox_map[id])
                    segm_map_entry[id]['segmentation'] = [bboxSegm]

                else:
                    for child_id in children_map[id]:
                        if child_id not in segm_map_entry:
                            print(f"Object {child_id} not in segmentation map")
                            continue
                        segm_map_entry[id]['segmentation'] = segm_map_entry[id]['segmentation'] + \
                                                                 segm_map_entry[child_id]['segmentation']

            # generate the 'empty_index' list for current image
            obj_ids = []
            empty_index_entry = {}
            for id in segm_map_entry:
                obj_ids.append(segm_map_entry[id])
            # order list by 'object_id' for it to correspond to the order in the attributes file
            obj_ids.sort(key=lambda x: x['object_id'])
            empty_index_entry['empty_index'] = obj_ids

            # Write down entry
            lock.acquire()
            segm_map[image_id] = empty_index_entry
            lock.release()
            q.task_done()

    for i in range(num_workers):
        t = Thread(target=worker)
        t.daemon = True
        t.start()

    q.join()

    # save
    with open(output_json_path, 'w') as obj_out_file:
        json.dump(segm_map, obj_out_file)

    print("constructed segmentations")
