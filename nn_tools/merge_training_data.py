import os
import numpy as np
import shutil
import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

if __name__ == '__main__':
    root_dir = '/public/pangshumao/data/Spark_Challenge/in'
    study_dir_1 = os.path.join(root_dir, 'lumbar_train150')
    study_dir_2 = os.path.join(root_dir, 'lumbar_train51')

    json_path_1 = os.path.join(root_dir, 'lumbar_train150_annotation.json')
    json_path_2 = os.path.join(root_dir, 'lumbar_train51_annotation.json')

    out_json_path = os.path.join(root_dir, 'lumbar_train201_annotation.json')


    merge_train_dir = os.path.join(root_dir, 'lumbar_train201')
    if not os.path.exists(merge_train_dir):
        os.makedirs(merge_train_dir)
        for study in os.listdir(study_dir_1):
            shutil.copytree(os.path.join(study_dir_1, study), os.path.join(merge_train_dir, study))
        for study in os.listdir(study_dir_2):
            shutil.copytree(os.path.join(study_dir_2, study), os.path.join(merge_train_dir, study))

    with open(json_path_1, 'r') as file:
        annotations_1 = json.load(file)

    with open(json_path_2, 'r') as file:
        annotations_2 = json.load(file)

    merge_annotations = []

    for annotation in annotations_1:
        merge_annotations.append(annotation)

    for annotation in annotations_2:
        merge_annotations.append(annotation)


    with open(out_json_path, 'w') as file:
        json.dump(merge_annotations, file, cls=NpEncoder)



