"""
Ningning Zhao
buaazhaonn@gmail.com
"""
import os
from glob import glob 
from sklearn.model_selection import KFold
from operator import itemgetter
import pdb

_NUM_FOLDS_ = 4

def write2txt(output_filename, image_list, mask_list):
    filename_list = []
    for idx, (image_name, mask_name) in enumerate(zip(image_list, mask_list)):
        image_id = os.path.basename(image_name).strip('_IMG.png')
        assert image_id == os.path.basename(mask_name).strip('_LBL.png')
        filename_list.append(image_id + ',' + image_name + ',' + mask_name)

    f = open(output_filename, "w")
    for line in filename_list:
        f.write(line)
        f.write("\n")
    f.close()

def read_txt(filename):
    with open(filename, "r") as f:
        data_list = [line.rstrip() for line in f.readlines()]
    f.close() 
    return data_list

def write2txt_v2(id_list_filename, txt_filename, image_list, mask_list):
    id_list = read_txt(id_list_filename)
    filename_list = []
    for idx, (image_name, mask_name) in enumerate(zip(image_list, mask_list)):
        image_id = os.path.basename(image_name).strip('_IMG.png')
        if image_id in id_list:
            assert image_id == os.path.basename(mask_name).strip('_LBL.png')
            filename_list.append(image_id + ',' + image_name + ',' + mask_name)

    f = open(txt_filename, "w")
    for line in filename_list:
        f.write(line)
        f.write("\n")
    f.close()

if __name__ == "__main__":
    output_path = 'data/data_name'
    os.makedirs(output_path, exist_ok=True)
    is_cross_validation = True


    if not is_cross_validation:
        for stage in ['train', 'test']:
            data_folder = 'location of raw data' + stage
        output_path = 'location of prepared data'
            os.makedirs(output_path, exist_ok=True)

            image_list = sorted(glob(os.path.join(data_folder, '*_IMG*')))
            label_list = sorted(glob(os.path.join(data_folder, '*_LBL*'))) 

            txt_filename = os.path.join(output_path, stage +'_list.txt')
            write2txt(txt_filename, image_list, label_list)
    else:
        # cross_validation
        data_folder = 'location of raw data'
        output_path = 'location of prepared data'
        os.makedirs(output_path, exist_ok=True)

        _NUM_FOLDS_ = 4 
        image_list = sorted(glob(os.path.join(data_folder, '*_IMG*')))
        label_list = sorted(glob(os.path.join(data_folder, '*_LBL*')))

        kf = KFold(n_splits=_NUM_FOLDS_, random_state=42, shuffle=True) 
        for fold in range(_NUM_FOLDS_):
            for i, (train_idx, test_idx) in enumerate(kf.split(range(len(image_list)))):
                if i == fold:
                    train_index = train_idx
                    test_index = test_idx
                    break
            train_image_list = list(itemgetter(*train_index)(image_list)) 
            test_image_list = list(itemgetter(*test_index)(image_list)) 
            train_mask_list = list(itemgetter(*train_index)(label_list)) 
            test_mask_list = list(itemgetter(*test_index)(label_list)) 

            train_filename = os.path.join(output_path, 'f'+str(fold)+'_train_list.txt')
            test_file_name = os.path.join(output_path, 'f'+str(fold)+'_test_list.txt')
            write2txt(train_filename, train_image_list, train_mask_list)
            write2txt(test_file_name, test_image_list, test_mask_list)
        


