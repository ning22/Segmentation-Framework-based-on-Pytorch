import os
import cv2
import numpy as np
from operator import itemgetter
from glob import glob 
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from src.augmentations.aug_albumentations import transform
# from src.augmentations.augmentations_v2 import transform
from albumentations import Resize
from matplotlib import pyplot as plt 
import pdb


class HiatusDataset(Dataset):
    def __init__(self, patient_filename_list, transform=None):
        self.patient_filename_list = patient_filename_list
        self.transform = transform

    def __len__(self):
        return len(self.patient_filename_list)

    def __getitem__(self, index, is_rgb=True):
        image_name = self.patient_filename_list[index].split(',')[1]
        mask_name = self.patient_filename_list[index].split(',')[2]
        # pdb.set_trace()
        image = cv2.imread(image_name, 1)
        if not is_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.repeat(image[...,np.newaxis], 3, axis=-1)
        sample = {'image': image}
        mask = cv2.imread(mask_name, 0)
        mask = mask/mask.max()
        sample['mask'] = mask[...,np.newaxis]

        if self.transform is not None:
            sample = self.transform(**sample)
        else:
            # test_v1_list
            sample['original_size'] = mask.shape
            test_transform = Resize(544, 768)
            sample = test_transform(**sample)
        sample['image_id'] = self.patient_filename_list[index].split(',')[0]
        return sample

def make_data_loader(dataset, batch_size, num_workers, shuffle, sampler=None):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers, pin_memory=True)

def read_txt(filename):
    with open(filename, "r") as f:
        data_list = [line.rstrip() for line in f.readlines()]
    f.close() 
    return data_list

def make_data_fold(data_folder, fold, batch_size=1, num_workers=0):
    train_filename = os.path.join(data_folder, 'f'+str(fold) + '_train_list.txt')
    test_filename = os.path.join(data_folder, 'f'+str(fold) + '_test_list.txt')
    train_list = read_txt(train_filename)
    test_list = read_txt(test_filename)

    train_transform = transform()
    train_dataset = HiatusDataset(train_list, transform=train_transform)
    train_loader = make_data_loader(dataset=train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)

    test_dataset = HiatusDataset(test_list, transform=None)
    val_loader = make_data_loader(dataset=test_dataset, shuffle=False, batch_size=batch_size*2, num_workers=num_workers)
    
    test_loader = make_data_loader(dataset=test_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    return {'train': train_loader, 'val':val_loader, 'test':test_loader}

def make_data(data_folder, batch_size=1, num_workers=0):
    train_filename = os.path.join(data_folder, 'train_list.txt')
    test_filename = os.path.join(data_folder, 'test_v1_list.txt')
    train_list = read_txt(train_filename)
    test_list = read_txt(test_filename)

    train_transform = transform()
    train_dataset = HiatusDataset(train_list, transform=train_transform)
    # tt = train_dataset.__getitem__(10)
    # pdb.set_trace()
    train_loader = make_data_loader(dataset=train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    
    test_dataset = HiatusDataset(test_list, transform=None)
    # test_dataset.__getitem__(10)
    val_loader = make_data_loader(dataset=test_dataset, shuffle=False, batch_size=batch_size*2, num_workers=num_workers)
    
    test_loader = make_data_loader(dataset=test_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    return {'train': train_loader, 'val':val_loader, 'test':test_loader}

# def make_pred_dataset(data_folder, fold=None):
#     if fold is None:
#         test_filename = os.path.join(data_folder, 'test_list.txt')
#         test_list = read_txt(test_filename)

#         test_dataset = HiatusDataset(test_list, transform=None)
#     else:
#         test_filename = os.path.join(data_folder, 'f'+str(fold) + '_test_list.txt')
#         test_list = read_txt(test_filename)

#         test_dataset = HiatusDataset(test_list, transform=None)
#     return test_dataset

