import os
import cv2
import numpy as np
from glob import glob 
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from operator import itemgetter

# from albumentations import (HorizontalFlip, VerticalFlip, RandomRotate90, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, 
#     Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue, 
#     IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, Resize,
#     IAASharpen, IAAEmboss, RandomBrightnessContrast, ElasticTransform, Flip, OneOf, Compose 
# )
# IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']
# def is_image_file(filename):
#     filename_lower = filename.lower()
#     return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)

class HiatusDataset(Dataset):
    def __init__(self, img_filenames, mask_filenames, transform=None):
        self.img_filenames = img_filenames
        self.mask_filenames = mask_filenames
        self.transform = transform

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, index):
        image = cv2.imread(self.img_filenames[index], 1)
        sample = {'image': image}

        mask = cv2.imread(self.mask_filenames[index], 0)
        mask = mask/mask.max()
        sample['mask'] = mask[...,np.newaxis]

        if self.transform is not None:
            sample = self.transform(**sample)
        sample['image_id'] = os.path.basename(self.img_filenames[index]).strip('_IMG.png')
        return sample


def make_data_loader(dataset, batch_size, num_workers, shuffle, sampler=None):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers, pin_memory=False)


def make_data(data_folder, fold):
    train_list, test_list = train_test_cv_split(data_folder, fold, num_folds=4, shuffle=True)
    train_transform = strong_aug()
    train_dataset = HiatusDataset(
        img_filenames=train_list[0], mask_filenames=train_list[1], transform=train_transform 
    )
    train_loader = make_data_loader(dataset=train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)

    test_dataset = HiatusDataset(img_filenames=test_list[0], mask_filenames=test_list[1], transform=None) 
    
    test_loader = make_data_loader(dataset=test_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    return {'train': train_loader, 'val':test_loader}