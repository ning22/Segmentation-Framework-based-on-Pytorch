from albumentations import (HorizontalFlip, VerticalFlip, RandomRotate90, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, 
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue, 
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, Resize,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, ElasticTransform, Flip, OneOf, Compose 
)

def transform(p=1):
    aug = Compose([
        RandomRotate90(0.5),
        HueSaturationValue(hue_shift_limit=(-25,0),sat_shift_limit=0,val_shift_limit=0,p=0),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.1),
        OneOf([
            MedianBlur(blur_limit=3, p=0),
            Blur(blur_limit=3, p=0.1),
        ], p=0.1),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2,0.2),p=0.3),
        ], p=0.3),
        OneOf([
            ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            GridDistortion(p=0.5),
            OpticalDistortion(p=1, distort_limit=1, shift_limit=0.5)                  
        ], p=0.2),
        Resize(512,512)
    ], p=p)
    return aug 