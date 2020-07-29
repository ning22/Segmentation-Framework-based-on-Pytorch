import argparse
from pathlib import Path
from pprint import pprint
import torch, os
from matplotlib import pyplot as plt 

# import src.segmentation_models_pytorch as smp
# from src.models.seg_hrnet import get_hrseg_model
from src.loaders.hiatus_dataset import make_data
from src.utils import get_config, set_global_seeds
from src.callbacks import create_callbacks 
from src.trainer import Trainer
from src.models.segnet import segnet
from src.models.seg_hrnet import get_hrseg_model
from src.models.unet.model import Unet 
# import pdb


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Hiatus Segmentation')
    parser.add_argument('--netname', type=str, default='segnet', help='default network name') 
    parser.add_argument('--config_dir', type=str, default='/home/nzhao/Documents/PJ_Pelvis/main/configs/segnet.yaml')
    return parser.parse_args()


def get_model(model_name, config):
    if model_name == 'unet_resnet34':
        model = Unet("resnet34", encoder_weights="imagenet", activation=None)
    elif model_name == 'hrnet':
        model = get_hrseg_model(config)
    elif model_name == 'segnet':
        model = segnet(pretrained=True)
    return model


def main():
    args = parse_args()  

    for args.netname in ['unet_resnet34', 'segnet', 'hrnet']:
        if args.netname == 'segnet':
            args.config_dir = '/home/nzhao/Documents/PJ_Pelvis/main/configs/segnet.yaml'
        elif args.netname == 'hrnet':
            args.config_dir = '/home/nzhao/Documents/PJ_Pelvis/main/configs/seg_hrnet.yaml'
        else:
            args.config_dir = '/home/nzhao/Documents/PJ_Pelvis/main/configs/unet_resnet34.yaml'

        config = get_config(args.config_dir)
        set_global_seeds(456)

        netname = args.netname
        netname = f'{args.netname}/split'
        log_save_dir = Path(config['dumps']['path']) / config['dumps']['logs'] / netname 
        model_save_dir = Path(config['dumps']['path']) / config['dumps']['weights'] / netname 
        os.makedirs(log_save_dir, exist_ok=True)
        os.makedirs(model_save_dir, exist_ok=True)
        config['train_params']['netname'] = netname
        config['train_params']['model_path'] = model_save_dir
        pprint(config) 
        # if os.path.isfile(os.path.join(model_save_dir, 'model.pth')):
        #     config['MODEL']['PRETRAINED'] = os.path.join(model_save_dir, 'model.pth')
        
        model = get_model(args.netname, config)
        data_loaders = make_data(**config['data_params']) 
        callbacks = create_callbacks(netname, config['dumps'])
        model_trainer = Trainer(model, callbacks, data_loaders, **config['train_params'])
        model_trainer.start()
        
        # PLOT TRAINING
        losses = model_trainer.losses
        dice_scores = model_trainer.dice_scores # overall dice
        iou_scores = model_trainer.iou_scores

        def plot(scores, name=args.netname, is_save=True):
            plt.figure(figsize=(15,5))
            plt.plot(range(len(scores["train"])), scores["train"], label=f'train {name}')
            plt.plot(range(len(scores["train"])), scores["val"], label=f'val {name}')
            plt.title(f'{name} plot'); plt.xlabel('Epoch'); plt.ylabel(f'{name}')
            plt.legend()
            # plt.show()
            if is_save:
                plt.savefig(os.path.join(log_save_dir, name+'.png'))

        plot(losses, "Loss")
        plot(dice_scores, "Dice score")
        plot(iou_scores, "IoU score")



if __name__ == '__main__':
    main()