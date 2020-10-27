"""
Ningning Zhao
buaazhaonn@gmail.com
"""
import argparse, os, torch
from collections import defaultdict
from glob import glob
from pathlib import Path
from src.utils import get_config
from src.models.segnet import segnet
from src.models.seg_hrnet import get_hrseg_model
from src.models.unet.model import Unet 
from src.loaders.hiatus_dataset import make_data
from src.inference import PytorchInference
from matplotlib import pyplot as plt 
import pdb

def parse_args():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Segmentation framework based on Pytorch')
    parser.add_argument('--netname', type=str, default='unet_resnet34', help='default network name') 
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='configuration file') 
    parser.add_argument('--data_path', type=str, default='location of your data')
    parser.add_argument('--output_path', type=str, default='location of your output')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--patientID', type=str, default='patientID used for testing (demo)')
    return parser.parse_args()


def get_model(model_name, config_file):
    if model_name == 'unet_resnet34':
        model = Unet("resnet34", encoder_weights="imagenet", activation=None)
    elif model_name == 'hrnet':
        config = get_config(config_file)
        model = get_hrseg_model(config)
    elif model_name == 'segnet':
        model = segnet(pretrained=True)
    return model


def collect_weight_paths(path):
    return os.path.join(path, 'epoch_39.pth')


def cal_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    netname = f'{args.netname}/split'
    output_dir = os.path.join(args.output_path, netname)
    os.makedirs(output_dir, exist_ok=True)

    model_save_dir = Path('./dumps') / 'weights' / 'RGB' /netname 
    weights_paths = collect_weight_paths(model_save_dir)
    model = get_model(args.netname, args.config)
    model.load_state_dict(torch.load(weights_paths))

    test_loader = make_data(args.data_path)['test']
    runner = PytorchInference()
    runner.predict(model, test_loader, output_dir, args.patientID)


if __name__ == '__main__':
    main()
