from collections import defaultdict
from glob import glob
import argparse, os, torch
from tqdm import tqdm
from src.utils import get_config
from src.inference import PytorchInference


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type='/home/nzhao/Documents/PJ_Pelvis/main/data/hiatus')
    parser.add_argument('--output_path', type='/home/nzhao/Documents/PJ_Pelvis/main/data/hiatus_output')
    parser.add_argument('--batch_size', type='/home/nzhao/Documents/PJ_Pelvis/main/data/hiatus_output')
    parser.add_argument('--num_workers', type='/home/nzhao/Documents/PJ_Pelvis/main/data/hiatus_output')
    return parser.parse_args()

def collect_weight_paths(path, fold):
    return sorted(glob(os.path.join(path, f'fold{fold}', '*.pth')))

def main():
    args = parse_args()
    config = get_config(args.config)
    device = 'cpu'
    runner = PytorchInference(device)

    test_loader = make_data(**config['data_params'], mode='test')
    weights_paths = collect_weight_paths(args.path, args.fold)

    os.makedirs(args.output, exist_ok=True)
    predictions = defaultdict(lambda: {'mask': 0, 'empty': 0})
    for i, weights_path in enumerate(weights_paths):
        torch.cuda.empty_cache()
        print(f'weights: {weights_path}')
        config['train_params']['weights'] = weights_path
        factory = Factory(config['train_params'])
        model = factory.make_model(device)
        for result in tqdm(
            iterable=runner.predict(model, test_loader), total=len(test_loader) * config['data_params']['batch_size']
        ):
            ensemble_mask = (predictions[result['image_id']]['mask'] * i + result['mask']) / (i + 1)
            ensemble_empty = (predictions[result['image_id']]['empty'] * i + result['empty']) / (i + 1)
            predictions[result['image_id']]['mask'] = ensemble_mask
            predictions[result['image_id']]['empty'] = ensemble_empty
    mmcv.dump(dict(predictions), osp.join(args.output, f'fold_{args.fold}.pkl'))


if __name__ == '__main__':
    main()