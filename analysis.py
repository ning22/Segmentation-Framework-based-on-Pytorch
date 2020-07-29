import pandas as pd 
import numpy as np 
import os

def read_xlsx(network, filename):
    df = pd.read_excel(filename)
    ids, dices, ious, times = df['PatientID'], df['Dice'], df['IoU'], df['Time(s)']
    dice_max = dices.max()
    dice_min = dices.min()
    print('Best case %s' %(ids[list(dices).index(dice_max)]))
    print('Worst case %s' %(ids[list(dices).index(dice_min)]))
    dice_mean = np.mean(np.array(dices))
    iou_mean = np.mean(np.array(ious))
    time_mean = np.mean(np.array(times))

    bad_cases = np.where(dices<=0.9)[0]
    num_bad = len(bad_cases)
    strings = 'network: %s, dice: %.4f, iou: %.4f, max dice: %.4f, min dice: %.4f, time: %.4f, #bad case: %d'
    print( strings %(network, dice_mean, iou_mean, dice_max, dice_min, time_mean, num_bad))
    print('*************************************************************************************************')
    



if __name__ == "__main__":
    output_path = '/home/nzhao/Documents/PJ_Pelvis/main/output/hiatus'
    for netname in ['unet_resnet34', 'segnet', 'hrnet']:
        netname = f'{netname}'
        output_dir = os.path.join(output_path, netname)
        xlsx_filename = os.path.join(output_dir, 'split.xlsx')
        read_xlsx(netname, xlsx_filename)



