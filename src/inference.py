import numpy as np
import torch
import os
import pandas as pd 
from time import time
import cv2
from matplotlib import pyplot as plt 
import matplotlib.patches as mpatches
from src.trainer import Meter
import pdb

def predict(X, threshold=0.5):
    X_p = np.copy(X)
    preds = (X_p > threshold).astype('uint8')
    return preds

def plot_image(images, mets):
    im, tr, pr = images['image'], images['truth'], images['pred']
    _, thresh = cv2.threshold(tr, 100, 255, cv2.THRESH_BINARY)
    contours_tr, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    _, thresh = cv2.threshold(pr, 100, 255, cv2.THRESH_BINARY)
    contours_pr, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(im, contours_tr, -1, (255,0,0), 3)
    cv2.drawContours(im, contours_pr, -1, (0,255,0), 3)

    plt.imshow(im)
    plt.title('ID: %s, DICE: %.4f, IOU: %.4f' %(mets[0], mets[1], mets[2]))
    plt.axis('off')
    plt.tight_layout()
    cmap = {1:[1,0,0], 2:[0,1,0]}
    labels = {1:'truth', 2:'pred'}
    patches =[mpatches.Patch(color=cmap[i],label=labels[i]) for i in cmap]
    plt.legend(handles=patches, loc=4, borderaxespad=0.)
    plt.show()

def save_images(images, mets, output_filename, is_show=False):
    fig = plt.figure(figsize=(10,5))
    for ind, title in enumerate(images):
        # ax = fig.add_subplot(1, 3, ind+1)
        # plt.imshow(images[title])
        # ax.set_title(title)
        # ax.set_axis_off()
        plt.subplot(1,3,ind+1)
        plt.imshow(images[title])
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    fig.suptitle('ID: %s, DICE: %.4f, IOU: %.4f' %(mets[0], mets[1], mets[2]))
    fig.savefig(output_filename)
    plt.close()

    pardir = os.path.join(output_filename, os.pardir) 
    image_dir = os.path.abspath(os.path.join(pardir, 'images'))
    os.makedirs(image_dir, exist_ok=True)
    for ind, title in enumerate(images):
        cv2.imwrite(os.path.join(image_dir, mets[0]+'_'+title+'.png'), images[title])
        fig = plt.figure(figsize=(4,4))
        plt.imshow(images[title])
        plt.title(title)
        plt.axis('off')
        plt.margins(0)
        fig.savefig(os.path.join(image_dir, mets[0]+'_'+title+'.png'))
        plt.close()   
    if is_show:
        plt.show()
        

def save2xlsx(filename, results_dict):
    df = pd.DataFrame(results_dict, columns = ['PatientID', 'Dice', 'IoU', 'Time(s)'])
    df.to_excel(filename, index = False, header=True)



class PytorchInference(object):
    def __init__(self, device='cpu'):
        self.device = device

    @staticmethod
    def to_numpy(images):
        return images.data.cpu().numpy()
    
    def resize_image(self, image, h, w):
        res = cv2.resize(image, dsize=(h, w), interpolation=cv2.INTER_CUBIC)
        return res

    def predict(self, model, loader, output_dir, pID, is_save=False):
        model = model.to(self.device).eval()
        pids, dices, ious, durations = [], [], [], []
        with torch.no_grad():
            print('Load image...')
            for data in loader:
                image_ids, images, targets, original_size = data['image_id'], data['image'], data['mask'], data['original_size']
                if image_ids[0] != pID: continue
                print('Start predict...')
                start_time = time()
                images_torch = images.type(torch.FloatTensor).permute(0,3,1,2)
                targets_torch = targets.type(torch.FloatTensor).permute(0,3,1,2)
                outputs = model(images_torch)  
                seconds_elapsed = time() - start_time
                metrics = Meter()             
                metrics.update(targets_torch.detach().cpu(), outputs.detach().cpu())
                dice, iou = metrics.get_metrics()
                print('ID: %s, DICE: %.4f, IOU: %.4f' %(image_ids[0], dice, iou))
                
                pids.append(image_ids[0])
                dices.append(dice)
                ious.append(iou)
                durations.append(seconds_elapsed)

                probs = torch.sigmoid(outputs)
                preds = self.to_numpy(probs)
                preds = (preds.squeeze() > 0.5)
                preds = preds.astype('uint8')
                pred = preds*255
                
                image = self.to_numpy(images).squeeze().astype('uint8')
                truth = self.to_numpy(targets).squeeze().astype('uint8')*255
                ori_w, ori_h = self.to_numpy(original_size[0])[0], self.to_numpy(original_size[1])[0]

                im = self.resize_image(image, ori_h, ori_w)
                tr = self.resize_image(truth, ori_h, ori_w)
                pr = self.resize_image(pred, ori_h, ori_w)
                ims = {'image': im, 'truth':tr, 'pred':pr} 

                print('Dispaly...')
                plot_image(ims, [image_ids[0], dice, iou])
                #TODO: calculate the metrics for resized images
                # print('***re***ID: %s, DICE: %.4f, IOU: %.4f' %(image_ids[0], dice, iou))
                if is_save:               
                    save_images(ims, [image_ids[0], dice, iou], os.path.join(output_dir, image_ids[0]+'.png'))

        # return [image, truth, pred]
                
        filename = os.path.join(os.path.join(output_dir, '..'), 'split.xlsx')
        filename = os.path.abspath(filename)
        results = {'PatientID': pids, 'Dice': dices, 'IoU': ious, 'Time(s)': durations}
        save2xlsx(filename, results)        

                
