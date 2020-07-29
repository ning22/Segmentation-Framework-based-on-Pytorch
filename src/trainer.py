import time, os, pdb 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
from src.losses.losses import MixedLoss
import pdb

def predict(X, threshold=0.5):
    X_p = np.copy(X)
    preds = (X_p > threshold).astype('uint8')
    return preds
    
def compute_dice(probability, truth, threshold=0.5):
    batch_size = len(truth)
    with torch.no_grad():
        probability = probability.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert(probability.shape == truth.shape)

        p = (probability > threshold).float()
        t = (truth > 0.5).float()
        dice = 2 * (p*t).sum(-1)/((p+t).sum(-1))
    return dice 

def compute_ious(pred, label, classes, only_present=True):
    '''computes iou for one ground truth mask and predicted mask'''
    ious = []
    for c in classes:
        label_c = np.squeeze(label == c)
        pred_c = np.squeeze(pred == c)
        intersection = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        if union != 0:
            ious.append(intersection / union)
    return ious if ious else [1]


def compute_iou_batch(outputs, labels, classes=None):
    '''computes mean iou for a batch of ground truth masks and predicted masks'''
    ious = []
    preds = np.copy(outputs) # copy is imp
    labels = np.array(labels) # tensor to np
    for pred, label in zip(preds, labels):
        ious.append(np.nanmean(compute_ious(pred, label, classes)))
    iou = np.nanmean(ious)
    return iou


class Meter:
    '''A meter to keep track of iou and dice scores throughout an epoch'''
    def __init__(self):
        self.base_threshold = 0.5 # <<<<<<<<<<< here's the threshold
        self.base_dice_scores = []
        self.iou_scores = []
        # self.train_metrics = {}
        # self.val_metrics = {}

    def update(self, targets, outputs):
        probs = torch.sigmoid(outputs)
        dice = compute_dice(probs, targets, self.base_threshold)
        self.base_dice_scores.extend(dice)
        preds = predict(probs, self.base_threshold)
        iou = compute_iou_batch(preds, targets, classes=[1])
        self.iou_scores.append(iou)

    def get_metrics(self):
        dice = np.nanmean(self.base_dice_scores)
        iou = np.nanmean(self.iou_scores)
        return dice, iou


class Trainer(object):
    '''This class takes care of training and validation of our model'''
    def __init__(self, model, callbacks, data_loaders, netname, num_folds, fold, num_workers, batch_size, lr, num_epochs, gpu_id, model_path):
        self.net = model
        self.netname = netname 
        self.fold = fold
        self.total_folds = num_folds
        self.num_workers = num_workers
        self.save_path = model_path
        self.batch_size = {"train": batch_size, "val": batch_size}
        self.accumulation_steps = 32 // self.batch_size['train']
        self.lr = float(lr)
        self.num_epochs = num_epochs
        # self.params = {'name': netname, 'num_folds': num_folds, 'lr': lr, 'batch_size': batch_size, 'num_epochs': num_epochs}
        self.params = {'name': netname, 'lr': lr, 'batch_size': batch_size, 'num_epochs': num_epochs}

        # self.best_loss = float("inf")
        self.best_loss = 0 # dice coefficient
        self.phases = ["train", "val"]
        self.device = torch.device("cuda:" + str(gpu_id))
        # torch.set_default_tensor_type("torch.cuda.FloatTensor")
        
        self.criterion = MixedLoss(10.0, 50.0)  # 10.0, 2.0
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=3)
        # self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=5, verbose=True, factor=0.8)
        self.net = self.net.to(self.device)
        cudnn.benchmark = True
        self.dataloaders = data_loaders
        self.losses = {phase: [] for phase in self.phases}
        self.iou_scores = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}
        # self._metrics = None
        self.callbacks = callbacks
        if callbacks is not None:
            self.callbacks.set_runner(self)
        self.callbacks = callbacks
        self.metrics=Meter()
    
    @property
    def model(self):
        return self.net

    @property
    def loss(self):
        return self.criterion

    # @property
    # def metrics(self):
    #     if self._metrics is None:
    #         self._metrics = self.make_metrics()
    #     return self._metrics

    # def make_metrics(self):
    #     for metric, params in self.params['metrics'].items() :
    #         return Meter(metric.split('.')[-1])

    def epoch_log(self, phase, epoch, epoch_loss, start):
        '''logging the metrics at the end of an epoch'''
        dice, iou = self.metrics.get_metrics()
        print("Loss: %0.4f | dice: %0.4f | IoU: %0.4f" % (epoch_loss, dice, iou))
        return dice, iou

    def forward(self, images, targets):
        images = images.to(self.device)
        targets = targets.to(self.device)
        outputs = self.net(images)
        loss = self.criterion(outputs, targets)
        return loss, outputs

    def iterate(self, epoch, phase):
        is_train = True if phase == 'train' else False
        # meter = Meter()
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")
        # batch_size = self.batch_size[phase]
        # self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
        with torch.set_grad_enabled(is_train):
            self.optimizer.zero_grad()
            for itr, batch in enumerate(dataloader):
                self.callbacks.on_batch_begin(itr)
                images, targets = batch['image'], batch['mask']
                if images.shape[-1] == 3 and targets.shape[-1]==1:
                    images = images.type(torch.cuda.FloatTensor).permute(0,3,1,2)
                    targets = targets.permute(0,3,1,2)
                else:
                    images = images.type(torch.cuda.FloatTensor)
                    targets = targets
                loss, outputs = self.forward(images, targets)
                loss = loss / self.accumulation_steps
                if phase == "train":
                    loss.backward()
                    if (itr + 1 ) % self.accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                running_loss += loss.item()
                self.metrics.update(targets.detach().cpu(), outputs.detach().cpu())
                dice, iou = self.metrics.get_metrics()
                self.callbacks.on_batch_end(itr, step_report={'loss':loss, 'dice':dice, 'iou':iou}, is_train=is_train)
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        dice, iou = self.epoch_log(phase, epoch, epoch_loss, start)
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        self.iou_scores[phase].append(iou)
        torch.cuda.empty_cache()
        del loss, outputs, batch
        return dice

    def start(self):
        self.callbacks.on_train_begin()
        for epoch in range(self.num_epochs):
            self.callbacks.on_epoch_begin(epoch)
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            
            val_loss = self.iterate(epoch, "val")
            self.scheduler.step(val_loss)
            if val_loss > self.best_loss: # val_loss if the dice coefficient here
                print("******** New optimal found, saving state ********")
                state["best_loss"] = self.best_loss = val_loss
                torch.save(state, os.path.join(self.save_path, "model.pth"))
            print()
            epoch_info = {'loss':self.losses, 'dice':self.dice_scores, 'iou':self.iou_scores}
            self.callbacks.on_epoch_end(epoch, epoch_info)
        self.callbacks.on_train_end()