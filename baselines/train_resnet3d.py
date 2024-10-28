#%%
import torch
import json
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import numpy as np

# Load and transform the dataset
import os, sys
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

device = sys.argv[1]
task = sys.argv[2]
predict_mode = sys.argv[3] == 'predict'
print('Prediction mode:', predict_mode)

#%%
phases = ['train', 'val', 'test']

phase_ids = { ph: np.loadtxt(f'../splits/{ph}_wbu.txt', dtype=int) for ph in phases }
#%%

class VolumeDataset(Dataset):
    def __init__(self, task, root_dir, split_ids, stats=None):
        self.root_dir = root_dir
        split_ids = { i: True for i in split_ids}
        # self.transform = transform
        # existing_images = list(sorted([i for i in os.listdir(root_dir) if i.endswith('_20253_2_0-top.jpg')]))
        with open(root_dir + '.txt') as fl:
            existing_images = [ln.strip().split('/')[1] for ln in fl]

        if task == 'idps':
            self.ldf = pd.read_csv('../../phenotypes/brain_biomarkers.csv').set_index('FID')
        else:
            self.ldf = pd.read_csv('../../phenotypes/brain_related_match20253_f204.csv').set_index('FID')

        self.labels = self.ldf
        self.labels = self.labels.loc[self.labels.index.intersection(split_ids.keys())]
        if stats is None:
            self.stats = (self.labels.mean(0), self.labels.std(0))
        else:
            self.stats = stats
        self.labels -= self.stats[0]
        self.labels /= self.stats[1]
        self.labels = { id: v for id, v in zip(self.labels.index, self.labels.values) }

        existing_ids = { int(i.split('_')[0]): True for i in existing_images }
        self.ids = [id for id in self.labels if id in existing_ids and id in split_ids]

        print('Found :', len(existing_images))
        print('Loaded:', len(self.ids))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        base_name = self.ids[index]

        bvol = np.load(f'/u/scratch/u/ulzee/brainsplat/data/volumes/{base_name}.npz')['arr_0']
        bvol = bvol.astype(np.float32) / 256

        lvec = self.labels[base_name]
        nanmask = np.isnan(lvec)
        return bvol, nanmask, lvec

trainset = VolumeDataset(task, '/u/scratch/u/ulzee/brainsplat/data/images/20253', phase_ids['train'])
dsets = dict(
    train=trainset,
    val=VolumeDataset(task, '/u/scratch/u/ulzee/brainsplat/data/images/20253', phase_ids['val'], stats=trainset.stats),
    test=VolumeDataset(task, '/u/scratch/u/ulzee/brainsplat/data/images/20253', phase_ids['test'], stats=trainset.stats),
)
#%%

# Define the model
from models import resnet3ds18 as Resnet3d
resnet_name = 'resnet3ds18'

sample = trainset[0]
net = Resnet3d(num_classes=len(sample[-1])).to(device)
if predict_mode:
    net.load_state_dict(torch.load(f'saved/{task}/{resnet_name}_best.pth'))
#%%

loaders = { ph: torch.utils.data.DataLoader(
                ds, batch_size=32,
                shuffle=ph == 'train', num_workers=2) for ph, ds in dsets.items() }

# Define the loss function and optimizer
import torch.optim as optim

criterion = nn.MSELoss()
optimizer = optim.AdamW(
    net.parameters(),
    lr=1e-4,
    weight_decay=0.01)
# Train the network
hist = []
best_val_loss = float('inf')
best_model_state = None

preds = []
for epoch in range(100 if not predict_mode else 1):  # loop over the dataset multiple times

    byphase = []
    for ph, loader in loaders.items():
        if predict_mode:
            if ph in ['train', 'val']: continue
        net.train() if ph == 'train' else net.eval()
        pbar = tqdm(loader)

        stats = []
        for i, data in enumerate(pbar):
            # get the inputs; data is a list of [inputs, labels]
            labels = data[-1].to(device).float()
            obsmask = ~data[-2].to(device).bool()
            inputs = data[0].to(device).unsqueeze(1)

            # zero the parameter gradients
            if ph == 'train': optimizer.zero_grad()

            # forward + backward + optimize
            if ph == 'train':
                outputs = net(inputs)
                loss = criterion(outputs[obsmask], labels[obsmask])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                optimizer.step()
            else:
                with torch.no_grad():
                    outputs = net(inputs)
                    loss = criterion(outputs[obsmask], labels[obsmask])

                    if predict_mode:
                        preds += [(outputs.cpu().numpy(), labels.cpu().numpy())]

            pbar.set_postfix(dict(e=epoch, p=ph, ls='%.4f'%loss.item()))
            stats += [loss.item()]

        byphase += [stats]

        # Check if validation loss improves
        if ph == 'val' and sum(stats) / len(stats) < best_val_loss:
            best_val_loss = sum(stats) / len(stats)
            best_model_state = net.state_dict()

    hist += [byphase]


    if not predict_mode:
        # Save the model if validation loss improves
        if best_model_state is not None:
            torch.save(best_model_state, f'saved/{task}/{resnet_name}_best.pth')

        with open(f'saved/{task}/{resnet_name}_hist.json', 'w') as fl:
            json.dump(hist, fl)
print('Finished Training')
#%%
if predict_mode:
    preds, labs = [np.concatenate(t) for t in zip(*preds)]

    for colvals, saveto in zip([preds, labs], [f'saved/{task}/predictions_test_{resnet_name}.csv']):
        dfdict = dict(ids=dsets['test'].ids)
        for ci, (cname, col) in enumerate(zip(trainset.ldf.columns, colvals.T)):
            dfdict[cname] = (col*trainset.stats[1].values[ci]) + col*trainset.stats[0].values[ci]
        pd.DataFrame(dfdict).set_index('ids').to_csv(saveto)
