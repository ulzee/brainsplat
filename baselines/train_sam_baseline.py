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
#%%
import argparse

parser = argparse.ArgumentParser(description='Train embedding model')
parser.add_argument('--device', type=str, required=True, help='Device to use (e.g., cuda:0)')
# parser.add_argument('--task', type=str, required=True, help='Task name')
# parser.add_argument('--emb-type', type=str, default=None, help='Embedding type')
# parser.add_argument('--emb-file', type=str, default=None, help='Embedding type')
parser.add_argument('--predict', action='store_true', help='Run in prediction mode')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--lr', type=float, default=1e-5)
# parser.add_argument('--lrstep', type=int, default=5)

args = parser.parse_args()

assert not (args.emb_type is None and args.emb_file is None)

if args.emb_file is not None:
    args.emb_type = 'oct21_vae'

device = args.device
task = args.task
emb_type = args.emb_type
predict_mode = args.predict
lrstep = args.epochs//4
phases = ['train', 'val', 'test']
embname = emb_type.replace('/', '-')
print('Prediction mode:', predict_mode)
print(embname)
#%%

# wbuids = set(pd.read_csv('../splits/impSNPs_unrel_EUR_maf001_info.9_geno.1_hwe1em7_chr22.fam', sep='\t').values[:, 0])

# phase_ids = { ph: np.loadtxt(f'../../splits/20253_{ph}_ids.txt', dtype=int) for ph in phases }
# for ph, ids in phase_ids.items():
#     np.savetxt(f'../splits/{ph}_wbu.txt', [i for i in ids if i in wbuids], fmt='%d')
phase_ids = { ph: np.loadtxt(f'../splits/{ph}_wbu.txt', dtype=int) for ph in phases }
#%%

class EmbDataset(Dataset):
    def __init__(self, task, root_dir, emb_dir, split_ids, stats=None):

        # NOTE: loading is based on images to stay consistent in evals later
        # FIXME: some embs from insts 2 and 3 might have been overwritten

        self.root_dir = root_dir
        self.emb_dir = emb_dir
        split_ids = { i: True for i in split_ids}

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

        if 'tsv' in self.emb_dir:
            self.emb_df = pd.read_csv(self.emb_dir, sep='\t').set_index('IID')

        self.ids = [id for id in self.ids if id in self.emb_df.index]

        if 'tsv' in self.emb_dir:
            self.emb_vecs = np.array([self.emb_df.loc[id].values for id in self.ids])

        print('Found :', len(existing_images))
        print('Loaded:', len(self.ids))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        base_name = self.ids[index]

        if hasattr(self, 'emb_vecs'):
            evec = self.emb_vecs[index]
        else:
            evec = np.load(f'{self.emb_dir}/{base_name}.npz')['arr_0']
            evec = np.load(f'{self.emb_dir}/{base_name}.npz')['arr_0']

        lvec = self.labels[base_name]
        nanmask = np.isnan(lvec)

        return evec, nanmask, lvec

imgs_dir = '/u/scratch/u/ulzee/brainsplat/data/images/20253'
if args.emb_file is not None:
    embs_dir = args.emb_file
else:
    embs_dir = f'/u/scratch/u/ulzee/brainsplat/data/embs/{emb_type}'
trainset = EmbDataset(task, imgs_dir, embs_dir, phase_ids['train'])
dsets = dict(
    train=trainset,
    val=EmbDataset(task, imgs_dir, embs_dir, phase_ids['val'], stats=trainset.stats),
    test=EmbDataset(task, imgs_dir, embs_dir, phase_ids['test'], stats=trainset.stats),
)
# assert False
#%%

class MLP(nn.Module):
    def __init__(self, indim, outdim, hdim=512, nlayers=3, normfn=None):
        super().__init__()

        layers = []
        for ni in range(nlayers):
            if ni != 0:
                layers += [nn.ReLU()]
                if normfn is not None:
                    if normfn == nn.BatchNorm1d:
                        layers += [normfn(hdim)]
            layers += [nn.Linear(
                hdim if ni !=0 else indim,
                hdim if ni != nlayers - 1 else outdim)]

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


sample = trainset[0]
net = MLP(
    indim=len(sample[0]),
    outdim=len(sample[1]),
    nlayers=3,
    normfn=nn.BatchNorm1d,
    # normfn=nn.LayerNorm,
).to(device)

if predict_mode:
    # NOTE: save
    net.load_state_dict(torch.load(f'saved/{task}/mlp_best_{embname}.pth'))
#%%

loaders = { ph: torch.utils.data.DataLoader(
                ds, batch_size=64,
                shuffle=ph == 'train', num_workers=2) for ph, ds in dsets.items() }

# Define the loss function and optimizer
import torch.optim as optim

criterion = nn.MSELoss()
optimizer = optim.AdamW(
    net.parameters(), lr=args.lr,
    weight_decay=args.weight_decay
)
scheduler = optim.lr_scheduler.StepLR(optimizer, lrstep, 0.5)

# Train the network
hist = []
best_val_loss = float('inf')
best_model_state = None

preds = []
for epoch in range(args.epochs if not predict_mode else 1):  # loop over the dataset multiple times

    byphase = []
    for ph, loader in loaders.items():
        if predict_mode:
            if ph in ['train', 'val']: continue
        net.train() if ph == 'train' else net.eval()
        pbar = tqdm(loader)

        running_loss = [0, 0]
        stats = []
        for i, data in enumerate(pbar):
            # get the inputs; data is a list of [inputs, labels]
            labels = data[-1].to(device).float()
            inputs = data[0].to(device).float()
            obsmask = ~data[1].to(device).bool()

            # zero the parameter gradients
            if ph == 'train': optimizer.zero_grad()

            # forward + backward + optimize
            if ph == 'train':
                outputs = net(inputs)
                loss = criterion(outputs[obsmask], labels[obsmask])
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    outputs = net(inputs)
                    loss = criterion(outputs[obsmask], labels[obsmask])

                    if predict_mode:
                        preds += [(outputs.cpu().numpy(), labels.cpu().numpy())]
            running_loss[0] += len(data[0])*loss.item()
            running_loss[1] += len(data[0])

            pbar.set_postfix(dict(e=epoch, p=ph, ls='%.4f'%(running_loss[0]/running_loss[1])))
            stats += [loss.item()]

        byphase += [stats]

        # Check if validation loss improves
        if ph == 'val' and sum(stats) / len(stats) < best_val_loss:
            best_val_loss = sum(stats) / len(stats)
            best_model_state = net.state_dict()

    hist += [byphase]
    scheduler.step()


    if not predict_mode:
        # Save the model if validation loss improves
        if best_model_state is not None:
            torch.save(best_model_state, f'saved/{task}/mlp_best_{embname}.pth')

        with open(f'saved/{task}/mlp_hist_{embname}.json', 'w') as fl:
            json.dump(hist, fl)
print('Finished Training')
#%%
if predict_mode:
    savename = f'mlp_best_{embname}'
    preds, labs = [np.concatenate(t) for t in zip(*preds)]

    for colvals, saveto in zip([preds], [f'saved/{task}/predictions_test_{savename}.csv']):
        dfdict = dict(ids=dsets['test'].ids)
        for ci, (cname, col) in enumerate(zip(trainset.ldf.columns, colvals.T)):
            dfdict[cname] = (col*trainset.stats[1].values[ci]) + col*trainset.stats[0].values[ci]
        pd.DataFrame(dfdict).set_index('ids').to_csv(saveto)
