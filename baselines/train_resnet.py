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

# wbuids = set(pd.read_csv('../splits/impSNPs_unrel_EUR_maf001_info.9_geno.1_hwe1em7_chr22.fam', sep='\t').values[:, 0])

# phase_ids = { ph: np.loadtxt(f'../../splits/20253_{ph}_ids.txt', dtype=int) for ph in phases }
# for ph, ids in phase_ids.items():
#     np.savetxt(f'../splits/{ph}_wbu.txt', [i for i in ids if i in wbuids], fmt='%d')
phase_ids = { ph: np.loadtxt(f'../splits/{ph}_wbu.txt', dtype=int) for ph in phases }
#%%

class CustomDataset(Dataset):
    def __init__(self, task, root_dir, split_ids, stats=None):
        self.root_dir = root_dir
        split_ids = { i: True for i in split_ids}
        # self.transform = transform
        # existing_images = list(sorted([i for i in os.listdir(root_dir) if i.endswith('_20253_2_0-top.jpg')]))
        with open(root_dir + '.txt') as fl:
            existing_images = [ln.strip().split('/')[1] for ln in fl]
        # icount = dict()
        # byside = []
        # for side in ['top', 'side', 'front']:
        #     for img in os.listdir(root_dir) if img.endswith('-top.jpg'):
        #         pid = int(img.split('/')[-1].spilt('.')[0].spit('_')[0])
        #         c = icount.get(pid, 0)
        #         icount[pid] = c + 1
        # pids_hasall =
        # self.images = [img for img in os.listdir(root_dir) if img.endswith('-top.jpg') and icount[int(img.split('/')[-1].split('.')[0].split('_')[0])] == 3]

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
        top_image_path = os.path.join(self.root_dir, f"{base_name}_20253_2_0-top.jpg")
        side_image_path = os.path.join(self.root_dir, f"{base_name}_20253_2_0-side.jpg")
        front_image_path = os.path.join(self.root_dir, f"{base_name}_20253_2_0-front.jpg")

        assert (os.path.exists(side_image_path) and os.path.exists(front_image_path))

        top_image = np.array(Image.open(top_image_path)).astype(np.float32)/256*2-1
        side_image = np.array(Image.open(side_image_path)).astype(np.float32)/256*2-1
        front_image = np.array(Image.open(front_image_path)).astype(np.float32)/256*2-1

        # top_image = self.transform(top_image)
        # side_image = self.transform(side_image)
        # front_image = self.transform(front_image)

        lvec = self.labels[base_name]
        nanmask = np.isnan(lvec)
        return top_image, side_image, front_image, nanmask, lvec

trainset = CustomDataset(task, '/u/scratch/u/ulzee/brainsplat/data/images/20253', phase_ids['train'])
dsets = dict(
    train=trainset,
    val=CustomDataset(task, '/u/scratch/u/ulzee/brainsplat/data/images/20253', phase_ids['val'], stats=trainset.stats),
    test=CustomDataset(task, '/u/scratch/u/ulzee/brainsplat/data/images/20253', phase_ids['test'], stats=trainset.stats),
)
#%%

# Define the model
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock

class MultiViewResNet(ResNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def fwd_branch(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def _forward_impl(self, *xs):
        xsum = 0.0
        for x in xs:
            xsum += self.fwd_branch(x)

        x = self.avgpool(xsum)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, *xs):
        return self._forward_impl(*xs)


sample = trainset[0]
net = MultiViewResNet(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=len(sample[-1])).to(device)
if predict_mode:
    net.load_state_dict(torch.load(f'saved/{task}/resnet_best.pth'))
out = net(*[torch.from_numpy(t).unsqueeze(0).unsqueeze(1).to(device) for t in sample[:3]])
out.shape
#%%

loaders = { ph: torch.utils.data.DataLoader(
                ds, batch_size=64,
                shuffle=ph == 'train', num_workers=2) for ph, ds in dsets.items() }

# Define the loss function and optimizer
import torch.optim as optim

criterion = nn.MSELoss()
optimizer = optim.AdamW(net.parameters(), lr=1e-4)
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
            inputs = [t.to(device).unsqueeze(1) for t in data[:3]]

            # zero the parameter gradients
            if ph == 'train': optimizer.zero_grad()

            # forward + backward + optimize
            if ph == 'train':
                outputs = net(*inputs)
                loss = criterion(outputs[obsmask], labels[obsmask])
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    outputs = net(*inputs)
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
            torch.save(best_model_state, f'saved/{task}/resnset_best.pth')

        with open(f'saved/{task}/resnset_hist.json', 'w') as fl:
            json.dump(hist, fl)
print('Finished Training')
#%%
if predict_mode:
    preds, labs = [np.concatenate(t) for t in zip(*preds)]

    for colvals, saveto in zip([preds, labs], [f'saved/{task}/predictions_test_resnet.csv', f'saved/{task}/targets_test.csv']):
        dfdict = dict(ids=dsets['test'].ids)
        for ci, (cname, col) in enumerate(zip(trainset.ldf.columns, colvals.T)):
            dfdict[cname] = (col*trainset.stats[1].values[ci]) + col*trainset.stats[0].values[ci]
        pd.DataFrame(dfdict).set_index('ids').to_csv(saveto)
