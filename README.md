# Brainsplat

## Folder structure

Some required folders to load and save models are:
```bash
saved/idps
saved/generic
```

## (optional) Generating the data

### 3-axis images

### Lowres volumes

### Embeddings

## Running baselines

### 3-axis Resnet-18


We fit a slightly modified Resnet-18 from scratch to test how well a conventional deep learning model predicts IDPs and generic phenotypes.
The resnet runs the convolution layers on centric top, side, and front views (3 image input) of the MRI volume. The latent feature maps are then summed, and passed through the standard Resnet bottleneck.
The number of pixels of the 3 images (256^2 * 3 = 196,608) is essentially the same as the embedding dimension with no projections (16^2 * 256 * 3, where 256 is the non-projected dim).

The model can be trained as:
```bash
python train_resnet.py cuda:0
```

### 64^3 3D Resnet-18

## Predictions from embeddings

```bash
python train_emb.py --device cuda:0 --task idps \
    --emb-type oct16/proj_normal_k10 --epochs 100
```


## Scoring

This script is best if run interactively to see the figures.
The `score.py` file can be run to generate figures that compare the prediction accuracy of the various models that are trained.

## Where the data is located

```bash
/u/scratch/u/ulzee/brainsplat/data/images # images for baselines
/u/scratch/u/ulzee/brainsplat/data/embs/**/proj_normal_k10 # embeddings
...
```