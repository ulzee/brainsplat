# Brainsplat

## Folder structure

Some required folders to load and save models are:
```bash
saved/idps
saved/generic
```

## Generating data for baselines

### 3-axis images

coming soon...

### Lowres volumes

coming soon...

## Generating BrainSplat embeddings

Since the embedding script works closely with SAM, please pull this project repo which is a fork of MedSAM:

```bash
git clone https://github.com/ulzee/MedSAM-Flex
```

To reproduce the accuracy observed in our work, the SAM weights should be chosen instead of the MedSAM weights suggested in the project's description.
The SAM weights should be downloaded from `https://github.com/facebookresearch/segment-anything` and placed as such; `MedSAM/work_dir/sam_vit_b_01ec64.pth`.

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
mv sam_vit_b_01ec64.pth MedSAM/work_dir/sam_vit_b_01ec64.pth
```

Then, the `medsam_embed.py` script can be used to process the UK Biobank volumes. A manifest file (just a plaintext file with filenames of the volumes to process)
should be generated before running the file, such as `artifacts/20253_wbu.txt`:

```bash
1000102_20253_2_0.zip
1000293_20253_2_0.zip
1000315_20253_2_0.zip
1000742_20253_2_0.zip
1000881_20253_2_0.zip
1001258_20253_2_0.zip
...
```

Also, random projection matrices of size `256 x K` should be generated. The embedder will by default
look for `artifacts/proj_normal_k10.npy` and `artifacts/proj_normal_k100.npy`, but other projections can be explored.

The embedding script can then be run as:

```bash
python medsam_embed.py --encoder SAM --manifest artifacts/20253_wbu.txt \
    --start 0 --many 5000 --batch_size 32 --dataroot /input/volumes/location --saveto /output/directory \
    --device cuda:0 --all_slices
```

The input and output folers should be specified using `--dataroot` and `--saveto`.
The script will create three folders in the output directory pertaining to three resolutions of random projections (K=10, K=100, and ident).

The `--all_slices` tells the script to process every single slice of the volume; not passing this option will lead the script to observe only every 10th slice.


## Running baselines

### 3-crossections (256^2 x 3) Resnet-18 (baseline)


We fit a slightly modified Resnet-18 from scratch to test how well a conventional deep learning model predicts IDPs and generic phenotypes.
The resnet runs the convolution layers on centric top, side, and front views (3 image input) of the MRI volume. The latent feature maps are then summed, and passed through the standard Resnet bottleneck.
The number of pixels of the 3 images (256^2 * 3 = 196,608) is essentially the same as the embedding dimension with no projections (16^2 * 256 * 3, where 256 is the non-projected dim).

The model can be trained as:
```bash
python train_resnet.py cuda:0
```


### 224^3 3D Resnet-18

The 3d-resnet can be run similarly usnig the script `train_resnet3d.py`.
More coming soon...

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