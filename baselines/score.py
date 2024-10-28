#%%
import matplotlib.pyplot as plt
import pandas as pd
import json
from tqdm import tqdm
import numpy as np
#%%
task = 'idps'
# task = 'generic'
#%%
target = pd.read_csv(f'saved/{task}/targets_test.csv').set_index('ids')
#%%
mdls = [
    f'saved/{task}/predictions_test_resnet3ds18.csv',

    f'saved/{task}/predictions_test_mlp_best_oct17-proj_normal_k10.csv',
    f'saved/{task}/predictions_test_mlp_best_oct17-proj_normal_k100.csv',
    f'saved/{task}/predictions_test_mlp_best_oct17-proj_identity.csv',

    f'saved/{task}/predictions_test_mlp_best_oct16-proj_normal_k10.csv',
    f'saved/{task}/predictions_test_mlp_best_oct16-proj_normal_k100.csv',
    f'saved/{task}/predictions_test_mlp_best_oct16-proj_identity.csv',

    f'saved/{task}/predictions_test_mlp_best_oct18_allslices-proj_normal_k10.csv',
    f'saved/{task}/predictions_test_mlp_best_oct18_allslices-proj_normal_k100.csv',
    f'saved/{task}/predictions_test_mlp_best_oct18_allslices-proj_identity.csv',
    # f'saved/{task}/predictions_test_mlp_best_oct17-proj_identity.csv',

    # other models
    f'saved/{task}/predictions_test_mlp_best_oct21_vae.csv',
    f'saved/{task}/predictions_test_resnet.csv',
]
preds = [(f.split('predictions_test_')[-1].split('.')[0], pd.read_csv(f).set_index('ids')) for f in mdls]
nboots = 10

nsamples = len(preds[0][1])
# boot_ixs = [np.random.choice(nsamples, size=nsamples, replace=True).tolist() for _ in range(nboots)]
# np.save(f'saved/{task}/boots10.npy', boot_ixs)
boot_ixs = np.load(f'saved/{task}/boots10.npy')

nsamples
# %%
r2_bymodel = dict()
r2stds_bymodel = dict()

idref = preds[-2][1]
r2stats_bymodel = dict()
# r2avg_stds_bymodel = dict()
for mname, pdf in tqdm(preds):
    cr2s = []
    cr2stds = []

    allstats = []
    for c in target.columns:
        if nboots is None:
            cr2s += [np.corrcoef(target[c], pdf[c])[0, 1]**2]
        else:
            stats = []
            for bixs in boot_ixs:
                boot_ids = target.index.values[bixs]
                idict = { id: True for id in idref.index}
                matched_ids = [id for id in boot_ids if id in idict]
                stats += [np.corrcoef(
                    target[c].loc[matched_ids],
                    pdf[c].loc[matched_ids])[0, 1]**2]
            stats = np.array(stats)

            est = stats.mean(0)
            std = stats.std(0)
            allstats += [stats]
            cr2s += [est]
            cr2stds += [std]
    r2_bymodel[mname] = np.array(cr2s)
    r2stds_bymodel[mname] = np.array(cr2stds)

    allstats = np.array(allstats) #.mean(0)

    r2stats_bymodel[mname] = allstats
    # r2avg_bymodel[mname] = avgstat.mean()
    # r2avg_stds_bymodel[mname] = avgstat.std()
    # allstats.mean(0)
# # %%
# for mname, mr2s in r2_bymodel.items():
#     plt.figure()
#     plt.title(mname)
#     plt.hist(mr2s)
#     plt.show()
# #%%
# plt.figure(figsize=(5, 5))
# plt.scatter(r2_bymodel[preds[0][0]], r2_bymodel[preds[-1][0]])
# plt.ylabel('resnet2d')
# plt.xlabel('resnet3d')
# plt.axline((0,0), slope=1, color='gray')
# plt.show()
# %%
readable_names = {
    '':'',
    'mlp_best_oct17-proj_normal_k10': 'BrainSplat (K=10)',
    'mlp_best_oct17-proj_normal_k100': 'BrainSplat (K=100)',
    'mlp_best_oct17-proj_identity': 'BrainSplat (Identity)',
    'mlp_best_oct17-proj_normal_k10': 'BrainSplat (K=10)',
    'mlp_best_oct17-proj_normal_k100': 'BrainSplat (K=100)',
    'mlp_best_oct17-proj_identity': 'BrainSplat (Identity)',
    'mlp_best_oct18_allslices-proj_normal_k10': 'BrainSplat (K=10)',
    'mlp_best_oct18_allslices-proj_normal_k100': 'BrainSplat (K=100)',
    'mlp_best_oct18_allslices-proj_identity': 'BrainSplat (identity)',
}
plt.figure(figsize=(12, 4.5))
# plt.suptitle('Slice sampling: Every 10 slice')
plt.suptitle('Slice sampling: Full volume')
# for si, other in enumerate(preds[1:3+1]):

compare_against = preds[0][0]
for si, other in enumerate(preds[6+1:6+1+3]):
    plt.subplot(1, 3, si+1)
    other_name = other[0]
    # plt.title('Median: %.2f vs %.2f' % (
    #     np.median(r2_bymodel[preds[0][0]]), np.median(r2_bymodel[other_name])
    # ))
    plt.scatter(r2_bymodel[compare_against], r2_bymodel[other_name], s=5, color='gray')
    plt.axline((0,0), slope=1, color='gray', linestyle='dashed')

    plt.errorbar(
        r2_bymodel[compare_against], r2_bymodel[other_name],
        1.96*r2stds_bymodel[other_name], color='C2', linestyle='none', alpha=0.5
    )
    plt.errorbar(
        r2_bymodel[compare_against], r2_bymodel[other_name],
        xerr=1.96*r2stds_bymodel[compare_against], color='C2', linestyle='none', alpha=0.5
    )

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.ylabel(readable_names[other_name])
    plt.xlabel('Resnet 3d')
plt.tight_layout()
# plt.savefig('figures/dump/compare_r2_skip10.jpg', dpi=150)
plt.savefig('figures/dump/compare_r2_fullvol.jpg', dpi=150)
plt.show()
# %%
plt.figure(figsize=(7, 5))

other_r2s = [r2_bymodel[other_name] for other_name, _ in preds[4:4+3]]
plt.plot([0, 10, 100, 256], [0] + [np.mean(ls) for ls in other_r2s], label='brainsplat (MedSAM)')
plt.scatter([0, 10, 100, 256], [0] + [np.mean(ls) for ls in other_r2s])

other_r2s = [r2_bymodel[other_name] for other_name, _ in preds[1:3+1]]
plt.plot([0, 10, 100, 256], [0] + [np.mean(ls) for ls in other_r2s], label='brainsplat (SAM)')
plt.scatter([0, 10, 100, 256], [0] + [np.mean(ls) for ls in other_r2s])

other_r2s = [r2_bymodel[other_name] for other_name, _ in preds[7:7+3]]
plt.plot([0, 10, 100, 256], [0] + [np.mean(ls) for ls in other_r2s], label='brainsplat (SAM fullvol)')
plt.scatter([0, 10, 100, 256], [0] + [np.mean(ls) for ls in other_r2s])

# plt.errorbar([10, 100, 256], [np.mean(ls) for ls in other_r2s],
#              [np.std(ls) for ls in other_r2s], color='black', capsize=5, linestyle='none')
plt.axhline(np.mean(r2_bymodel[preds[0][0]]), color='cyan', label='3d resnet')
plt.axhline(np.mean(r2_bymodel[preds[-1]
[0]]), color='gray', label='3slice resnet')
plt.ylim(0, 1)
plt.xticks([0, 10, 100, 256])
plt.title('Avg r2 accuracy (199 IDPs)')
plt.xlabel('"K"')
plt.legend()
plt.show()
#%%
plt.figure(figsize=(7, 7))

plt.title('Performance vs Input size')
other_r2s = [r2_bymodel[other_name] for other_name, _ in preds]
nvoxels = [16*16*10*3,16*16*100*3, 16*16*256*3]
nvoxels = [v*4/1024 for v in nvoxels]
meanr2s = [np.mean(ls) for ls in other_r2s[7:10]]
plt.scatter(nvoxels, meanr2s)
for nv, r2v, (mname, _) in zip(nvoxels, meanr2s, preds[1:4] + [('(impossible)',None)]):
    plt.text(nv, r2v, mname)
plt.ylim(0, 1)
plt.xlim(0, max(nvoxels)*1.5)

plt.axvline(256**2*3 * 4/1024, color='orange')
plt.text(256**2*3 * 4/1024, 0.9, '3-slices\n(256x256)')
plt.axvline(256**2 * 4/1024, color='orange')
plt.text(256**2 * 4/1024, 0.9, '1-slice\n(256x256)')
plt.axvline(64**2 * 4/1024, color='orange')
plt.text(64**2 * 4/1024, 0.9, '64x64\npatch')

plt.axvline(64**3 * 4/1024, color='orange')
plt.text(64**3 * 4/1024, 0.9, '64^3\nvolume')

plt.ylabel('Accuracy (r2)')
plt.xlabel('Input size (KBs)\n(256^3 uint8 volume ~ 16MB)')
plt.show()
#%%
plt.figure(figsize=(12, 4))
# plt.title('Performance vs Input size')

shown_lines = {
    1: 'BrainSplat (10-th slice)',
    # 4: 'MedBrainSplat (10-th slice)',
    7: 'BrainSplat (full volume, SAM)',
    4: 'BrainSplat (full volume, MedSAM)'
}
for offset in [1, 7, 4]:
    other_r2s = [r2_bymodel[other_name] for other_name, _ in preds]
    nvoxels = [16*16*10*3,16*16*100*3, 16*16*256*3]
    nvoxels = [0] + [v*4/1024 for v in nvoxels]
    meanr2s = [0] + [np.mean(ls) for ls in other_r2s[offset:offset+3]]
    plt.scatter(nvoxels, meanr2s)
    plt.plot(nvoxels, meanr2s, label=shown_lines[offset])
    for ki, (nv, r2v, (mname, _)) in enumerate(zip(nvoxels, meanr2s, [('', None)] + preds[1:4])):
        plt.text(nv, r2v+0.02, str(['', 'K=10', 'K=100', 'Ident'][ki]))
plt.ylim(0, 1)
# plt.xlim(0, max(nvoxels)*1.5)

plt.axvline(256**2*3 * 4/1024, color='gray', linestyle='dashed')
plt.text(256**2*3 * 4/1024, 0.9, '3-slices\n(256x256)')
plt.axvline(256**2 * 4/1024, color='gray', linestyle='dashed')
plt.text(256**2 * 4/1024, 0.9, '1-slice\n(256x256)')

# plt.axvline(64**3 * 4/1024, color='gray', linestyle='dashed')
# plt.text(64**3 * 4/1024, 0.9, '64^3\nvolume')
plt.scatter(1100, np.mean(other_r2s[0]))
plt.plot([-200, 1100], [np.mean(other_r2s[0])]*2, label='Resnet (volume)', alpha=0.3)

plt.scatter(256**2*3 * 4/1024, np.mean(other_r2s[-1]))
plt.plot([-200, 256**2*3 * 4/1024], [np.mean(other_r2s[-1])]*2, label='Resnet (3-slices)', alpha=0.3)

plt.scatter(512 * 4/1024, np.mean(other_r2s[-2]))
plt.plot([-200, 512 * 4/1024], [np.mean(other_r2s[-2])]*2, label='VAE (512 dim)', alpha=0.3)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.xlim(-100, 1200)
plt.ylabel('Accuracy (r2)')
plt.xlabel('Modality size in memory')

xs = [100] + [200*i for i in range(5)]
ts = [f'{k}KB' for k in xs]
plt.xticks(xs + [1100], ts + ['10MB\n224 voxel volume\nuint8 quantized'])

plt.legend(frameon=False)

plt.tight_layout()
plt.savefig('figures/dump/scaling.jpg', dpi=150)
plt.show()
# %%
with open('saved/idps/mlp_hist.json') as fl:
    hist = json.load(fl)
# %%
len(hist), len(hist[0])
# %%
import scipy.stats
# Count number of IDPs that are not significantly worse

r0 = r2stats_bymodel['resnet3ds18']

for mname, _ in preds[1:]:
    r1 = r2stats_bymodel[mname]
    # r1 = r2stats_bymodel['mlp_best_oct18_allslices-proj_normal_k100']
    # r1 = r2stats_bymodel['mlp_best_oct18_allslices-proj_identity']

    diffs = r0 - r1

    stat = diffs.mean(1)
    std = diffs.std(1)

    zscore = stat/std

    thresh = 0.05/199

    pvals = scipy.stats.norm.sf(abs(zscore))*2

    sigdiff = pvals < thresh

    lowersig = (stat > 0) & sigdiff
    highersig = (stat < 0) & sigdiff

    print(mname)
    # print('%d %d %d' % (199-lowersig.sum()- highersig.sum(), lowersig.sum(), highersig.sum(), ))
    print('%.3f (%.3f)' % (r1.mean(0).mean(0), 1.96*r1.mean(0).std(0)))




# diffs.shape
# %%
