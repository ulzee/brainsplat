from glob import glob
import numpy as np
import sys, os, shutil, warnings
from PIL import Image
import nibabel as nib
from tqdm import tqdm
import zipfile
import zlib
import matplotlib.pyplot as plt
import argparse


####### ARGS #######
parser = argparse.ArgumentParser()
parser.add_argument('--jobidx', type=str, required=True)
parser.add_argument('--root', type=str, required=True) # /u/project/sgss/UKBB/imaging/bulk/20253/
parser.add_argument('--odir', type=str, required=True) # /u/scratch/a/adityago/imaging/processed/20253/
parser.add_argument('--sufix', type=str, default='20253_2_0.zip') #
parser.add_argument('--mode', type=str, default='T2_FLAIR') #
# parser.add_argument('--view', type=str,  default="coronal")
# parser.add_argument('--slices', type=int, default=64) # None == full volume, rec: 64
parser.add_argument('--bsize', type=int, default=100)
args = parser.parse_args()
#####################

def read_raw_t2flair(bfl, mode = 'T2_FLAIR'):
	target = 'T2_FLAIR_brain_to_MNI.nii.gz'
	with zipfile.ZipFile(bfl, 'r') as zip_ref:
		fls_in_zip = [fl for fl in zip_ref.namelist()]
		if f"{mode}/{target}" not in fls_in_zip:
			return None
		nidat = zlib.decompress(bytearray(zip_ref.read(f"{mode}/{target}")), 15 + 32)

	return nidat

if __name__ == '__main__':
	with open(f'{args.root}.txt') as fl:
		mriraw_files = [f'{args.root}/{f.strip()}' for f in fl]
	print(f"{len(mriraw_files)} to process...")
	print(f"Example: {mriraw_files[0]}")

	### Perform checks
	valid_modes = ['T2_FLAIR']
	# valid_views = ['coronal']
	if args.mode not in valid_modes:
		raise f"{args.mode} is not a valid mode we can process yet!"
	# if args.view not in valid_views:
	# 	raise f"{args.view} is not a valid view we can process yet!"
	###
	jobnum = int(args.jobidx) - 1

	tmp_dir_name = "temp/temp_" + str(jobnum)
	tmp_dir_path = tmp_dir_name #os.path.join(args.odir, tmp_dir_name)
	os.makedirs(tmp_dir_path, exist_ok=True)
	print(f"Created temporary directory at {tmp_dir_path}")

	select_mriraw_files = mriraw_files[args.bsize*jobnum:args.bsize*(jobnum+1)]

	for bfl in tqdm(select_mriraw_files):
		fID = os.path.splitext(
				os.path.basename(bfl))[0]

		# Uncompresse + select correct file
		if args.mode == 'T2_FLAIR':
			nidat = read_raw_t2flair(bfl)
		else:
			raise NotImplementedError

		if nidat is None:
			continue
		# save in tmp dir
		with open(f"{tmp_dir_path}/{fID}.nii", 'wb') as niifile:
			niifile.write(nidat)
			del nidat

		### Process volume
		nidata = nib.load(f"{tmp_dir_path}/{fID}.nii").get_fdata()
		zz, yy, xx = nidata.shape

		fmax = 1024 + 256
		nidata[nidata < 0] = 0
		nidata[nidata > fmax] = fmax

		nidata /= fmax
		nidata_img = (nidata * 256).astype(np.uint8)

		vol224 = np.zeros((224, 224, 224), dtype=np.uint8)
		start_z = (224 - zz) // 2
		start_y = (224 - yy) // 2
		start_x = (224 - xx) // 2
		vol224[start_z:start_z+zz, start_y:start_y+yy, start_x:start_x+xx] = nidata_img

		# print(nidata.shape)

		# assert False

		# z_dim = nidata.shape[0]  # Sagittal slice
		# y_dim = nidata.shape[1]  # Coronal slice
		# x_dim = nidata.shape[2]  # Axial slice

		# if args.view == 'coronal':
		# 	final_volume = process_coronal_view(
		# 						nidata, z_dim, y_dim, x_dim,
		# 						args.slices, start_slice = 36, end_slice = 185)
		# else:
		# 	raise NotImplementedError

		# # save final lower res volume
		save_path = f'{args.odir}/{fID.split("_")[0]}.npz'
		np.savez_compressed(save_path, vol224)

		os.remove(f'{tmp_dir_path}/{fID}.nii')

		# clean up temp dir
		# shutil.rmtree(tmp_dir_path)

	print('DONE')
