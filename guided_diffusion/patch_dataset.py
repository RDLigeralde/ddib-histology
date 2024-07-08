import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

import mpi4py.MPI as MPI
import os
from typing import Optional, Iterable

from CLAM.dataset_modules.dataset_h5 import Whole_Slide_Bag_FP

class PatchBag(Dataset):
	def __init__(
		self,
		wsi_dir: str,
		h5_dir: str,
		img_transforms: transforms.Compose = None,
		shard: int = MPI.COMM_WORLD.Get_rank(),
		num_shards: int = MPI.COMM_WORLD.Get_size()
	) -> None:
		"""
		Dataset of patches from domain-specific WSIs

		Args:
			wsi_dir (str): directory containing WSIs from single domain
			h5_dir (str): directory with .h5 files for each equivalently named WSI
			img_transforms (transforms.Compose, optional): patch transformations on PIL images
			shard (int, optional): rank of current process
			num_shards (int, optional): total number of processes
		"""
		self.img_transforms = img_transforms
		self.slide_bags = []
		lengths = [0]

		all_wsi_fs = os.listdir(wsi_dir)
		sharded = all_wsi_fs[shard::num_shards]

		for wsi_path in sharded:
			h5_path = os.path.join(h5_dir, os.path.splitext(wsi_path)[0] + '.h5')
			if not os.path.exists(h5_path):
				continue
			else:
				wsi_path = os.path.join(wsi_dir, wsi_path)
				slide_bag = Whole_Slide_Bag_FP(h5_path, wsi_path, img_transforms)
				self.slide_bags.append(slide_bag)
				lengths.append(len(slide_bag))

		self.length = sum(lengths)
		self.cutoffs = np.cumsum(lengths)[:-1] # cutoffs[i]: index of first patch in slide_bags[i]
		self.ImageToTensor = transforms.Compose([
			transforms.PILToTensor()
		])
	
	def __len__(self):
		return self.length
	
	def __getitem__(self, idx: int) -> torch.Tensor: # TODO: check if returning coordinates is useful
		slide_idx = np.searchsorted(self.cutoffs, idx, side='right') - 1
		slide_bag = self.slide_bags[slide_idx]
		patch_idx = idx - self.cutoffs[slide_idx]
		img = slide_bag[patch_idx][0]
		#HACK: (img, {}) tuple for guided_diffusion compatibility
		return self.ImageToTensor(img).float(), dict() # N x C x H x W

def load_patchbag(
	wsi_dir: str,
	h5_dir: str,
	batch_size: int,
	img_transforms: Optional[transforms.Compose] = None,
	deterministic: bool = False,
	num_workers: int = 4 # 4 GPUs
) -> Iterable:
	"""
	Data generator for patches from single-domain WSIs

	Args:
		wsi_dir (str): path to WSIs
		h5_dir (str): path to .h5 files w/ patch coords
		batch_size (int): patches per batch
		img_transforms (Optional[transforms.Compose], optional): patch transformations
		deterministic (bool, optional): whether to used fixed ordering
		num_workers (int, optional): number of workers for data loading
	"""
	dataset = PatchBag(
		wsi_dir,
		h5_dir, 
		img_transforms,
		shard = MPI.COMM_WORLD.Get_rank(),
		num_shards = MPI.COMM_WORLD.Get_size(),
	)

	if deterministic:
		loader = DataLoader(
			dataset, 
			batch_size = batch_size, 
			shuffle = False, 
			num_workers = num_workers, 
			drop_last = True
		)
	else:
		loader = DataLoader(
			dataset, 
			batch_size = batch_size, 
			shuffle = True, 
			num_workers = num_workers, 
			drop_last = True
		)

	while True:
		yield from loader
