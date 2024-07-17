from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import torch

from typing import Optional, Iterable, List, Tuple
import blobfile as bf
import os

from CLAM.dataset_modules.dataset_h5 import Whole_Slide_Bag_FP

class PatchBag(Dataset):
	def __init__(
		self,
		wsi_dir: str,
		h5_dir: str,
		img_transforms: transforms.Compose = None,
		filepaths: List[str] = None
	) -> None:
		"""
		Dataset of patches from domain-specific WSIs

		Args:
			wsi_dir (str): directory containing WSIs from single domain
			h5_dir (str): directory with .h5 files for each equivalently named WSI
			img_transforms (transforms.Compose, optional): patch transformations on PIL images
			filepaths (List[str], optional): list of filepaths to patches
		"""
		self.img_transforms = img_transforms
		self.slide_bags = []
		self.filepaths = filepaths
		self.lengths = [0]

		all_wsi_fs = sorted(bf.listdir(wsi_dir))
		for wsi_path in all_wsi_fs:
			h5_path = os.path.join(h5_dir, os.path.splitext(wsi_path)[0] + '.h5')
			if not os.path.exists(h5_path):
				continue
			else:
				wsi_path = os.path.join(wsi_dir, wsi_path)
				slide_bag = Whole_Slide_Bag_FP(h5_path, wsi_path, img_transforms)
				self.slide_bags.append(slide_bag)
				self.lengths.append(len(slide_bag))

		self.length = sum(self.lengths)
		self.cutoffs = np.cumsum(self.lengths)[:-1] # cutoffs[i]: index of first patch in slide_bags[i]
		self.ImageToTensor = transforms.Compose([
			transforms.PILToTensor()
		])
	
	def __len__(self):
		return self.length
	
	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
		slide_idx = np.searchsorted(self.cutoffs, idx, side='right') - 1
		slide_bag = self.slide_bags[slide_idx]
		patch_idx = idx - self.cutoffs[slide_idx]
		img = slide_bag[patch_idx][0]

		out_dict = dict()
		if self.filepaths is not None:
			out_dict['filepath'] = self.filepaths[slide_idx]
		return self.ImageToTensor(img).float(), out_dict # N x C x H x W
	
class MultiPatchBag(Dataset):
	def __init__(
		self,
		wsi_dirs: List[str],
		h5_dirs: List[str],
	) -> None:
		"""
		Dataset of patches from multiple domain-specific WSIs

		Args:
			wsi_dirs (List[str]): directories containing WSIs from single domain
			h5_dirs (List[str]): directories with .h5 files for each equivalently named WSI
		"""
		self.patch_bags = []
		self.lengths = [0]
		self.pb_lengths = [0]

		for wsi_dir, h5_dir in zip(wsi_dirs, h5_dirs):
			patch_bag = PatchBag(wsi_dir, h5_dir)
			self.patch_bags.append(patch_bag)
			self.lengths.extend(patch_bag.lengths)
			self.pb_lengths.append(patch_bag.__len__())

		self.length = sum(self.lengths)
		self.idx_cutoffs = np.cumsum(self.lengths)[:-1] # idx_cutoffs[i]: index of first patch in implicit slide_bags[i]
		self.pb_cutoffs = np.cumsum(self.pb_lengths)[:-1] # pb_cutoffs[i]: index of first patch in patch_bags[i]
		
	def __len__(self):
		return self.length

	def __getitem__(self, idx: int) -> torch.Tensor:
		pb_idx = np.searchsorted(self.pb_cutoffs, idx, side='right') - 1
		slide_idx = np.searchsorted(self.idx_cutoffs, idx, side='right') - 1
		patch_bag = self.patch_bags[pb_idx]
		patch_idx = idx - self.idx_cutoffs[slide_idx]
		img = patch_bag[patch_idx][0]
		return img.float()

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
		img_transforms
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

def load_multipatchbag(
	wsi_dirs: List[str],
	h5_dirs: List[str],
	batch_size: int,
	img_transforms: Optional[transforms.Compose] = None,
	deterministic: bool = False,
	num_workers: int = 4
) -> Iterable:
	"""
	Data generator for patches from multiple-domain WSIs

	Args:
		wsi_dirs (List[str]): paths to WSIs
		h5_dirs (List[str]): paths to .h5 files w/ patch coords
		batch_size (int): patches per batch
		img_transforms (Optional[transforms.Compose], optional): patch transformations
		deterministic (bool, optional): whether to used fixed ordering
		num_workers (int, optional): number of workers for data loading
	"""
	dataset = MultiPatchBag(
		wsi_dirs,
		h5_dirs
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

def load_source_data_for_domain_translation(
	*,
	batch_size: int,
	image_size: int = 256,
	wsi_dir: str = "/project/kimlab_hshisto/WSI/bright",
	h5_dir: str = '/home/roblig22/ddib/datasets/histology/tilings/bright/patches',
	in_channels: int = 3,
	class_cond: bool = False
) -> Iterable:
	"""
	Loads source domain images for translation by creating
	(image, kwargs) pair generator

	Loads each patch only once, as opposed to allowing repetition
	as in load_patchbag

	Args:
		batch_size (int): number of pairs per batch
		image_size (int): size to which images are resized
		data_dir (str): path to source domain images
		in_channels (int): number of input channels
		class_cond (bool): whether to use class labels
	"""
	if not wsi_dir or not h5_dir:
		raise ValueError('Unspecified data directory')
	filepaths = [f for f in list_wsi_files(wsi_dir)]
	dataset = PatchBag(
		wsi_dir=wsi_dir,
		h5_dir=h5_dir,
		filepaths=filepaths
	)
	loader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=1
	)
	yield from loader

def list_wsi_files(wsi_dir: str) -> List[str]:
	"""
	List  WSI files in directory

	Args:
		_dir (str): path to WSIs
	"""
	files = sorted(bf.listdir(wsi_dir))
	results = []
	for entry in files:
		full_path = bf.join(wsi_dir, entry)
		ext = entry.split('.')[-1]
		if '.' in entry and ext.lower() in ['qptiff', 'tif', 'svs']:
			results.append(full_path)
	return results
