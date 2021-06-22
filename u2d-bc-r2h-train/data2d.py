import os
import sys
import cv2
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.ndimage.interpolation import rotate


class Train(Dataset):
	def __init__(self, prefix, percent, patch_size=(256,256), rigid_aug=True,
	             elastic=True, prob=0.8, pad=108//2, angle=(0,359)):
		super(Train, self).__init__()
		# construct names
		im_npy = prefix + '_ims.npy'
		seg_prefix = 'model-019000-segs-train'
		
		# parameters
		self.crop_size = np.array(patch_size)
		self.rigid_aug = rigid_aug
		
		# elastic deformation
		self.elastic = elastic
		self.prob = prob
		self.pad = pad
		self.angle = angle
		
		# load ims
		self.ims = np.load(im_npy)
		self.shape = self.ims.shape
		
		# load segs and binarize
		self.segs = np.load(seg_prefix + '.npy')
		self.segs[self.segs>=0.5] = 1
		self.segs[self.segs<0.5] = 0
		self.segs = self.segs.astype(np.uint8)
		
		# load masks
		if percent < 100:
			print('Loading %s' % seg_prefix + '-mk%02d.npy'%percent)
			self.mks = np.load(seg_prefix + '-mk%02d.npy'%percent)
		elif percent == 100:
			self.mks = np.ones(self.shape, np.uint8)
		else:
			raise AttributeError

	def __getitem__(self, _dump):
		# pad for deformation
		crop_size = np.copy(self.crop_size)
		if self.elastic and random.uniform(0,1) < self.prob:
			do_elastic = True
			crop_size += self.pad*2
		else:
			do_elastic = False
		
		# random crop
		i = random.randint(0, self.shape[0]-1)
		j = random.randint(0, self.shape[1]-crop_size[0])
		k = random.randint(0, self.shape[2]-crop_size[1])
		im = self.ims[i, j:j+crop_size[0], k:k+crop_size[1]].copy()
		seg = self.segs[i, j:j+crop_size[0], k:k+crop_size[1]].copy()
		mk = self.mks[i, j:j+crop_size[0], k:k+crop_size[1]].copy()
		
		# rigid augmentation
		if self.rigid_aug:
			if random.uniform(0,1) < 0.5:
				im = np.flip(im, axis=0)
				seg = np.flip(seg, axis=0)
				mk = np.flip(mk, axis=0)
			if random.uniform(0,1) < 0.5:
				im = np.flip(im, axis=1)
				seg = np.flip(seg, axis=1)
				mk = np.flip(mk, axis=1)
			
			k = random.choice([0,1,2,3])
			im = np.rot90(im, k)
			seg = np.rot90(seg, k)
			mk = np.rot90(mk, k)
		
		# elastic deformation
		if do_elastic:
			angle = random.randint(self.angle[0], self.angle[1])
			im = im.astype(np.float32)
			im = rotate(im, angle, axes=(0,1), reshape=False, order=3)
			seg = rotate(seg, angle, axes=(0,1), reshape=False, order=0)
			mk = rotate(mk, angle, axes=(0,1), reshape=False, order=0)
			
			# central crop
			out_shape = crop_size
			out_shape -= self.pad*2
			
			start_inds = (im.shape - out_shape) // 2
			im = im[start_inds[0]:start_inds[0]+out_shape[0],
			        start_inds[1]:start_inds[1]+out_shape[1]]
			seg = seg[start_inds[0]:start_inds[0]+out_shape[0],
			          start_inds[1]:start_inds[1]+out_shape[1]]
			mk = mk[start_inds[0]:start_inds[0]+out_shape[0],
			        start_inds[1]:start_inds[1]+out_shape[1]]
			
			# clamp
			im[im>255] = 255
			im[im<0] = 0
			im = im.astype(np.uint8)
		
		# format
		im = np.expand_dims(im.astype(np.float32)/255.0, axis=0)
		seg = seg.astype(np.long)
		return im, seg, mk.copy()
	
	def __len__(self):
		return int(sys.maxsize)


class Provider(object):
	def __init__(self, prefix, batch_size, num_workers, **kwargs):
		self.data = Train(prefix, percent=kwargs['percent'])
		self.batch_size = batch_size
		self.num_workers = num_workers
		
		self.is_cuda = True
		self.data_iter = None
		self.iteration = 0
		self.epoch = 1
	
	def __len__(self):
		return int(sys.maxsize)
	
	def build(self):
		self.data_iter = iter(DataLoader(dataset=self.data, batch_size=self.batch_size, num_workers=self.num_workers,
		                                 shuffle=False, drop_last=False, pin_memory=False))
	
	def next(self):
		if self.data_iter is None:
			self.build()
		try:
			batch = self.data_iter.next()
			self.iteration += 1
			if self.is_cuda:
				batch[0] = batch[0].cuda()
				batch[1] = batch[1].cuda()
				batch[2] = batch[2].cuda()
			return batch[0], batch[1], batch[2]
		except StopIteration:
			self.epoch += 1
			self.build()
			self.iteration += 1
			batch = self.data_iter.next()
			if self.is_cuda:
				batch[0] = batch[0].cuda()
				batch[1] = batch[1].cuda()
				batch[2] = batch[2].cuda()
			return batch[0], batch[1], batch[2]


class Valid(Dataset):
	def __init__(self, prefix):
		super(Valid, self).__init__()
		# construct names
		im_npy = prefix + '_ims.npy'
		seg_npy = prefix + '_segs.npy'
		
		# load ims and segs
		self.ims = np.load(im_npy)
		self.segs = np.load(seg_npy)
		
		# binarize segs
		self.segs[self.segs>0] = 1
		self.segs = self.segs.astype(np.uint8)
