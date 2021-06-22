import os
import numpy as np


seg_prefix = 'model-019000-segs-train'
segs = np.load(seg_prefix + '.npy')

idx_pos = segs>=0.5
idx_neg = segs<0.5
del segs

stds = np.load('model-019000-stds-train.npy')
stds_pos = stds[idx_pos]
stds_neg = stds[idx_neg]

for percent in [80,85,90,95]:
	th_pos = np.percentile(stds_pos, percent)
	th_neg = np.percentile(stds_neg, percent)
	print('%d, %.6f, %.6f' % (percent,th_pos,th_neg))
	mask = np.logical_or(np.logical_and(idx_pos, stds<th_pos), np.logical_and(idx_neg, stds<th_neg))
	mask = mask.astype(np.uint8)
	np.save(seg_prefix + '-mk%02d.npy'%percent, mask)
