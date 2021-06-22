import os
import torch
import numpy as np
from PIL import Image
from rsunet2d_uc import RSUNet
from collections import OrderedDict


# load ims
prefix = '../mitoem/mito_human_train'
ims = np.load(prefix + '_ims.npy')

# build model
model = RSUNet([16,32,48,64,80], dp=0.5).cuda()
model = model.eval()

# restore weights
weight = '../u2d-bc-rat-uc-train/models_p0.5/model-019000-0.9785.pth'
new_state_dict = OrderedDict()
state_dict = torch.load(weight)
for k, v in state_dict.items():
	name = k[7:] # remove module.
	new_state_dict[name] = v
model.load_state_dict(new_state_dict)

if torch.cuda.device_count() > 1:
	model = torch.nn.DataParallel(model)

# pad for overlap
bd = 32
ims = np.pad(ims, ((0,0),(bd,bd),(bd,bd)), mode='symmetric')

# settings
in_shape = [100, 576, 576]
stride = out_shape = [100, 512, 512]
output_segs = np.zeros([ims.shape[0] - (in_shape[0] - out_shape[0]),
                        ims.shape[1] - (in_shape[1] - out_shape[1]),
                        ims.shape[2] - (in_shape[2] - out_shape[2])], np.float32)
output_stds = np.zeros(output_segs.shape, np.float32)

# inference loop
repeat_times = 10
with torch.no_grad():
	for z in list(np.arange(0, ims.shape[0] - in_shape[0], stride[0])) + [ims.shape[0] - in_shape[0]]:
		for y in list(np.arange(0, ims.shape[1] - in_shape[1], stride[1])) + [ims.shape[1] - in_shape[1]]:
			for x in list(np.arange(0, ims.shape[2] - in_shape[2], stride[2])) + [ims.shape[2] - in_shape[2]]:
				im = ims[z : z + in_shape[0], y : y + in_shape[1], x : x + in_shape[2]]
				im = im.astype(np.float32) / 255.0
				im = np.expand_dims(im, axis=1)
				im = torch.Tensor(im).cuda()
				
				preds = []
				for i in range(repeat_times):
					pred = model(im)
					pred = torch.nn.functional.pad(pred, tuple([-32]*4))
				
					pred = torch.nn.functional.softmax(pred, dim=1)
					pred = pred[:,1]
					pred = np.squeeze(pred.data.cpu().numpy())
					
					pred = np.expand_dims(pred, axis=0)
					preds.append(pred)
				
				preds = np.concatenate(preds, axis=0)
				pred_mean = np.mean(preds, axis=0)
				pred_std = np.std(preds, axis=0)
				
				output_segs[z : z + out_shape[0], y : y + out_shape[1], x : x + out_shape[2]] = pred_mean
				output_stds[z : z + out_shape[0], y : y + out_shape[1], x : x + out_shape[2]] = pred_std

np.save('model-019000-segs-train.npy', output_segs)
np.save('model-019000-stds-train.npy', output_stds)
