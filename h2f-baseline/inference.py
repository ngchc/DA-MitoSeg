import os
import torch
import numpy as np
from PIL import Image
from rsunet2d import RSUNet
from collections import OrderedDict

from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef


# load ims
files = os.listdir('../fafb-valid/im')
files.sort()
ims = []
for i in range(len(files)):
	im = np.asarray(Image.open(os.path.join('../fafb-valid/im', files[i])))
	im = np.expand_dims(im, axis=0)
	ims.append(im)
ims = np.concatenate(ims, axis=0)

# load segs
files = os.listdir('../fafb-valid/seg')
files.sort()
segs = []
for i in range(len(files)):
	seg = np.array(Image.open(os.path.join('../fafb-valid/seg', files[i])))
	
	seg = seg[:,:,0]
	seg[seg>=128] = 255
	seg[seg<128] = 0
	seg = seg // 255
	
	seg = np.expand_dims(seg, axis=0)
	segs.append(seg)
segs = np.concatenate(segs, axis=0)

# build model
model = RSUNet([16,32,48,64,80]).cuda()
model = model.eval()

# restore weights
weight = '../h-oracle/model-human.pth'
new_state_dict = OrderedDict()
state_dict = torch.load(weight)
for k, v in state_dict.items():
	name = k[7:] # remove module.
	new_state_dict[name] = v
model.load_state_dict(new_state_dict)

if torch.cuda.device_count() > 1:
	model = torch.nn.DataParallel(model)

# pad for overlap
bd = 39
ims = np.pad(ims, ((0,0),(bd,bd),(bd,bd)), mode='symmetric')

# settings
in_shape = [8, 512, 512]
stride = out_shape = [8, 434, 434]
output_segs = np.zeros([ims.shape[0] - (in_shape[0] - out_shape[0]),
                        ims.shape[1] - (in_shape[1] - out_shape[1]),
                        ims.shape[2] - (in_shape[2] - out_shape[2])], np.float32)
# inference loop
with torch.no_grad():
	for z in list(np.arange(0, ims.shape[0] - in_shape[0], stride[0])) + [ims.shape[0] - in_shape[0]]:
		for y in list(np.arange(0, ims.shape[1] - in_shape[1], stride[1])) + [ims.shape[1] - in_shape[1]]:
			for x in list(np.arange(0, ims.shape[2] - in_shape[2], stride[2])) + [ims.shape[2] - in_shape[2]]:
				im = ims[z : z + in_shape[0], y : y + in_shape[1], x : x + in_shape[2]]
				im = im.astype(np.float32) / 255.0
				im = np.expand_dims(im, axis=1)
				im = torch.Tensor(im).cuda()
				
				pred = model(im)
				pred = torch.nn.functional.pad(pred, tuple([-bd]*4))
				
				pred = torch.nn.functional.softmax(pred, dim=1)
				pred = pred[:,1]
				pred = np.squeeze(pred.data.cpu().numpy())
				
				output_segs[z : z + out_shape[0], y : y + out_shape[1], x : x + out_shape[2]] = pred
del ims

# measure prediction
serial_segs = segs.reshape(-1)
mAP = average_precision_score(serial_segs, output_segs.reshape(-1))

bin_segs = output_segs
bin_segs[bin_segs>=0.5] = 1
bin_segs[bin_segs<0.5] = 0
serial_bin_segs = bin_segs.reshape(-1)

intersection = np.logical_and(serial_segs==1, serial_bin_segs==1)
union = np.logical_or(serial_segs==1, serial_bin_segs==1)
IoU = np.sum(intersection) / np.sum(union)

F1 = f1_score(serial_segs, serial_bin_segs)
MCC = matthews_corrcoef(serial_segs, serial_bin_segs)
print('mAP=%.4f, F1=%.4f, MCC=%.4f, IoU=%.4f' % (mAP, F1, MCC, IoU))
# mAP=0.8562, F1=0.6910, MCC=0.6905, IoU=0.5278
