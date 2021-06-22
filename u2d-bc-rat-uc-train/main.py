import os
import re
import cv2
import argparse
import logging
import numpy as np
from time import time
from datetime import datetime
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rsunet2d_uc import RSUNet
from data2d import Provider, Valid
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef


parser = argparse.ArgumentParser()
# project settings
parser.add_argument('-np', '--num-process', type=int, default=12)
parser.add_argument('-df', '--display-freq', type=int, default=100)
parser.add_argument('-vf', '--valid-freq', type=int, default=500)
parser.add_argument('-sf', '--save-freq', type=int, default=500)
parser.add_argument('-sp', '--save-path', type=str, default='./models_p0.5')
parser.add_argument('-cp', '--cache-path', type=str, default='./caches_p0.5')
parser.add_argument('-re', '--resume', action='store_true', default=False)
# training settings
parser.add_argument('-bs', '--batch-size', type=int, default=64)
parser.add_argument('-wd', '--weight-decay', type=float, default=None)
parser.add_argument('-ti', '--total-iters', type=int, default=20000)
parser.add_argument('-di', '--decay-iters', type=int, default=20000)
parser.add_argument('-wi', '--warmup-iters', type=int, default=1000)
parser.add_argument('-bl', '--base-lr', type=float, default=0.001)
parser.add_argument('-el', '--end-lr', type=float, default=0.0001)
parser.add_argument('-pw', '--power', type=float, default=2.0)
parser.add_argument('-dp', '--drop-ratio', type=float, default=0.5)
opt = parser.parse_args()

# enable cudnn
#torch.backends.cudnn.benchmark = True

def init_project():
	def init_logging(path):
		logging.basicConfig(
			    level    = logging.INFO,
			    format   = '%(message)s',
			    datefmt  = '%m-%d %H:%M',
			    filename = path,
			    filemode = 'w')
	
		# define a Handler which writes INFO messages or higher to the sys.stderr
		console = logging.StreamHandler()
		console.setLevel(logging.INFO)
	
		# set a format which is simpler for console use
		formatter = logging.Formatter('%(message)s')
		# tell the handler to use this format
		console.setFormatter(formatter)
		logging.getLogger('').addHandler(console)
	
	if torch.cuda.is_available() is False:
		raise AttributeError('No GPU available')
	
	prefix = datetime.now().strftime('%Y.%m.%d-%H:%M:%S')
	if opt.resume is False:
		if not os.path.exists(opt.save_path):
			os.makedirs(opt.save_path)
		if not os.path.exists(opt.cache_path):
			os.makedirs(opt.cache_path)
	init_logging(os.path.join(opt.save_path, prefix + '.log'))
	logging.info(opt)


def load_dataset():
	print('Caching datasets ... ', end='', flush=True)
	t1 = time()
	train_provider = Provider('../mitoem/mito_rat_train', batch_size=opt.batch_size, num_workers=opt.num_process)
	valid = Valid('../mitoem/mito_rat_valid')
	print('Done (time: %.2fs)' % (time() - t1))
	return train_provider, valid


def build_model():
	print('Building model on ', end='', flush=True)
	t1 = time()
	device = torch.device('cuda:0')
	model = RSUNet([16,32,48,64,80], opt.drop_ratio).to(device)
	
	cuda_count = torch.cuda.device_count()
	if cuda_count > 1:
		print('%d GPUs ... ' % cuda_count, end='', flush=True)
		model = nn.DataParallel(model)
	else:
		print('a single GPU ... ', end='', flush=True)
	print('Done (time: %.2fs)' % (time() - t1))
	return model


def resume_params(model, optimizer, resume):
	if resume:
		t1 = time()
		last_iter = 0
		for files in os.listdir(opt.save_path):
			if 'model' in files:
				it = int(re.sub('\D', '', files))
				if it > last_iter:
					last_iter = it
		model_path = os.path.join(opt.save_path, 'model-%d.ckpt' % last_iter)
		
		print('Resuming weights from %s ... ' % model_path, end='', flush=True)
		if os.path.isfile(model_path):
			checkpoint = torch.load(model_path)
			model.load_state_dict(checkpoint['model_weights'])
			optimizer.load_state_dict(checkpoint['optimizer_weights'])
		else:
			raise AttributeError('No checkpoint found at %s' % filename)
		print('Done (time: %.2fs)' % (time() - t1))
		print('valid %d, loss = %.4f' % (checkpoint['current_iter'], checkpoint['valid_result']))
		return model, optimizer, checkpoint['current_iter']
	else:
		return model, optimizer, 0


def loop(train_provider, valid, model, optimizer, iters):
	def calculate_lr(iters):
		if iters < opt.warmup_iters:
			current_lr = (opt.base_lr - opt.end_lr) * pow(float(iters) / opt.warmup_iters, opt.power) + opt.end_lr
		else:
			if iters < opt.decay_iters:
				current_lr = (opt.base_lr - opt.end_lr) * pow(1 - float(iters - opt.warmup_iters) / opt.decay_iters, opt.power) + opt.end_lr
			else:
				current_lr = opt.end_lr
		return current_lr
	
	def valid_step(iters, model, valid):
		model = model.eval()
		ims, segs = valid.ims, valid.segs
		
		""" quick evaluation """
		ims = ims[:,:1024,:1024]
		segs = segs[:,:1024,:1024]
		
		in_shape = stride = out_shape = [100, 256, 256]
		output_segs = np.zeros([ims.shape[0] - (in_shape[0] - out_shape[0]),
		                        ims.shape[1] - (in_shape[1] - out_shape[1]),
		                        ims.shape[2] - (in_shape[2] - out_shape[2])], np.float32)
		with torch.no_grad():
			for z in list(np.arange(0, ims.shape[0] - in_shape[0], stride[0])) + [ims.shape[0] - in_shape[0]]:
				for y in list(np.arange(0, ims.shape[1] - in_shape[1], stride[1])) + [ims.shape[1] - in_shape[1]]:
					for x in list(np.arange(0, ims.shape[2] - in_shape[2], stride[2])) + [ims.shape[2] - in_shape[2]]:
						im = ims[z : z + in_shape[0], y : y + in_shape[1], x : x + in_shape[2]]
						im = im.astype(np.float32) / 255.0
						im = np.expand_dims(im, axis=1)
						im = torch.Tensor(im).cuda()
						
						pred = model(im)
						pred = torch.nn.functional.softmax(pred, dim=1)
						pred = pred.data.cpu().numpy()
						pred = pred[:,1]
						
						output_segs[z : z + out_shape[0], y : y + out_shape[1], x : x + out_shape[2]] = pred
		
		# metrics
		serial_segs = segs.reshape(-1)
		mAP = average_precision_score(serial_segs, output_segs.reshape(-1))
		
		#bin_segs = output_segs.copy()
		#bin_segs[bin_segs>=0.5] = 1
		#bin_segs[bin_segs<0.5] = 0
		#serial_bin_segs = bin_segs.reshape(-1)
		#F1 = f1_score(serial_segs, serial_bin_segs)
		#MCC = matthews_corrcoef(serial_segs, serial_bin_segs)
		
		# snapshot
		cache_folder = os.path.join(opt.cache_path, '%06d'%iters)
		os.makedirs(cache_folder)
		
		output_segs = (output_segs*255).astype(np.uint8)
		mix = np.concatenate([ims,segs*255,output_segs], axis=2)
		for i in range(mix.shape[0]):
			ds_mix = cv2.resize(mix[i], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
			Image.fromarray(ds_mix).save(os.path.join(cache_folder, '%02d.png'%i))
		return mAP
		#return mAP, F1, MCC
	
	# valid 0
	mAP = valid_step(0, model, valid)
	logging.info('valid %d, mAP=%.4f' % (0, mAP))
	#mAP, F1, MCC = valid_step(0, model, valid)
	#logging.info('valid %d, mAP=%.4f, F1=%.4f, MCC=%.4f' % (0, mAP, F1, MCC))
	
	rcd_time = []
	sum_time = 0
	sum_loss = 0
	cn_loss = nn.CrossEntropyLoss()
	while iters <= opt.total_iters:
		# train
		iters += 1
		t1 = time()
		im, seg = train_provider.next()
		
		# decay learning rate
		if opt.end_lr == opt.base_lr:
			current_lr = opt.base_lr
		else:
			current_lr = calculate_lr(iters)
			for param_group in optimizer.param_groups:
				param_group['lr'] = current_lr
		
		optimizer.zero_grad()
		model = model.train()
		
		out = model(im)
		
		loss = cn_loss(out, seg)
		loss.backward()
		if opt.weight_decay is not None:
			for group in optimizer.param_groups:
				for param in group['params']:
					param.data = param.data.add(-opt.weight_decay * group['lr'], param.data)
		optimizer.step()
		
		sum_loss += loss.item()
		sum_time += time() - t1
		
		# log train
		wt = 100
		df = opt.display_freq
		if iters % df == 0:
			rcd_time.append(sum_time)
			logging.info('step %d, loss=%.4f (wt:*%d, lr:%.8f, et:%.2f sec, rd:%.2f min)'
		                 % (iters, sum_loss/df*wt, wt, current_lr, sum_time,
			               (opt.total_iters - iters) / df * np.mean(np.asarray(rcd_time)) / 60))
			sum_time = 0
			sum_loss = 0
		
		# valid
		if iters % opt.valid_freq == 0:
			mAP = valid_step(iters, model, valid)
			logging.info('valid %d, mAP=%.4f' % (iters, mAP))
			#mAP, F1, MCC = valid_step(iters, model, valid)
			#logging.info('valid %d, mAP=%.4f, F1=%.4f, MCC=%.4f' % (iters, mAP, F1, MCC))
		
		# save
		if iters % opt.save_freq == 0:
			torch.save(model.state_dict(), os.path.join(opt.save_path, 'model-%06d-%.4f.pth' % (iters, mAP)))
			torch.save(optimizer.state_dict(), os.path.join(opt.save_path, 'optim-lastest.pth'))


if __name__ == '__main__':
	init_project()
	train_provider, valid = load_dataset()
	model = build_model()
	optimizer = optim.Adam(model.parameters(), lr=opt.base_lr, betas=(0.9, 0.999), eps=1e-8, amsgrad=False)
	model, optimizer, init_iters = resume_params(model, optimizer, opt.resume)
	
	loop(train_provider, valid, model, optimizer, init_iters)
