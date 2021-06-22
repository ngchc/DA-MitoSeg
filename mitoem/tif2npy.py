import os
import numpy as np
from tqdm import tqdm
from PIL import Image


seg_folder = 'mito_train'
files = os.listdir(seg_folder)
files.sort()

segs = np.zeros([400,4096,4096], dtype=np.uint16)
for i in tqdm(range(len(files))):
	segs[i] = np.asarray(Image.open(os.path.join(seg_folder, files[i])))

np.save('mito_human_train_segs.npy', segs)
