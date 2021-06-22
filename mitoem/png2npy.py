import os
import numpy as np
from tqdm import tqdm
from PIL import Image


files = os.listdir('im')
files.sort()
files = files[:400]

ims = np.zeros([400,4096,4096], dtype=np.uint8)
for i in tqdm(range(len(files))):
	ims[i] = np.asarray(Image.open(os.path.join('im', files[i])))

np.save('mito_human_train_ims.npy', ims)
