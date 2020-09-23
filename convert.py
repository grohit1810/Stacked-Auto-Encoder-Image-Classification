import os
import numpy as np
from scipy import misc
from PIL import Image
from random import shuffle
import pickle

EXAMPLES = 2012

X = [[] for i in range(EXAMPLES)]
Y = [[0, 0, 0, 0, 0] for i in range(EXAMPLES)]

des = './compressed/'
folders = ['Leopards', 'airplanes']
bw_folders =  ['car_side', 'Motorbikes', 'butterfly']

i = 0

for folder in folders:
	des_in = des + folder
	for r, _, f in os.walk(des_in):
		for image in f:
			img = Image.open(r + '/' + image)
			arr = np.array(img)
			for y in arr:
				for x in y:
					for rgb in x:
						X[i].append(rgb/255.0)
			print i, r + '/' + image
			Y[i][folders.index(folder)] = 1
			i += 1

def to_rgb1(im):
	# I think this will be slow
	w, h = im.shape
	ret = np.empty((w, h, 3), dtype=np.uint8)
	ret[:, :, 0] = im
	ret[:, :, 1] = im
	ret[:, :, 2] = im
	return ret

for folder in bw_folders:
	des_in = des + folder
	for r, _, f in os.walk(des_in):
		for image in f:
			img = Image.open(r + '/' + image)
			bw_arr = np.array(img)
			if isinstance(bw_arr[0][0], np.uint8):
				arr = to_rgb1(bw_arr)
			else:
				arr = bw_arr
			for y in arr:
				for x in y:
					for rgb in x:
						X[i].append(rgb/255.0)
			print i, r + '/' + image
			Y[i][bw_folders.index(folder)+2] = 1
			i += 1


#pickle.dump(X,open("i.pkl","wb"))
pickle.dump(Y,open("o.pkl","wb"))

print 'done'
