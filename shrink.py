import os
from PIL import Image

src = './Dataset/'
des = './shrink/'
track = 0

os.mkdir(des)

folders = ['Leopards', 'airplanes', 'car_side', 'butterfly', 'Motorbikes']
for i in folders:
	os.mkdir(des + i)

for r, d, f in os.walk(src):
	des_in = des + r.split('/')[-1]
	for i in f:
		track += 1
		print track
		foo = Image.open(r + '/' + i)
		foo = foo.resize((16, 16), Image.ANTIALIAS)
		foo.save(des_in + '/' + i, optimize=True, quality=100)
