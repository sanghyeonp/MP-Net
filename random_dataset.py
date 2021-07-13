import os
import random
import shutil

copy2 = os.path.join(os.getcwd(), 'dataset')	# Images divided into datasets will be saved here.
fl, m = os.listdir(os.path.join(os.getcwd(), 'data'))
fl_dir, mask_dir = os.path.join(os.getcwd(), 'data', fl), os.path.join(os.getcwd(), 'data', 'mask_patches')
imgs = sorted(os.listdir(fl_dir))

counter = 1
dataset_counter = 2
seen = []
dataset = {1:[], 2:[], 3:[], 4:[], 5:[]} 

# Divide the images into dataset 2, 3, 4, and 5 randomly.
# Each datasets aforementioned will have 15 images.
while counter <= (73 - 13):
	img = random.choice(imgs)
	while img in seen:
		img = random.choice(imgs)
	seen.append(img)

	if dataset_counter == 2:
		dataset[2].append(img)
	elif dataset_counter == 3:
		dataset[3].append(img)
	elif dataset_counter == 4:
		dataset[4].append(img)
	elif dataset_counter == 5:
		dataset[5].append(img)

	dataset_counter += 1

	if dataset_counter > 5:
		dataset_counter = 2

	counter += 1

# From total of 73 images, 60 images were evenly divided into 4 different datasets.
# The rest of images will be copied into dataset 1, which is used as test set.
for img in imgs:
	if img not in seen:
		dataset[1].append(img)

# Copying of the randomly divided images will be performed.
for n in dataset.keys():
	for img in dataset[n]:
		copy2_dir = os.path.join(copy2, 'dataset_{}'.format(n))
		if not os.path.exists(copy2_dir):
			os.mkdir(copy2_dir)
		copy2_labels_dir = os.path.join(copy2, 'dataset_{}'.format(n), 'labels')
		if not os.path.exists(copy2_labels_dir):
			os.mkdir(copy2_labels_dir)
		
		shutil.copy(os.path.join(fl_dir, img), os.path.join(copy2_dir, img))
		shutil.copy(os.path.join(mask_dir, img), os.path.join(copy2_labels_dir, img))
