import torch
import os
import numpy as np
from PIL import Image
from torchvision.transforms import transforms as torch_transforms
from torchvision.transforms.functional import crop
from torchvision.utils import save_image
import albumentations as A
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# torch 0. is black

def adjustSize(pil_img, crop_size=256):
	"""
	return : PIL image
	"""
	width, height = pil_img.size

	if width % crop_size == 0 and height % crop_size == 0:
		return pil_img, 0, 0

	added_width = 0
	added_height = 0
	if width % crop_size != 0:
		added_width = crop_size - (width % crop_size)
		width = width + added_width
	if height % crop_size != 0:
		added_height = crop_size - (height % crop_size)
		height = height + added_height

	new_img = Image.new(pil_img.mode, (width, height), (0, 0, 0))
	new_img.paste(pil_img)
	return new_img, added_width, added_height


def predict(model, device, fl_path, TTA):
	# Torch image : [channel, height, width]
	# np to pil : [width, height, channel] -> PIL Image
	"""
	1 fl image for 1 cv parameters
	Produce a predictive mask corresponding to the given fl image

	ReturnL: predicted image as tensor
	"""
	# Initialize the model with saved parameters


	normalize = torch_transforms.Normalize(mean=[0.1034, 0.0308, 0.0346],
										 std=[0.0932, 0.0273, 0.0302])
	transform = torch_transforms.Compose([torch_transforms.ToTensor(),
											 	normalize])


	fl = Image.open(fl_path).convert('RGB')	# PIL Image

	adjusted_fl_pil_image, added_width, added_height = adjustSize(fl, crop_size=256)
	width, height = adjusted_fl_pil_image.size

	pred = None

	for y in range(0, height, 256):
		pred_row = None
		for x in range(0, width, 256):
			fl_crop = crop(adjusted_fl_pil_image, y, x, 256, 256)

			if TTA is False:
				pred_crop = ((model(transform(fl_crop).unsqueeze(0).to(device, dtype=torch.float32))) > 0.5).float().squeeze(0)
			else:
				pred_crops = [((model(transform(fl_crop).unsqueeze(0).to(device, dtype=torch.float32))) > 0.5).float().squeeze(0)]

				for albu_type in transformation:
					aug_transform = A.Compose([ albu_type ])
					fl_crop_transform = transform(Image.fromarray(aug_transform(image=fl_crop)['image'])).unsqueeze(0).to(device, dtype=torch.float32)
					pred_crops.append(model(fl_crop_transform).squeeze(0))
				pred_crop = ((sum(pred_crops) / (len(transformation) + 1)) > 0.5).float()

			if x + 256 == width and y + 256 == height:		# right-bottom corner
				pred_crop = pred_crop[:, :256 - added_height, :256 - added_width]
			elif x + 256 == width:		# right sides
				pred_crop = pred_crop[:, :, :256 - added_width]
					
			elif y + 256 == height:		# bottom sides
				pred_crop = pred_crop[:, :256 - added_height, :]

			if pred_row is None:
				pred_row = pred_crop
			else:	
				pred_row = torch.cat((pred_row, pred_crop), dim=2)	# (aka: C * H * W)]

		if pred is None:
			pred = pred_row
		else:
			pred = torch.cat((pred, pred_row), dim=1)

	return pred.cpu() # Model will make prediction for finding MP as 1.


def testset_evaluation(model, device, testset_path, weight, metrics, save2, write2, TTA):

	model.load_state_dict(torch.load(weight))
	model.to(device)
	model.eval()

	fl_img_names = sorted([fl for fl in os.listdir(testset_path) if fl != 'labels'])

	running_performances = np.array([0 for _ in range(len(metrics))], dtype='float64')
	running_confusion = np.array([0 for _ in range(4)], dtype='float64') 
	for fl_name in tqdm(fl_img_names, desc="Test set evaluation", leave=False):
		pred_mask = predict(model=model, device=device, fl_path=os.path.join(testset_path, fl_name), TTA=TTA)
		save_image((~(pred_mask.bool())).float(), os.path.join(save2, fl_name))

		gt_mask = torch_transforms.ToTensor()(Image.open(os.path.join(testset_path, 'labels', fl_name)).convert('L'))
		tn, fp, fn, tp = confusion_matrix((~(gt_mask.bool())).float().flatten().numpy(), pred_mask.flatten().numpy(), labels=[0, 1]).ravel()
		running_confusion += np.array([tp, fp, fn, tn])

		performance_scores = []
		for i, metric in enumerate(metrics):
			performance_score = metric(pred_mask, (~gt_mask.bool()).float()).item()
			performance_scores.append(performance_score)
		running_performances += np.array(performance_scores)
		write2.writerow([fl_name.split(sep='.')[0]] + performance_scores + [''] + [tp, fp, fn, tn])
	write2.writerow(["Mean"] + list(running_performances/len(fl_img_names)) + [''] + list(running_confusion/len(fl_img_names)))
	return list(running_performances/len(fl_img_names)), list(running_confusion/len(fl_img_names))
