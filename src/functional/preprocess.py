from PIL import Image
import os
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import albumentations as A


class Microplastic_data(Dataset):
	def __init__(self, path, transform=None):
		self.images = [os.path.join(path, fl) for fl in sorted(os.listdir(path)) if os.path.isfile(os.path.join(path, fl))]
		self.masks = [os.path.join(path, 'labels', mask) for mask in sorted(os.listdir(os.path.join(path, 'labels')))]
		
		normalize = transforms.Normalize(mean=[0.1034, 0.0308, 0.0346],
										 std=[0.0932, 0.0273, 0.0302])
		self.fl_final_transform = transforms.Compose([transforms.ToTensor(),
												 	normalize])
		self.mask_final_transform = transforms.Compose([	transforms.Grayscale(num_output_channels=1),
															transforms.ToTensor()
														])

		self.transform = transform

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		if self.transform is not None:
			self.transform = A.Compose(self.transform, p=0.6)
			transformation = self.transform(image=np.asarray(Image.open(self.images[idx]).convert('RGB')), mask=np.asarray(Image.open(self.masks[idx]).convert('RGB')))
			return self.fl_final_transform(Image.fromarray(transformation['image'])), self.mask_final_transform(Image.fromarray(transformation['mask']))

		return self.fl_final_transform(Image.open(self.images[idx]).convert('RGB')), self.mask_final_transform(Image.open(self.masks[idx]).convert('RGB'))

	def getNumber(self, idx):
		return self.images[idx].split(sep='/')[-1].split(sep='_')[0]

	def getImageName(self, idx):
		return self.images[idx].split(sep='/')[-1].split(sep='_')
