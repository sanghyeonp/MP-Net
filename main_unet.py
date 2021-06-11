# loss.py from https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
# unet arch. and evaluation metrics from https://github.com/qubvel/segmentation_models.pytorch
# Accuracy in segmentation_models.segmentation_models_pytorch.utils.metrics.py was modified to calculate balanced accuracy
import code
import os
import torch
import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d
import src.segmentation_models.segmentation_models_pytorch as smp
from torchvision.models.segmentation import fcn_resnet101, deeplabv3_resnet101
from src.pytorch_nested_unet.archs import NestedUNet
from src.functional.loss import DiceLoss, DiceBCELoss
from src.segmentation_models.segmentation_models_pytorch.utils.metrics import Accuracy, IoU, Recall, Fscore, Precision
from src.segmentation_models.segmentation_models_pytorch.utils.base import Activation
from src.functional.preprocess import Microplastic_data


def main(model, train=True, weights=None, test=True, optimizer='adam', epoch=20, batch_size=10, criterion='dice', transformation=None):
	"""
	model : choose model from either 'unet', 'fcn', 'deeplabv3', 'nested_unet'
			these models will be pre-trained models
			'unet' will have ResNet101 as encoder backbone
			'fcn' and 'deeplabv3' will have their classification layer
	
	criterion : choose loss function from either 'bce', 'dice', 'dicebce'

	optimizer : choose optimization method from either 'adam', 'sgd'
	"""
	if train:
		# Create saving location
		if not os.path.exists(os.path.join(os.getcwd(), 'train_log')):
			os.mkdir(os.path.join(os.getcwd(), 'train_log'))
			os.mkdir(os.path.join(os.getcwd(), 'train_log', '{}'.format(model)))


		device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		evaluation_metrics = [	Accuracy(balanced=True, threshold=None).to(device), Recall(threshold=None).to(device), 
								Precision(threshold=None).to(device), Fscore(threshold=None).to(device), IoU(threshold=None).to(device)]

		# Initialize model
		if model == 'unet':		# Initialize U-Net model
			model = smp.Unet(encoder_name="resnet101", in_channels=3, classes=1, encoder_weights="imagenet")
		elif model == 'fcn':	# Initialize FCN model
			model = fcn_resnet101(pretrained=True)
			# The last convolutional layer is modified to produce a binary output.
			model.classifier._modules['4'] = Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
			model.aux_classifier._modules['4'] = Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
		elif model == 'deeplabv3':	# Initialize Deeplabv3 model
			model = deeplabv3_resnet101(pretrained=True)
			# The last convolutional layer is modified to produce a binary output.
			model.classifier._modules['4'] = Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
			model.aux_classifier._modules['4'] = Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
		elif model == 'nested_unet':
			model = NestedUNet(num_classes=1, input_channels=3, deep_supervision=False)

		# Initialize optimizer
		if optimizer == 'adam':
			optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
		elif optimizer == 'sgd':
			optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=0.001)

		# Initialize criterion
		if criterion == 'bce':
			criterion = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor(9))
		elif criterion == 'dice':
			criterion = DiceLoss()
		elif criterion == 'dicebce':
			criterion = DiceBCELoss()

		activation = Activation(activation='sigmoid') # U-Net architecture is not initialized with activation at the end

		model.to(device)
		criterion.to(device)
		activation.to(device)

		train_set = Microplastic_data(os.path.join(os.getcwd(), 'dataset', 'train'))
		train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
		val_set = Microplastic_data(os.path.join(os.getcwd(), 'dataset', 'validation'))
		val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
		

	else:	# If training is not being done. Load saved model.
		if weights is None:
			pass # Raise error

		pass



	# with open(os.path.join(os.getcwd(), 'train_log', ))


if __name__ == '__main__':
	# parameters
	criterion = 'dice'
	epoch = 20
	batch_size = 10
	transformation = None
	main(model='fcn')
	# code.interact(local=dict(globals(), **locals()))
