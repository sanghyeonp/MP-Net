import sys
import os
import csv
import argparse
from datetime import datetime
from tqdm import tqdm
import albumentations as A
import torch
import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d
from torch.utils.data import DataLoader
from torchvision.models.segmentation import fcn_resnet101, deeplabv3_resnet101
from src.logging import Logger
import src.segmentation_models.segmentation_models_pytorch as smp
from src.pytorch_nested_unet.archs import NestedUNet
from src.functional.loss import DiceLoss, DiceBCELoss
from src.segmentation_models.segmentation_models_pytorch.utils.metrics import Accuracy, IoU, Recall, Fscore, Precision
from src.segmentation_models.segmentation_models_pytorch.utils.base import Activation
from src.functional.preprocess import Microplastic_data
from src.functional.fit_unet import evaluate as evaluate_unet
from src.functional.fit_unet import train_model as train_unet
from src.functional.fit_torchvision_model import evaluate as evaluate_torchvision
from src.functional.fit_torchvision_model import train_model as train_torchvision
from src.functional.evaluation_unet import testset_evaluation as testset_evaluation_unet
from src.functional.evaluation_torchvision_model import testset_evaluation as testset_evaluation_torchvision


def parse_args():
	parser = argparse.ArgumentParser(description='Microplastics Segmentation')

	parser.add_argument('--model', type=str, required=True, choices=['unet', 'fcn', 'deeplabv3', 'nested_unet'], 
	 						help='Specify the model (U-Net, FCN, Deeplabv3, or U-Net++).')
	parser.add_argument('--train', action='store_true',
						help='Specify whether to train the model (default=False).')
	parser.add_argument('--weights', type=str, default=None,
						help='Provide absolute path of pre-trained weights (default=None).')
	parser.add_argument('--test', action='store_true',
						help='Specify whether to evaluate the model (default=False).')
	parser.add_argument('--epoch', type=int, default=20,
						help='Specify number of epochs (default=20).')
	parser.add_argument('--batch_size', type=int, default=10,
						help='Specify the batch size (default=10).')
	parser.add_argument('--criterion', type=str, choices=['bce', 'dice', 'dicebce'], default='dice', 
	 					help='Specify the loss function (BCEWithLogits loss, SoftDice loss, DiceBCE loss).')
	parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], default='sgd', 
	 					help='Specify the optimizer (SGD, ADAM).')
	parser.add_argument('--momentum', type=float, default=0.9,
						help='Specify the momentum value for SGD optimizer (default=0.9).')
	parser.add_argument('--lr', type=float, default=0.001,
						help='Specify the learning rate (default=0.001).')
	parser.add_argument('--TTA', nargs='+', choices=['B', 'C', 'HSV'], default=None,
						help='Specify the image augmentation being used for test-time augmentation (default=None).')

	args = parser.parse_args()
	return args


def main(model, train, weights, test, optimizer, momentum, lr, epoch, batch_size, criterion, TTA):

	torch.manual_seed(0)
	model_name, loss_name, optim_name = model, criterion, optimizer
	
	# Initialize image augmentation
	TTA_is = TTA
	augs = {'B' : A.RandomBrightness(limit=[-0.2, 0.2], always_apply=False, p=0.5), \
			'C' : A.RandomContrast(limit=[0.2, 0.6], always_apply=False, p=0.5), \
			'HSV' : A.HueSaturationValue(hue_shift_limit=[-10, -10], sat_shift_limit=[50, 50], val_shift_limit=[10, 50], always_apply=False, p=0.5) \
			}
	TTA_name = None
	if TTA is not None:
		TTA = [augs[i] for i in TTA]
		for tran in TTA_is:
			if TTA_name is None:
				TTA_name = tran
			else:
				TTA_name += ('_' + tran)

	time = datetime.now()
	t = time.strftime("%Y-%m-%d-%H-%M-%S")
	if train:
		filename = '{}_[{}]_[{}]_[{}]_[{}_{}]_TTA[{}]'.format(t, model_name, criterion, optimizer, epoch, lr, TTA_name)
	else:
		if weights is None:
			raise AttributeError("Need to provide pre-trained weight if performing only testing.")
		filename = '{}_[{}]_pretrained[{}]_TTA[{}]'.format(t, model_name, weights.split(sep='/')[-1], TTA_name)

	if not os.path.exists(os.path.join(os.getcwd(), 'result')):
		os.mkdir(os.path.join(os.getcwd(), 'result'))
	if not os.path.exists(os.path.join(os.getcwd(), 'result', 'train_result')):
		os.mkdir(os.path.join(os.getcwd(), 'result', 'train_result'))
	if not os.path.exists(os.path.join(os.getcwd(), 'result', 'train_result', model_name)):
		os.mkdir(os.path.join(os.getcwd(), 'result', 'train_result', model_name))
	if not os.path.exists(os.path.join(os.getcwd(), 'result', 'model_saved')):
		os.mkdir(os.path.join(os.getcwd(), 'result', 'model_saved'))
	if not os.path.exists(os.path.join(os.getcwd(), 'result', 'model_saved', model_name)):
		os.mkdir(os.path.join(os.getcwd(), 'result', 'model_saved', model_name))
	if not os.path.exists(os.path.join(os.getcwd(), 'result', 'pred_mask')):
		os.mkdir(os.path.join(os.getcwd(), 'result', 'pred_mask'))
	if not os.path.exists(os.path.join(os.getcwd(), 'result', 'pred_mask', model_name)):
		os.mkdir(os.path.join(os.getcwd(), 'result', 'pred_mask', model_name))
	if not os.path.exists(os.path.join(os.getcwd(), 'result', 'pred_mask', model_name, filename)):
		os.mkdir(os.path.join(os.getcwd(), 'result', 'pred_mask', model_name, filename))
	if not os.path.exists(os.path.join(os.getcwd(), 'result', 'evaluation')):
		os.mkdir(os.path.join(os.getcwd(), 'result', 'evaluation'))
	if not os.path.exists(os.path.join(os.getcwd(), 'result', 'evaluation', model_name)):
		os.mkdir(os.path.join(os.getcwd(), 'result', 'evaluation', model_name))
	if not os.path.exists(os.path.join(os.getcwd(), 'result', 'stdout')):
		os.mkdir(os.path.join(os.getcwd(), 'result', 'stdout'))
	if not os.path.exists(os.path.join(os.getcwd(), 'result', 'stdout', model_name)):
		os.mkdir(os.path.join(os.getcwd(), 'result', 'stdout', model_name))

	sys.stdout = Logger(os.path.join(os.getcwd(), 'result', 'stdout', model_name, filename+'.txt'))

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	evaluation_metrics = [	Accuracy(balanced=True, threshold=None).to(device), Recall(threshold=None).to(device), 
							Precision(threshold=None).to(device), Fscore(threshold=None).to(device), IoU(threshold=None).to(device)]
	# Initialize model
	if model_name == 'unet':		# Initialize U-Net model
		model = smp.Unet(encoder_name="resnet101", in_channels=3, classes=1, encoder_weights="imagenet")
		evaluate, train_model, testset_evaluation = evaluate_unet, train_unet, testset_evaluation_unet
	elif model_name == 'fcn':	# Initialize FCN model
		model = fcn_resnet101(pretrained=True)
		# The last convolutional layer is modified to produce a binary output.
		model.classifier._modules['4'] = Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
		model.aux_classifier._modules['4'] = Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
		evaluate, train_model, testset_evaluation = evaluate_torchvision, train_torchvision, testset_evaluation_torchvision
	elif model_name == 'deeplabv3':	# Initialize Deeplabv3 model
		model = deeplabv3_resnet101(pretrained=True)
		# The last convolutional layer is modified to produce a binary output.
		model.classifier._modules['4'] = Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
		model.aux_classifier._modules['4'] = Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
		evaluate, train_model, testset_evaluation = evaluate_torchvision, train_torchvision, testset_evaluation_torchvision
	elif model_name == 'nested_unet':
		model = NestedUNet(num_classes=1, input_channels=3, deep_supervision=False)
		evaluate, train_model, testset_evaluation = evaluate_unet, train_unet, testset_evaluation_unet

	if train:

		print("Initiating... Model [{}] Loss function [{}] Optimizer [{}]".format(model_name, loss_name, optim_name))		
		print("File name [{}]".format(filename))

		if weights is not None:
			model.load_state_dict(torch.load(weights))

		with open(os.path.join(os.getcwd(), 'result', 'train_result', model_name, filename+'.csv'), 'wt', newline='') as f1:
			f1_writer = csv.writer(f1)
			f1_writer.writerow(['Epoch', 'Train loss', 'Val loss', 'Accuracy', 'Recall', 'Precision', 'F1-score', 'IoU'])

			# Initialize optimizer
			if optimizer == 'adam':
				optimizer = torch.optim.Adam(model.parameters(), lr=lr)
			elif optimizer == 'sgd':
				optimizer = torch.optim.SGD(model.parameters(), momentum=momentum, lr=lr)

			# Initialize criterion
			if criterion == 'bce':
				criterion = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor(9))
			elif criterion == 'dice':
				criterion = DiceLoss()
			elif criterion == 'dicebce':
				criterion = DiceBCELoss()

			# Separate TTA for training and testing. TTA used for training always have random Flip.
			if TTA is not None:
				TTA4train = TTA.copy()
				TTA4train.append(A.Flip(p=0.5))
			else:
				TTA4train = TTA

			activation = Activation(activation='sigmoid') # U-Net architecture is not initialized with activation at the end

			model.to(device)
			criterion.to(device)
			activation.to(device)

			train_set = Microplastic_data(path=os.path.join(os.getcwd(), 'dataset', 'train'), transform=TTA4train)
			train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
			val_set = Microplastic_data(path=os.path.join(os.getcwd(), 'dataset', 'validation'), transform=TTA4train)
			val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

			print("Time: {}".format(time))
			print("Start training...\n")
			val_loss, val_performances = evaluate(	model=model, device=device, total_epoch=epoch, epoch=0, val_loader=val_loader, \
													activation=activation, criterion=criterion, metrics=evaluation_metrics \
													)
			f1_writer.writerow([0, 'NA', val_loss] +  val_performances)

			best_epoch, evaluations = train_model(	model=model, device=device, total_epoch=epoch,\
													train_loader=train_loader, val_loader=val_loader, \
													criterion=criterion, optimizer=optimizer, metrics=evaluation_metrics, \
													activation=activation,	writer=f1_writer, filename=filename, \
													save2=os.path.join(os.getcwd(), 'result', 'model_saved', model_name)\
													)
			print("\nFinished training...")
			print("Time: {}\n".format(datetime.now()))
			print("Training result for [{}]:".format(filename))
			tqdm.write("Model saved at best epoch [{}]\nValidation loss [{:.4f}]\nAccuracy [{:.4f}]\nRecall [{:.4f}]\nPrecision [{:.4f}]\
						\nF1-score [{:.4f}]\nIoU [{:.4f}]\n".format(best_epoch, evaluations[0], evaluations[1], evaluations[2], \
																	evaluations[3], evaluations[4], evaluations[5] \
																	) \
						)
		
		if test:	# Train 및 test 다 True 일 때는, training 할때 save된 parameter로 test 하기
			print("Evaluating...")

			with open(os.path.join(os.getcwd(), 'result', 'evaluation', model_name, filename+'.csv'), 'wt', newline='') as f2:
				f2_writer =csv.writer(f2)
				f2_writer.writerow(['Image number', 'Accuracy', 'Recall', 'Precision', 'F1-score', 'IoU', '', 'TP', 'FP', 'FN', 'TN'])

				performances, confusion = testset_evaluation(	model=model, device=device, testset_path=os.path.join(os.getcwd(), 'dataset', 'test'), \
																weight=os.path.join(os.getcwd(), 'result', 'model_saved', model_name, filename+'.pth'), \
																metrics=evaluation_metrics, save2=os.path.join(os.getcwd(), 'result', 'pred_mask', model_name, filename), \
																write2=f2_writer, TTA=TTA \
																)
		
			print("Finished evaluation...\n")
			print("Evaluation result for [{}]:".format(filename))
			print("Accuracy [{:.4f}]\nRecall [{:.4f}]\nPrecision [{:.4f}]\nF1-score [{:.4f}]\nIoU [{:.4f}]\
					\nTP [{}] | FP [{}] | FN [{}] | TN [{}]\n".format(performances[0], performances[1], performances[2], performances[3], performances[4], \
																	confusion[0], confusion[1], confusion[2], confusion[3]) \
					)

	else:	# If training is not being done. Load saved model.
		if test:
			print("Evaluating...")
			with open(os.path.join(os.getcwd(), 'result', 'evaluation', model_name, filename+'.csv'), 'wt', newline='') as f2:
				f2_writer =csv.writer(f2)
				f2_writer.writerow(['Image number', 'Accuracy', 'Recall', 'Precision', 'F1-score', 'IoU', '', 'TP', 'FP', 'FN', 'TN'])

				performances, confusion = testset_evaluation(	model=model, device=device, testset_path=os.path.join(os.getcwd(), 'dataset', 'test'), \
																weight=weights, metrics=evaluation_metrics, \
																save2=os.path.join(os.getcwd(), 'result', 'pred_mask', model_name, filename), \
																write2=f2_writer, TTA=TTA \
																)
				
			print("Finished evaluation...\n")
			print("Evaluation result for [{}]".format(filename))
			print("Accuracy [{:.4f}]\nRecall [{:.4f}]\nPrecision [{:.4f}]\nF1-score [{:.4f}]\nIoU [{:.4f}]\
					\nTP [{}] | FP [{}] | FN [{}] | TN [{}]\n".format(performances[0], performances[1], performances[2], performances[3], performances[4], \
																	confusion[0], confusion[1], confusion[2], confusion[3]) \
					)


if __name__ == '__main__':
	args = parse_args()

	main(	model=args.model, train=args.train, weights=args.weights, test=args.test, epoch=args.epoch, \
			batch_size=args.batch_size, criterion=args.criterion, optimizer=args.optimizer, momentum=args.momentum, \
			lr=args.lr, TTA=args.TTA)
