import sys
import os
import csv
import shutil
import argparse
import config
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
	parser.add_argument('--model', type=str, required=True, choices=['unet', 'fcn', 'deeplabv3', 'unet++'], 
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
	 					help='Specify the loss function (BCEWithLogits loss => bce, SoftDice loss => dice, DiceBCE loss=>dicebce).')
	parser.add_argument('--pos_weight', type=float, default=9, 
	 					help='Specify the weight to positives when using BCEWithLogits loss. (default=9)')
	parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], default='sgd', 
	 					help='Specify the optimizer (SGD => sgd, ADAM => adam).')
	parser.add_argument('--momentum', type=float, default=0.9,
						help='Specify the momentum value for SGD optimizer (default=0.9).')
	parser.add_argument('--lr', type=float, default=0.001,
						help='Specify the learning rate (default=0.001).')
	parser.add_argument('--TTA', nargs='+', choices=['B', 'C', 'HSV'], default=None,
						help='Specify the image augmentation being used for test-time augmentation (default=None).')
	parser.add_argument('--cuda', type=int, default=0,
					help='Specify the cuda for GPU usage (default=0).')

	args = parser.parse_args()
	return args


def main(model, train, weights, test, optimizer, momentum, lr, epoch, batch_size, criterion, pos_weight, TTA, cuda):

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

	if criterion != 'bce':
		pos_weight = None

	if train:
		filename = '{}_[{}]_[{}]_[{}_{}]_[{}_{}_{}]_TTA[{}]'.format(t, model_name, epoch, criterion, pos_weight, optimizer, lr, momentum, TTA_name)
	else:
		if weights is None:
			raise AttributeError("Need the pre-trained weight if performing only testing.")
		filename = '{}_[{}]_pretrained[{}]_TTA[{}]'.format(t, model_name, weights.split(sep='/')[-1][:20], TTA_name)

	if not os.path.exists(os.path.join(os.getcwd(), 'result')):
		os.mkdir(os.path.join(os.getcwd(), 'result'))
	if not os.path.exists(os.path.join(os.getcwd(), 'result', 'train_result')):
			os.mkdir(os.path.join(os.getcwd(), 'result', 'train_result'))
	if not os.path.exists(os.path.join(os.getcwd(), 'result', 'model_saved')):
		os.mkdir(os.path.join(os.getcwd(), 'result', 'model_saved'))			
	if not os.path.exists(os.path.join(os.getcwd(), 'result', 'stdout')):
		os.mkdir(os.path.join(os.getcwd(), 'result', 'stdout'))
	if not os.path.exists(os.path.join(os.getcwd(), 'result', 'pred_mask')):
		os.mkdir(os.path.join(os.getcwd(), 'result', 'pred_mask'))
	if not os.path.exists(os.path.join(os.getcwd(), 'result', 'evaluation')):
		os.mkdir(os.path.join(os.getcwd(), 'result', 'evaluation'))
	if not os.path.exists(os.path.join(os.getcwd(), 'result', 'pred_mask', filename)):
		os.mkdir(os.path.join(os.getcwd(), 'result', 'pred_mask', filename))
	if not os.path.exists(os.path.join(os.getcwd(), 'result', 'evaluation', filename)):
		os.mkdir(os.path.join(os.getcwd(), 'result', 'evaluation', filename))
	if train:
		if not os.path.exists(os.path.join(os.getcwd(), 'result', 'train_result', filename)):
			os.mkdir(os.path.join(os.getcwd(), 'result', 'train_result', filename))
		if not os.path.exists(os.path.join(os.getcwd(), 'result', 'model_saved', filename)):
			os.mkdir(os.path.join(os.getcwd(), 'result', 'model_saved', filename))
		if not os.path.exists(os.path.join(os.getcwd(), 'result', 'stdout', filename)):
			os.mkdir(os.path.join(os.getcwd(), 'result', 'stdout', filename))

	device = torch.device('cuda:{}'.format(cuda) if torch.cuda.is_available() else 'cpu')
	evaluation_metrics = [	Accuracy(balanced=True, threshold=None).to(device), Recall(threshold=None).to(device), 
							Precision(threshold=None).to(device), Fscore(threshold=None).to(device), IoU(threshold=None).to(device)]

	if train:	# Train if true.
		sys.stdout = Logger(os.path.join(os.getcwd(), 'result', 'stdout', filename, filename+'.txt'))

		print("Initiating... Model [{}] Loss function [{}] Optimizer [{}]".format(model_name, loss_name, optim_name))		
		print("File name [{}]\n".format(filename))

		if config.DATASET_DIR is None:
			DATASET_DIR = os.path.join(os.getcwd(), 'dataset')
		else:
			DATASET_DIR = config.DATASET_DIR
		datasets_dir = os.listdir(DATASET_DIR)

		with open(os.path.join(os.getcwd(), 'result', 'train_result', filename, filename+'.csv'), 'wt', newline='') as f1:
			f1_writer = csv.writer(f1)
			f1_writer.writerow(['Cross validation', 'Epoch', 'Train loss', 'Val loss', 'Accuracy', 'Recall', 'Precision', 'F1-score', 'IoU'])

			# Initialize criterion
			if criterion == 'bce':
				criterion = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor(pos_weight))
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

			criterion.to(device)
			activation.to(device)

			cv_results = {1:{}, 2:{}, 3:{}, 4:{}}
			cv_val_loss = []
			cv_best_epoch = []

			for cv in tqdm(range(1, 5), desc='Cross-validation', leave=False):
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
				elif model_name == 'unet++':
					model = smp.UnetPlusPlus(encoder_name='resnet101', encoder_weights='imagenet', in_channels=3, classes=1)
					evaluate, train_model, testset_evaluation = evaluate_unet, train_unet, testset_evaluation_unet

				if weights is not None:
					model.load_state_dict(torch.load(weights))

				# Initialize optimizer
				if optim_name == 'adam':
					optimizer = torch.optim.Adam(model.parameters(), lr=lr)
				elif optim_name == 'sgd':
					optimizer = torch.optim.SGD(model.parameters(), momentum=momentum, lr=lr)

				model.to(device)

				train_sets = [Microplastic_data(path=os.path.join(DATASET_DIR, dataset), transform=TTA4train) for dataset in datasets_dir if str(cv + 1) not in dataset and '1' not in dataset]
				train_loaders = [DataLoader(train_set, batch_size=batch_size, shuffle=True) for train_set in train_sets]
				val_set = Microplastic_data(path=os.path.join(DATASET_DIR, 'dataset_' + str(cv + 1)), transform=TTA4train)
				val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

				tqdm.write('#'*133)
				tqdm.write("Time: {}".format(time))
				tqdm.write("Start training cross-validation [{}]...\n".format(cv))
				val_loss, val_performances = evaluate(	model=model, device=device, total_epoch=epoch, epoch=0, val_loader=val_loader, \
														activation=activation, criterion=criterion, metrics=evaluation_metrics \
														)
				f1_writer.writerow([cv, 0, 'NA', val_loss] +  val_performances)
				del val_loss; del val_performances

				best_epoch, evaluations = train_model(	model=model, device=device, total_epoch=epoch,\
														train_loaders=train_loaders, val_loader=val_loader, \
														criterion=criterion, optimizer=optimizer, metrics=evaluation_metrics, \
														activation=activation, cv_n=cv,	writer=f1_writer, filename=filename, \
														save2=os.path.join(os.getcwd(), 'result', 'model_saved', filename)\
														)
				tqdm.write("\nFinished training cross-validation [{}]...".format(cv))
				tqdm.write("Time: {}".format(datetime.now()))
				tqdm.write("For cross-validation [{}], best model saved at epoch [{}]:\nValidation loss [{:.4f}]\nAccuracy [{:.4f}]\nRecall [{:.4f}]\nPrecision [{:.4f}]\
							\nF1-score [{:.4f}]\nIoU [{:.4f}]\n".format(cv, best_epoch, evaluations[0], evaluations[1], evaluations[2], \
																		evaluations[3], evaluations[4], evaluations[5] \
																		) \
							)
				cv_results[cv] = [evaluations[0], evaluations[1], evaluations[2], evaluations[3], evaluations[4], evaluations[5]]
				cv_val_loss.append(evaluations[0])
				cv_best_epoch.append(best_epoch)
				del best_epoch; del evaluations

			tqdm.write('#'*133)
			tqdm.write("Training result for [{}]:".format(filename))
			best_cv_idx = cv_val_loss.index(min(cv_val_loss))
			best_cv_result = cv_results[best_cv_idx + 1]
			shutil.copy(os.path.join(os.getcwd(), 'result', 'model_saved', filename, filename+'_CV[{}].pth'.format(best_cv_idx + 1)), os.path.join(os.getcwd(), 'result', 'model_saved', filename, filename+'_best_CV.pth'))
			tqdm.write("The lowest validation loss obtained during cross-validation [{}] at epoch [{}]:\nValidation loss: {}\nAccuracy: {}\nRecall: {}\nPrecision: {}\nF1-score: {}\nIoU: {}\n".format(best_cv_idx + 1, cv_best_epoch[best_cv_idx], best_cv_result[0], best_cv_result[1], \
						best_cv_result[2], best_cv_result[3], best_cv_result[4], best_cv_result[5])
					)
		
		if test:	# Evaluate the trained models
			print("Evaluating...")

			with open(os.path.join(os.getcwd(), 'result', 'evaluation', filename, filename+'.csv'), 'wt', newline='') as f2:
				f2_writer =csv.writer(f2)
				f2_writer.writerow(['Cross validation', 'Image number', 'Accuracy', 'Recall', 'Precision', 'F1-score', 'IoU', '', 'TP', 'FP', 'FN', 'TN'])
				for cv in tqdm(range(1, 5), desc='Cross-validation', leave=False):
					_, _ = testset_evaluation(	model=model, device=device, testset_path=os.path.join(os.getcwd(), 'dataset', 'dataset_1'), \
																	weight=os.path.join(os.getcwd(), 'result', 'model_saved', filename, filename+'_CV[{}].pth'.format(cv)), \
																	metrics=evaluation_metrics, save2=os.path.join(os.getcwd(), 'result', 'pred_mask', filename, 'CV_{}'.format(cv)), \
																	write2=f2_writer, TTA=TTA, cv_n=cv\
																)
					if cv == best_cv_idx + 1:
						with open(os.path.join(os.getcwd(), 'result', 'evaluation', filename, filename+'_best_CV.csv'), 'wt', newline='') as f3:
							f3_writer =csv.writer(f3)
							f3_writer.writerow(['Image number', 'Accuracy', 'Recall', 'Precision', 'F1-score', 'IoU', '', 'TP', 'FP', 'FN', 'TN'])
							performances, confusion = testset_evaluation(	model=model, device=device, testset_path=os.path.join(os.getcwd(), 'dataset', 'dataset_1'), \
																	weight=os.path.join(os.getcwd(), 'result', 'model_saved', filename, filename+'_best_CV.pth'.format(cv)), \
																	metrics=evaluation_metrics, save2=os.path.join(os.getcwd(), 'result', 'pred_mask', filename, 'best_CV'.format(cv)), \
																	write2=f3_writer, TTA=TTA \
																)
			
			tqdm.write('#'*133)
			print("Finished evaluation...")
			print("Evaluation result for [{}]:".format(filename))
			print("Performance from test set for the best cross-validation [{}] at epoch [{}]:".format(best_cv_idx + 1, cv_best_epoch[best_cv_idx]))
			print("Accuracy [{:.4f}]\nRecall [{:.4f}]\nPrecision [{:.4f}]\nF1-score [{:.4f}]\nIoU [{:.4f}]\
					\nTP [{}] | FP [{}] | FN [{}] | TN [{}]\n".format(performances[0], performances[1], performances[2], performances[3], performances[4], \
																	confusion[0], confusion[1], confusion[2], confusion[3]) \
					)

	else:	# If training is not being done. Load saved model.
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
		elif model_name == 'unet++':
			model = smp.UnetPlusPlus(encoder_name='resnet101', encoder_weights='imagenet', in_channels=3, classes=1)
			evaluate, train_model, testset_evaluation = evaluate_unet, train_unet, testset_evaluation_unet

		if test:
			print("Evaluating...")
			with open(os.path.join(os.getcwd(), 'result', 'evaluation', filename, filename+'.csv'), 'wt', newline='') as f2:
				f2_writer =csv.writer(f2)
				f2_writer.writerow(['Image number', 'Accuracy', 'Recall', 'Precision', 'F1-score', 'IoU', '', 'TP', 'FP', 'FN', 'TN'])

				performances, confusion = testset_evaluation(	model=model, device=device, testset_path=os.path.join(os.getcwd(), 'dataset', 'dataset_1'), \
																weight=weights, metrics=evaluation_metrics, \
																save2=os.path.join(os.getcwd(), 'result', 'pred_mask', filename), \
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
			batch_size=args.batch_size, criterion=args.criterion, pos_weight=args.pos_weight, optimizer=args.optimizer, momentum=args.momentum, \
			lr=args.lr, TTA=args.TTA, cuda=args.cuda
		)
