import os
import argparse
import torch
import albumentations as A
import src.segmentation_models.segmentation_models_pytorch as smp
from torch.nn import Conv2d
from torchvision.models.segmentation import fcn_resnet101, deeplabv3_resnet101
from src.segmentation_models.segmentation_models_pytorch.utils.metrics import Accuracy, IoU, Recall, Fscore, Precision
from src.functional.fit_unet import evaluate as evaluate_unet
from src.functional.fit_unet import train_model as train_unet
from src.functional.fit_torchvision_model import evaluate as evaluate_torchvision
from src.functional.fit_torchvision_model import train_model as train_torchvision
from src.functional.evaluation_unet import testset_evaluation as testset_evaluation_unet
from src.functional.evaluation_torchvision_model import testset_evaluation as testset_evaluation_torchvision




def parse_args():
    parser = argparse.ArgumentParser(description='Microplastics Segmentation Reproducing Results')

    parser.add_argument('--reproduce', type=str, required=True, choices=['testset', 'spiked'], 
                        help='Specify what you want to reproduce.')
    parser.add_argument('--data', type=str, required=True,
                        help='Specify the directory path where the data downloaded from kaggle is located.')
    parser.add_argument('--model', type=str, required=True, choices=['unet', 'fcn', 'deeplabv3', 'unet++'], 
                             help='Specify the model (U-Net, FCN, Deeplabv3, or U-Net++).')
    parser.add_argument('--weights', type=str, default=None,
                        help='Provide absolute path of pre-trained weights (default=None).')
    parser.add_argument('--TTA', dest='TTA', action='store_true', help='Specify if TTA was integrated during training.')
    parser.set_defaults(TTA=False)
    parser.add_argument('--cuda', type=int, default=0,
                    help='Specify the cuda for GPU usage (default=0).')
    parser.add_argument('--out', type=str, default=None,
                help='Specify the output directory where masks will be saved (default=None).')

    args = parser.parse_args()
    return args


def get_model(model_name):
    if model_name == 'unet':		# Initialize U-Net model
        model = smp.Unet(encoder_name="resnet101", in_channels=3, classes=1, encoder_weights="imagenet")
        testset_evaluation = testset_evaluation_unet
    elif model_name == 'fcn':	# Initialize FCN model
        model = fcn_resnet101(pretrained=True)
        # The last convolutional layer is modified to produce a binary output.
        model.classifier._modules['4'] = Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
        model.aux_classifier._modules['4'] = Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
        testset_evaluation = testset_evaluation_torchvision
    elif model_name == 'deeplabv3':	# Initialize Deeplabv3 model
        model = deeplabv3_resnet101(pretrained=True)
        # The last convolutional layer is modified to produce a binary output.
        model.classifier._modules['4'] = Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
        model.aux_classifier._modules['4'] = Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
        testset_evaluation = testset_evaluation_torchvision
    elif model_name == 'unet++':
        model = smp.UnetPlusPlus(encoder_name='resnet101', encoder_weights='imagenet', in_channels=3, classes=1)
        testset_evaluation = testset_evaluation_unet
    return model, testset_evaluation


def get_data_path(dataPath):
    fls, masks = os.listdir(dataPath)
    return [os.path.join(dataPath, fls, fl) for fl in fls], [os.path.join(dataPath, masks, mask) for mask in masks]



def main(reproduce, dataPath, model_name, weight_path, cuda, save2, TTA):
    assert weight_path is not None, 'Must specify path to the pre-trained weight.'

    device = torch.device('cuda:{}'.format(cuda) if torch.cuda.is_available() else 'cpu')
    evaluation_metrics = [	Accuracy(balanced=True, threshold=None).to(device), Recall(threshold=None).to(device), 
                        Precision(threshold=None).to(device), Fscore(threshold=None).to(device), IoU(threshold=None).to(device)]
    if TTA:
        TTA = ['B', 'C', 'HSV']
        TTA_is = TTA.copy()
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
    
    model, testset_evaluation = get_model(model_name)

    if reproduce == 'testset':
        print("Evaluating testset...")
        performances, _ = testset_evaluation(	model=model, device=device, testset_path=dataPath, \
                                                        weight=weight_path, metrics=evaluation_metrics, \
                                                        save2=save2, \
                                                        write2=None, TTA=TTA \
                                                        )
        
        print("Finished evaluation...\n")
        print("Accuracy [{:.4f}]\nRecall [{:.4f}]\nPrecision [{:.4f}]\nF1-score [{:.4f}]\nIoU [{:.4f}]".format(performances[0], 
                                                            performances[1], performances[2], performances[3], performances[4]))
    else:   # spiked
        print("Producing predicted mask for spiked images...")
        _, _ = testset_evaluation(	model=model, device=device, testset_path=dataPath, \
                                                        weight=weight_path, metrics=evaluation_metrics, \
                                                        save2=save2, \
                                                        write2=None, TTA=TTA \
                                                        )
        
        print("Finished producing predicted mask for spiked images...\n")


if __name__ == '__main__':
    args = parse_args()
    main(reproduce=args.reproduce, 
        dataPath=args.data, 
        model_name=args.model, 
        weight_path=args.weights, 
        cuda=args.cuda,
        TTA=args.TTA,
        save2=args.out
        )