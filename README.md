# MP-Net

## Getting Started
Set up an environment.

Clone the repository.
```shell
git clone https://github.com/sanghyeonp/MP-Net.git
```

Go into the directory.
```shell
cd MP-Net
```

Install required dependencies using the requirements.txt.
```shell
pip install -r requirements.txt
```

Note: The code is written by utilizing and modifying other repositories. Detailed explanation and modification is listed below.

* U-Net architecture and evaluation metrics: Utilized https://github.com/qubvel/segmentation_models.pytorch (Note that a modification was done to accuracy to compute balanced accuracy.)
* Loss functions : https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch


## Data Preparation
*dataset* directory contains a total of 73 fluorescent patches (256x256) and their corresponding masks. They are randomly divided into 5 different datasets where dataset 1 consists of 13 patches and other datasets consist of 15 patches each. The segregation of patches is to perform 4-fold cross-validation where dataset 1 serves as test set and other datasets serve as either train or validaiton set.

*Note: If you want ot use your own dataset to perform 4-fold cross-validation, then modify the DATASET_DIR variable in config.py with the path to your dataset. Be careful to construct your dataset directory structure as below.*

```tree
dataset
├── dataset_1           # Test set
│   ├── 001.jpg         # Fluorescent patch
│   ├── ...
│   └── labels          # Contains mask patches
│       ├── 001.jpg     # Mask patch corresponding to the fluorescent patch
│       └── ...
├── dataset_2           # Either train or validation set
│   ├── 082.jpg         # Fluorescent patch
│   ├── ...
│   └── labels          # Contains mask patches
│       ├── 082.jpg     # Mask patch corresponding to the fluorescent patch
│       └── ...
├── dataset_3           # Either train or validation set
│   ├── 022.jpg         # Fluorescent patch
│   ├── ...
│   └── labels          # Contains mask patches
│       ├── 022.jpg     # Mask patch corresponding to the fluorescent patch
│       └── ...
├── dataset_4           # Either train or validation set
│   ├── 125.jpg         # Fluorescent patch
│   ├── ...
│   └── labels          # Contains mask patches
│       ├── 125.jpg     # Mask patch corresponding to the fluorescent patch
│       └── ...
└── dataset_5           # Either train or validation set
    ├── 50.jpg         # Fluorescent patch
    ├── ...
    └── labels          # Contains mask patches
        ├── 50.jpg     # Mask patch corresponding to the fluorescent patch
        └── ...

```

## Training and Testing

*Note: The training implements 4-fold cross-validation as default.*

The name of the files saved during training will be in the format as below: 

```
# FILE NAME
"YEAR-MONTH-DAY-HOUR-MINUTE-SECOND_[MODEL NAME]_[EPOCHS]_[LOSS FUNCTION, POSITIVE WEIGHT]_[OPTIMIZER_LEARNING RATE_MOMENTUM]_TTA[AUGMENTATION USED]"
```

Both training and testing can be performed using main.py, and a simple run sample is shown below:

*Train U-Net with ResNet-101 encoder and evaluate the performance of the best model saved for each cross-validation fold.*


```shell 
python main.py --model unet --train --test --epoch 20 --batch_size 10 --criterion bce --pos_weight 9 --optimizer sgd --momentum 0.9 --lr 0.001
```

The result from training will be saved under **./result**, if *--out* is not given.\
The directory where the results will be structured as below.

```tree
result
├── train_result    # Stores evaluation result during training for every epochs.
│   └── FILE NAME
│       └── FILE NAME.csv
├── stdout          # Stores information printed on the terminal.
│   └── FILE NAME
│       └── FILE NAME.txt
├── pred_mask       # Stores the prediction masks for fluorescent patches in the test set during 
│   │                 evaluation.
│   └── FILE NAME
│       ├── CV_1
│       │   ├── 005.png
│       │   └── 018.png
│       ├── CV_2
│       │   ├── 005.png
│       │   └── 018.png
│       ├── CV_3
│       │   ├── 005.png
│       │   └── 018.png
│       ├── CV_4
│       │   ├── 005.png
│       │   └── 018.png
│       └── best_CV
│           ├── 005.png
│           └── 018.png
├── model_saved     # Stores the best model at each cross-validation during training.
│   └── FILE NAME
│       ├── FILE NAME_CV[1].pth
│       ├── FILE NAME_CV[2].pth
│       ├── FILE NAME_CV[3].pth
│       ├── FILE NAME_CV[4].pth
│       └── FILE NAME_best_CV.pth
└── evaluation      # Stores the performance of the best model saved.
    └── FILE NAME
        ├── FILE NAME.csv
        └── FILE NAME_best_CV.csv

```

The parser arguments are as below:

```python
parser = argparse.ArgumentParser(description='Microplastics Segmentation')

parser.add_argument('--model', type=str, required=True, choices=['unet', 'fcn', 'deeplabv3',    
                    'unet++'], 
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
                    help='Specify the loss function (BCEWithLogits loss => bce, SoftDice loss => dice,
                    DiceBCE loss=>dicebce).')

parser.add_argument('--pos_weight', type=float, default=9, 
                    help='Specify the weight to positives when using BCEWithLogits loss. (default=9)')

parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], default='sgd', 
                    help='Specify the optimizer (SGD => sgd, ADAM => adam).')

parser.add_argument('--momentum', type=float, default=0.9,
                    help='Specify the momentum value for SGD optimizer (default=0.9).')

parser.add_argument('--lr', type=float, default=0.001,
                    help='Specify the learning rate (default=0.001).')

parser.add_argument('--TTA', nargs='+', choices=['B', 'C', 'HSV'], default=None,
                    help='Specify the image augmentation being used for test-time augmentation 
                    (default=None).')
parser.add_argument('--cuda', type=int, default=0,
					help='Specify the cuda for GPU usage (default=0).')

parser.add_argument('--out', type=str, default=None,
                    help='Specify the output directory where results will be saved (default=None).')
```


## Implementing Test-time Augmentation

Implementation of TTA can be performed by stating --TTA like below:

```shell
python main.py --model unet --train --test --epoch 20 --batch_size 10 --criterion bce --pos_weight 9 --optimizer sgd --momentum 0.9 --lr 0.001 --TTA B C HSV
```

## Test Model with Pre-trained Weight without Training

You can also perform only evaluation by removing --train argument. 

**However, you must provide an absolute path of the pre-trained weights.**

```shell
python main.py --model unet --test --weights ./unet4_best_model.pth
```

The name of the files saved by performing only testing will look like below: 

```
"YEAR-MONTH-DAY-HOUR-MINUTE-SECOND_[MODEL NAME]_pretrained[FIRST 20 CHARACTERS OF THE PRETRAINED WEIGHTS NAME]_TTA[AUGMENTATION USED]"
```

**Our pre-trained weights can be downloaded from the links below:**

* U-Net : [unet4_best_model.pth](https://drive.google.com/file/d/1wG1WYUtJ49oS0JYVET-33aYvShEKotjf/view?usp=sharing)
* FCN : [fcn_best_model.pth](https://drive.google.com/file/d/1SFhc1G6H0rXEkOXz7q3GM5HBizfr961T/view?usp=sharing)
* Deeplabv3 : [deeplabv3_best_model.pth](https://drive.google.com/file/d/1fbCICTgLOc57z5ETe4Fc6slEBZT9VbiY/view?usp=sharing)
* Nested-UNet : [nested_unet_best_model.pth](https://drive.google.com/file/d/1rTBOZLbK81agYtYVl0WV5Nf2qo6oGFQS/view?usp=sharing)


## Citation
Readers may use the following information to cite our research and the dataset.

Baek, J. Y., de Guzman, M. K., Park, H. M., Park, S., Shin, B., Velickovic, T. C., ... & De Neve, W. (2021). Developing a Segmentation Model for Microscopic Images of Microplastics Isolated from Clams. In Pattern Recognition. ICPR International Workshops and Challenges (pp. 86-97). Springer International Publishing.

The original paper can be found at the following URL:

https://www.springerprofessional.de/en/developing-a-segmentation-model-for-microscopic-images-of-microp/18900224


## Acknowledgement
The research and development activities described in this paper were funded by Ghent University Global Campus (GUGC) and by the Special Research Fund (BOF) of Ghent University (grant no. 01N01718).
