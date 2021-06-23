# MP-Net

## Getting started
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


## Data preparation
*dataset* directory contains a total of 73 fluorescent patches (256x256) and their corresponding masks. They are divided into train, validation, and test set where each set consists of 53, 10, 10 patches, respectively.

Just replace the dataset directory according to its structure if you want to use your own patches for training and testing. 


## Training and testing

Both training and testing can be performed using main.py and a simple run sample is shown below. It will train U-Net with ResNet50 encoder and perform evaluation as well.

```shell
python main.py --model unet --train --test --epoch 20 --batch_size 10 --criterion dice --optimizer sgd --momentum 0.9 --lr 0.001
```

The result from training will be saved under ./result which is structured as below.

```tree
result
├── train_result    # Stores evaluation result during training for every epochs.
│   └── unet
│       └── file.csv
├── stdout          # Stores information printed on the terminal.
│   └── unet
│       └── file.txt
├── pred_mask       # Stores the prediction masks for test set during testing.
│   └── unet
│       └── file
│           └── prediction_mask.png
├── model_saved     # Stores the best model during training.
│   └── unet
│       └── file.pth
└── evaluation      # Stores the performance of the best model saved.
    └── unet
        └── file.csv

```

The name of the files saved during training will be in the format: 

```
"YEAR-MONTH-DAY-HOUR-MINUTE-SECOND_[MODEL NAME]_[LOSS FUNCTION]_[OPTIMIZER]_[EPOCHS_LEARNING RATE]_TTA[AUGMENTATION USED]"
```

Implementation of TTA can be performed by stating --TTA like below.

```shell
python main.py --model unet --train --test --epoch 20 --batch_size 10 --criterion dice --optimizer sgd --momentum 0.9 --lr 0.001 --TTA B C HSV
```

You can also perform only evaluation by removing --train argument. However, you must provide an absolute path of pre-trained weights. 

```shell
python main.py --model unet --test --weights ./unet4_best_model.pth
```

The name of the files saved by conducting only evalution will look like below: 

```
"YEAR-MONTH-DAY-HOUR-MINUTE-SECOND_[MODEL NAME]_pretrained[NAME OF PRETRAINED WEIGHTS]_TTA[AUGMENTATION USED]"
```


## Pre-trained weights can be downloaded from below links:

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
