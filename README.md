## Introduction

This document describes the final product of my Master's thesis.
The deep learning system takes a pCT input in DICOM format and a corresponding cardiac mask in Nifti format and generates delineation masks for the following seven cardiac substructures: left and right ventricles, left and right atria, right coronary artery, left anterior descending artery, and the circumflex branch of the left coronary artery. For these substructures (ROI), the system generates predictions using an individually trained 2D U-Net per ROI and saves the generated masks in the output folder as Nrrd files, one per ROI. The following code will be implemented at Maastro Clinic (adapted for their windows landscape). 

Due to potential privacy implications of patient data and increased size, only code but no data is uploaded.

## Model

`from model import UNET`

The 2D U-Net model used during this work is implemented in `./ssseg/model.py`.
The `UNET` class accepts the following three parameters.

- **in_channels**: +integer, number of input channels. Because CT scans are black and white the input channels is set to one.
- **out_channels**: +integer, number of output channels. For a binary segmentation the output channels is set to one.
- **features**: [+integer], controls number of features for each double convolution on the down path and the depth of the network. We use [64, 128, 256, 512] as default value for four double convolutional layers with 64, 128, 256 and 512 features respectively.

## ROIS

The following naming convention is used in this code for the seven ROI.

| ROI                                           | key                 |
|-----------------------------------------------|---------------------|
| right ventricle                               | Ventricle_R         |
| left ventricle                                | Ventricle_L         |
| right atrium                                  | Atrium_R            |
| left atrium                                   | Atrium_L            |
| right coronary artery                         | Coronary_Atery_R    |
| left anterior descending artery               | Coronary_LAD        |
| circumflex branch of the left coronary artery | Coronary_Atery_CFLX |


## SSSEG Package
The final code is organized in the `./ssseg` folder as a local package and can be imported (`import ssseg`) as long as the importing file is on the same directory level as the ssseg folder.

The ssseg package (which stands for **s**ub**s**tructure **s**egmentation) is divided into four modules that handle loading and storing intermediate and final results, preparing the data, creating the segmentations, and exporting the final results. Each of the four modules is explained in more detail below:

### Loading
`from ssseg import loading`

The *loading* module is mainly used by the three modules internally to store and load temporary data as well as the final results. It contains the `createDirectories(output_dir)` function which only parameter `output_dir` defines the output directory where subsequent folders for temporary and final results will be stored. 

The subfolders *wip*, *raw* and *nrrd* are created into the output directory specified by the *output_dir* parameter. The *work in progress* (*wip*) folder is used to store intermediate results in temporary files needed in a subsequent step. The 'raw' folder is used to store raw predictions (confidence maps) as Numpy files that allow further processing without having to regenerate all pre- and post-processing steps and especially the (time-consuming) prediction step. The *nrrd* folder contains the final results, a generated mask as Nrrd file for each of the seven ROIs.

Parameter:
- **output_dir**: string, output directory where subsequent folders for temporary and final results will be stored.

### Prepare
`from ssseg import prepare`

The *prepare* module contains all functions required to preprocess the data starting from a patient pCT scan in DICOM format (which is a directory) and a heart mask (which is expected as a Nifti file). The number of random transformations for test time augmentation (TTA) can be given by the `num_random_ttaugvs` parameter. If set to 0, TTA is not used. 

The preprocessing steps of windowing, resampling, z-normalization, cropping, transformation to canonical orientation, and 2D slicing are identical to our initial preparation of the data before training. All preprocessing transformations are performed on the 3D image of a patient using the [TorchIO](https://TorchIO.readthedocs.io/) framework. Axial slices are then extracted from the preprocessed 3D pCT volumes to obtain 2D data (2D slicing).

`prepare.prepare(input_dir, hmask_file, output_dir, num_random_ttaugvs)`

Parameter:
- **input_dir**: string, input directory of the original pCT DICOM files.
- **hmask_file**: string, file path to the according heart mask in nifti format.
- **output_dir**: string, the output directory as defined in the loading step. 
- **num_random_ttaugvs**: +integer, the number of transformations for test time augmentation.


### Predict
`from ssseg import predict`

The *predict* module loads the prepared data and initializes the U-Net model with the trained weights for each ROI, to generate the predictions.
For each ROI, the weights are loaded into the U-Net model and in evaluation mode, all 2D slices are fed into the network to generate the predictions for that ROI. The process is repeated for each ROI as defined in the configuration (see section  `Config` below).

`predict.predict(configs, output_dir, device='cuda')`

Parameter:
- **configs**: dictionary, containing the setup for each ROI. See section `Config` below and the exemplary `config.json`.
- **output_dir**: string, the output directory as defined in the loading step.
- **device**: string, either 'cuda' for gpu acceleration or 'cpu' to run the model on the cpu instead.

### Finalize
`from ssseg import finalize`

The *finalize* module takes the predicted values from the previous step and recombines them to match the original pCT input. To do this, all predicted slices are arranged along the Z axis to form a 3D volume. The volume is then padded to undo the cropping step performed during preprocessing. The volume is then transferred to a [SITK](https://simpleitk.org/) image and resampled to match the original spacing. All SITK metadata (e.g. origin, direction, spacing) are then copied from the original to the SITK volume just created. The final volume is then exported as Nrrd mask.

`finalize.finalize(rois, input_dir, output_dir)`

Parameter:
- **rois**: [string], a list of all ROIs to be processed.
- **input_dir**: string, input directory of the original pCT DICOM files.
- **output_dir**: string, the output directory as defined in the loading step. 

## Config
The *predict* and *finalize* functions as introduced above expect setup information for each ROI in form of a configuration dictionary. The configuration object is structured as follows:

- Each setup (ROI) has its own entry. The key of each ROI is a unique identifier for the ROI.
- The setup contains two mandatory fields:
    - **masking**: boolean, defines weather masking is used or not. 
    - **weight_path**: string, the path to the trained weights in a PyTorch file.
    - **th**: float, the threshold used for simple output
    - **th_tta**: float, the threshold used for the output generated using test time augmentation.

*NOTE*: The **masking** parameter determines whether masking is applied during prediction. However, since all variants use masking, and it is important to apply all preprocessing and test time augmentation transformation steps to both, the subject and the corresponding mask, the preparation step is simplified by having a TorchIO subject with CT and a heart mask and applying all transformations to the subject. Thus, the mask and CT image are always transformed simultaneously. TorchIO automatically applies only those transformations to the mask that deform the content in some way (affine transformations, resampling, cropping but e.g. no windowing or z-normalization).

*NOTE*: The **loss_fn** parameter is added for completeness and identifies the loss function used to train the weights, but has no effect on the later implementation as this was only for training.

An exemplary configuration containing all parameters determined during the experiments in this work can be found in `./config.json`.

## Using TQDM in a Jupyter Notebook
The *prepare*, *predict* and *finalize* module use tqdm for progress tracking. Tqdm has a dedicated module optimized for Jupyter notebooks (`from tqdm.notebook import tqdm`). To use this, in all modules, the tqdm module can simply be replaced, e.g., `prepare.tqdm = tqdm`.


## CLI 
For each of the three main steps (*prepare*, *predict*, *finalize*) a dedicated command line interface (CLI) script written in python is available in the `./cli` folder.

The *run.sh* script is a handy shortcut that calls all three Python scripts. Here, I used two different Conda environments to overcome a limitation of the DSRI where TorchIO could not be installed in the same environment as the torch version compatible with the cuda kit. Therefore, a Conda environment was used for preparation and finalization and an environment for gpu-accelerated training. The example run.sh script therefore uses absolute paths into these two Conda environments.

```shell
zsh cli/run.sh data/patients/pz1 output/pz1 data/nifti/pz1/mask_cuore-in-toto.nii.gz 30 config.json
```

The above example call to the `./cli/run.sh` script passes the DICOM pCT scan folder as the first parameter, the output folder as the second parameter, and the path to the heart mask as the third parameter. The number of test time augmentations is passed as the fourth parameter (here set to t=30) and the file path to the configuration stored in a JSON file is given as the fifth parameter.

The script `./runall.sh` has been generated automatically and executes all patients of e.g. our test set one after the other by calling the `./cli/run.sh` script for each of them.

## Example:  
The following code is an example implementation in Python that combines all four steps as described above.

```python
from ssseg import loading, prepare, predict, finalize
from tqdm.notebook import tqdm
import json

# set tqdm to notebook version for all modules
prepare.tqdm = tqdm
predict.tqdm = tqdm
finalize.tqdm = tqdm

# define parameters
input_ct = 'data/patients/pz1'
input_heart_mask = 'data/nifti/pz1/mask_cuore-in-toto.nii.gz'
output_dir = 'output/pz1'
t = 0
device = 'cuda'

# load config
f = open('config.json', 'r')
config = json.load(f)
f.close()

# create output directory
loading.createDirectories(output_dir)

# prepare
prepare.prepare(input_ct, input_heart_mask, output_dir, num_random_ttaugvs=t)

# predict
predict.predict(config, output_dir, device=device)

# finalize
finalize.finalize(config, input_dir, output_dir)

````
