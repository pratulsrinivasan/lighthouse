# Lighthouse: Predicting Lighting Volumes for Spatially-Coherent Illumination
## Pratul P. Srinivasan, Ben Mildenhall, Matthew Tancik, Jonathan T. Barron, Richard Tucker, Noah Snavely, CVPR 2020

This release contains code for predicting incident illumination at any 3D
location within a scene. The algorithm takes a narrow-baseline stereo pair of
RGB images as input, and predicts a multiscale RGBA lighting volume.
Spatially-varying lighting within the volume can then be computed by standard
volume rendering.

## Running a pretrained model

``interiornet_test.py`` contains an example script for running a pretrained
model on the test set (formatted as .npz files). Please download and extract the
[pretrained model](https://drive.google.com/drive/folders/1VQjRpInmfspz0Rw0Dlm9RbdHX5ziFeDI?usp=sharing)
and
[testing examples](https://drive.google.com/a/berkeley.edu/file/d/121DHkPpbQlyedruI4huF36BF7T1LHcKX/view?usp=sharing)
files, and then include the corresponding file/directory names as command line
flags when running ``interiornet_test.py``.

Example usage (edit paths to match your directory structure):
``python -m lighthouse.interiornet_test --checkpoint_dir="lighthouse/model/" --data_dir="lighthouse/testset/" --output_dir="lighthouse/output/"``

## Training

Please refer to the ``train.py`` for code to use for training your own model.

This model was trained using the [InteriorNet](https://interiornet.org/)
dataset. It may be helpful to read ``data_loader.py`` to get an idea of how we
organized the InteriorNet dataset for training.

To train with the perceptual loss based on VGG features (as done in the paper),
please download the ``imagenet-vgg-verydeep-19.mat``
[pretrained VGG model](http://www.vlfeat.org/matconvnet/pretrained/#downloading-the-pre-trained-models),
and include the corresponding path as a command line flag when running
``train.py``.

Example usage (edit paths to match your directory structure):
``python -m lighthouse.train --vgg_model_file="lighthouse/model/imagenet-vgg-verydeep-19.mat" --load_dir="" --data_dir="lighthouse/data/InteriorNet/" --experiment_dir=lighthouse/training/``

## Extra

This model is quite memory-hungry, and we used a NVIDIA Tesla V100 GPU for
training and testing with a single example per minibatch. You may run into
memory constraints when training on a GPU with less than 16 GB memory or testing
on a GPU with less than 12 GB memory. If you wish to train a model on a GPU with
<16 GB memory, you may want to try removing the finest volume in the multiscale
representation (see the model parameters in ``train.py``).

If you find this code helpful, please cite our paper:
``
@article{Srinivasan2020,
  author    = {Pratul P. Srinivasan, Ben Mildenhall, Matthew Tancik, Jonathan T. Barron, Richard Tucker, Noah Snavely},
  title     = {Lighthouse: Predicting Lighting Volumes for Spatially-Coherent Illumination},
  journal   = {CVPR},
  year      = {2020},
}
``
