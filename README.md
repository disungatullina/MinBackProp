[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# MinBackProp – Backpropagating through Minimal Solvers

Source code for the paper [MinBackProp – Backpropagating through Minimal Solvers](https://arxiv.org/abs/2404.17993)

<br></br>
<p align="center">
  <img src="scheme-github.svg"/>
</p>

## Outlier Detection for Essential Matrix Estimation
We evaluate our MinBackProp on outlier detection for essential matrix estimation. This code is based on the baseline [∇-RANSAC](https://github.com/weitong8591/differentiable_ransac/tree/fc40fe0a5a7eeb0e2ec6b185d6218c2005a98cf5) we compare with; the forward pass is the same for both the baseline and MinBackProp, while the backward pass differs ($\color{rgb(192,0,0)}{\text{Autograd}}$ vs $\color{rgb(0,112,192)}{\text{DDN}}$ / $\color{rgb(0,176,80)}{\text{IFT}}$). 
<br></br>
<p align="center">
  <img src="table.svg" width=500/>
</p>

### Requirements and installation
Install the required packages
```
python = 3.8.10	
pytorch = 1.12.1
opencv = 3.4.2
tqdm
kornia
kornia_moons
tensorboardX
scikit-learn
einops
yacs
```
For inference, build [MAGSAC++](https://github.com/danini/magsac.git) with
``` bash
git clone https://github.com/disungatullina/magsac.git --recursive
cd magsac
mkdir build
cd build
cmake ..
make
cd ..
python setup.py install
```
Then clone the project with submodules
``` bash
git clone https://github.com/disungatullina/MinBackProp.git --recurse-submodules -j8
cd MinBackProp
```

### Training
Use ```-ift 1``` for the IFT, ```-ift 2``` for the DDN, and ```-ift 0``` for Autograd (baseline). Default is ```-ift 1```.
```bash
python train.py -ift 1 -nf 2000 -m pretrained_models/weights_init_net_3_sampler_0_epoch_1000_E_rs_r0.80_t0.00_w1_1.00_.net -bs 32 -e 10 -tr 1 -t 0.75 -pth <data_path>
```

### Evaluation
Models for inference are stored in the ```models``` directory.
```bash
python test_magsac.py -nf 2000 -m models/ift.net -bs 32 -bm 1 -t 2 -pth <data_path>
```

### Dataset
Download the RootSIFT features of the PhotoTourism dataset from [here](https://cmp.felk.cvut.cz/~weitong/nabla_ransac/diff_ransac_data.zip).

### Command line arguments
```
-ift: backprop method to use, 0-autograd, 1-ift, 2-ddn, default=1
-pth: path to the dataset
-nf: number of features, default=2000
-m: pretrained model to init or trained model for inference
-bs: batch size, default=32
-e: the number of epochs, default=10
-tr: train or test mode, default=0
-t: threshold, default=0.75
-lr: learning rate, default=1e-4
-bm: batch mode, using all the 12 testing scenes defined in utils.py, default=0
-ds: name of a scene, if single scene used, default="st_peters_square"
```
See more command line arguments in ```utils.py```.

## Toy examples

### 3D Point Registration with an Outlier
```bash
cd toy_examples
python estimate_rotation.py --ift --ddn --autograd --plot
```

### Fundamental Matrix Estimation with an Outlier
```bash
cd toy_examples
python estimate_fundamental.py --ift --ddn --autograd --plot
```

## TODO
- [ ] Docker image for outlier detection for essential matrix estimation

## Citation
If you use our algorithm, please cite
```
@misc{sungatullina2024minbackprop,
      title={MinBackProp -- Backpropagating through Minimal Solvers}, 
      author={Diana Sungatullina and Tomas Pajdla},
      year={2024},
      eprint={2404.17993},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
