[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# MinBackProp – Backpropagating through Minimal Solvers

Source code for the paper [MinBackProp – Backpropagating through Minimal Solvers](https://arxiv.org/abs/2404.17993)

<br></br>
<p align="center">
  <img src="scheme-github.svg"/>
</p> 

## Outlier Detection for Essential Matrix Estimation
We evaluate our MinBackProp on the outlier detection for essential matrix estimation. This code is based on the [baseline](https://github.com/weitong8591/differentiable_ransac/tree/fc40fe0a5a7eeb0e2ec6b185d6218c2005a98cf5) we compare with; the forward pass is the same for both the baseline and MinBackProp and the backward pass differs ($\color{rgb(192,0,0)}{\text{Autograd}}$ vs $\color{rgb(0,112,192)}{\text{DDN}}$ / $\color{rgb(0,176,80)}{\text{IFT}}$).

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
git clone https://github.com/weitong8591/magsac.git --recursive
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
git clone --recurse-submodules -j8 https://github.com/disungatullina/MinBackProp.git
cd MinBackProp
```

### Training
Use ```-ift 1``` for the IFT, ```-ift 2``` for the DDN, and ```-ift 0``` for Autograd (baseline). Default is ```-ift 1```.
```bash
python train.py -ift 1 -nf 2000 -m pretrained_models/weights_init_net_3_sampler_0_epoch_1000_E_rs_r0.80_t0.00_w1_1.00_.net -bs 32 -e 10 -tr 1 -t 0.75 -pth <data_path>
```

### Evaluation
Models for the inference stored in the ```models``` directory.
```bash
python test_magsac.py -nf 2000 -m models/ift.net -bs 32 -bm 1 -t 2 -pth <data_path>
```

### Dataset
Download the RootSIFT features of the PhotoTourism dataset from [here](https://cmp.felk.cvut.cz/~weitong/nabla_ransac/diff_ransac_data.zip).


<!-- ## Toy examples

### 3D Point Registration with an Outlier
blah

### Fundamental Matrix Estimation with an Outlier
blah

## Citation
Cite us as -->

## TODO
- [ ] Add toy examples
- [ ] Docker image for outlier detection for essential matrix estimation
