# Noise Contrastive Estimation (NCE)

### Introduction

This is an implementation of  [Noise Contrastive Estimation (NCE)]( http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf ) in PyTorch on 2D dataset. 

NCE is a method to estimate energy based models (EBM)

<img src="https://latex.codecogs.com/svg.image?p_\theta(x)&space;=&space;\frac{\exp[-f_\theta(x)]}{Z(\theta)}" title="p_\theta(x) = \frac{\exp[-f_\theta(x)]}{Z(\theta)}" />

where

<img src="https://latex.codecogs.com/svg.image?Z(\theta)&space;=&space;\int\exp[-f_\theta(x)]dx" title="Z(\theta) = \int\exp[-f_\theta(x)]dx" />

is the normalizing constant that is hard to compute. In NCE,  the normalizing constant is treated as a trainable parameter <img src="https://latex.codecogs.com/svg.image?c=\log&space;Z" title="c=\log Z" />. We cannot directly do maximum likelihood estimation (MLE) with <img src="https://latex.codecogs.com/svg.image?\inline&space;\max_\theta&space;p_\theta(x)" title="\inline \max_\theta p_\theta(x)" /> because <img src="https://latex.codecogs.com/svg.image?\inline&space;p_\theta(x)" title="\inline p_\theta(x)" /> can simply blow up to infinity by letting  <img src="https://latex.codecogs.com/svg.image?Z\to0" title="Z\to0" /> (or <img src="https://latex.codecogs.com/svg.image?c\to&space;-\infty" title="c\to -\infty" />). Instead, in Noise Contrastive Estimation, we train the energy based model by doing (nonlinear) logistic regression/classification between the data distribution <img src="https://latex.codecogs.com/svg.image?p_{\mathrm{data}}" title="p_{\mathrm{data}}" /> and some noise distribution <img src="https://latex.codecogs.com/svg.image?q" title="q" />. 

There are three requirements for the noise distribution <img src="https://latex.codecogs.com/svg.image?q" title="q" />:

1. log density can be evaluated on any input;
2. samples can be obtained from the distribution;
3. <img src="https://latex.codecogs.com/svg.image?q(x)\neq0" title="q(x)\neq0" /> for all <img src="https://latex.codecogs.com/svg.image?x" title="x" /> such that <img src="https://latex.codecogs.com/svg.image?p_{\mathrm{data}}(x)\neq0" title="p_{\mathrm{data}}(x)\neq0" />.

Here we use Multivariate Gaussian as the noise distribution. 

The objective is to _maximize_ the posterior log-likelihood of the classification

<img src="https://latex.codecogs.com/svg.image?V(\theta)&space;=&space;\mathbb{E}_{x\sim&space;p_{\text{data}}}\log\frac{p_\theta(x)}{p_\theta(x)&plus;q(x)}&space;&plus;&space;\mathbb{E}_{\tilde{x}\sim&space;q}\log\frac{q(x)}{p_\theta(\tilde{x})&space;&plus;&space;q(\tilde{x})}." title="V(\theta) = \mathbb{E}_{x\sim p_{\text{data}}}\log\frac{p_\theta(x)}{p_\theta(x)+q(x)} + \mathbb{E}_{\tilde{x}\sim q}\log\frac{q(x)}{p_\theta(\tilde{x}) + q(\tilde{x})}." />

This objective is implemented in the file [util.py](util.py) as the `value` function. In other word, we minimize <img src="https://latex.codecogs.com/svg.image?\inline&space;-V(\theta)" title="\inline -V(\theta)" />, and we use Adam as the optimizer.

### Training

To train the model, do

```shell
python trian.py --dataset=8gaussians 
```
Available datasets:
- `8gaussians` (default)
- `2spirals`
-  `checkerboard`
-  `rings`
-   `pinwheel`

A density plot is saved in the folder `images` after every epoch. After training, you can obtain gif images like below by excecuting the python script in the folder:

```shell
cd images
python create_gif.py
```


### Examples

Some visualizations of the learned energy densities are listed below.

- `8gaussians` dataset

![8gaussians](images/8gaussians.gif)

- `pinwheel` dataset

![pinwheel](images/pinwheel.gif)

- `2spirals` dataset

![2spirals](images/2spirals.gif)