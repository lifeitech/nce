# Noise Contrastive Estimation (NCE)

### The Method

This is an implementation of  [Noise Contrastive Estimation (NCE)]( http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf ) in PyTorch on 2D dataset. 

NCE is a method to estimate energy based models (EBM)

<a href="https://www.codecogs.com/eqnedit.php?latex=p(x)&space;=&space;\frac{\exp[-f_\theta(x)]}{Z(\theta)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p_\theta(x)&space;=&space;\frac{\exp[-f_\theta(x)]}{Z(\theta)}" title="p(x) = \frac{\exp[-f_\theta(x)]}{Z(\theta)}" /></a>

where

<a href="https://www.codecogs.com/eqnedit.php?latex=Z(\theta)&space;=&space;\int\exp[-f_\theta(x)]dx" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Z(\theta)&space;=&space;\int\exp[-f_\theta(x)]dx" title="Z(\theta) = \int\exp[-f_\theta(x)]dx" /></a>

is the normalizing constant that is hard to compute. In NCE,  the normalizing constant is specified as a trainable parameter <a href="https://www.codecogs.com/eqnedit.php?latex=c=\log&space;Z" target="_blank"><img src="https://latex.codecogs.com/gif.latex?c=\log&space;Z" title="c=\log Z" /></a>. We cannot directly do MLE training on the dataset because <a href="https://www.codecogs.com/eqnedit.php?latex=p" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p" title="p" /></a> can simply blow up to infinity by letting  <a href="https://www.codecogs.com/eqnedit.php?latex=Z\to0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Z\to0" title="Z\to0" /></a>(or <a href="https://www.codecogs.com/eqnedit.php?latex=c\to-\infty" target="_blank"><img src="https://latex.codecogs.com/gif.latex?c\to-\infty" title="c\to-\infty" /></a>). Instead, in NCE, we train the EBM by doing a (nonlinear) logistic regression/classification between the training data and some noise. We generally have three requirements for the noise distribution <a href="https://www.codecogs.com/eqnedit.php?latex=q" target="_blank"><img src="https://latex.codecogs.com/gif.latex?q" title="q" /></a>:

1. log density can be evaluated on any input;
2. samples can be obtained from the distribution;
3. <a href="https://www.codecogs.com/eqnedit.php?latex=q(x)\neq0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?q(x)\neq0" title="q(x)\neq0" /></a> for all <a href="https://www.codecogs.com/eqnedit.php?latex=x" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x" title="x" /></a> such that <a href="https://www.codecogs.com/eqnedit.php?latex=p_{\text{data}}(x)\neq&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p_{\text{data}}(x)\neq&space;0" title="p_{\text{data}}(x)\neq 0" /></a>.

----

### Training

The training objective is specified in the file [util.py](util.py). We _maximize_ the posterior log-likelihood of the classification

<a href="https://www.codecogs.com/eqnedit.php?latex=J(\theta)&space;=&space;\mathbb{E}_{x\sim&space;p_{\text{data}}}\log\frac{p_\theta(x)}{p_\theta(x)&plus;q(x)}&space;&plus;&space;\mathbb{E}_{\tilde{x}\sim&space;q}\log\frac{q(x)}{p_\theta(\tilde{x})&space;&plus;&space;q(\tilde{x})}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?J(\theta)&space;=&space;\mathbb{E}_{x\sim&space;p_{\text{data}}}\log\frac{p_\theta(x)}{p_\theta(x)&plus;q(x)}&space;&plus;&space;\mathbb{E}_{\tilde{x}\sim&space;q}\log\frac{q(x)}{p_\theta(\tilde{x})&space;&plus;&space;q(\tilde{x})}" title="J(\theta) = \mathbb{E}_{x\sim p_{\text{data}}}\log\frac{p_\theta(x)}{p_\theta(x)+q(x)} + \mathbb{E}_{\tilde{x}\sim q}\log\frac{q(x)}{p_\theta(\tilde{x}) + q(\tilde{x})}" /></a>

To train the model, do

```shell
python trian.py --dataset=8gaussians  # avaiable: '8gaussians', '2spirals', 'checkerboard', 'rings', 'pinwheel'
```

----

### Examples

Sample training processes:

- `8gaussians` dataset

![8gaussians](images/8gaussians.gif)

- `pinwheel` dataset

![pinwheel](images/pinwheel.gif)

- `2spirals` dataset

![2spirals](images/2spirals.gif)