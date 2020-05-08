# Noise Contrastive Estimation (NCE)

### The Method

This is an implementation of  [Noise Contrastive Estimation (NCE)]( http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf ) on 2D dataset. 

NCE is a method to estimate energy based models (EBM)

$$p(x) = \frac{\exp[-f_\theta(x)]}{Z(\theta)}$$

where

$$Z(\theta) = \int\exp[-f_\theta(x)]dx$$

is the normalizing constant that is hard to compute. In NCE,  the normalizing constant is specified as a trainable parameter $c=\log Z$. We cannot directly do MLE training on the dataset because $p$ can simply blow up to infinity by letting $Z\to0$ (or $c\to-\infty$). Instead, in NCE, we train the EBM by doing a (nonlinear) logistic regression/classification between the training data and some noise. 

----

### Training

The training objective is specified in the file [util.py](util.py). To train the model, do

```shell
python trian.py --dataset=8gaussians # avaiable: '8gaussians', '2spirals', 'checkerboard', 'rings', 'pinwheel'
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