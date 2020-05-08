import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def value(energy, noise, x, gen):
    logp_x = energy(x)
    logq_x = noise.log_prob(x).unsqueeze(1)
    logp_gen = energy(gen)
    logq_gen = noise.log_prob(gen).unsqueeze(1)

    ll_data = logp_x - torch.logsumexp(torch.cat([logp_x, logq_x], dim=1), dim=1, keepdim=True)
    ll_gen = logq_gen - torch.logsumexp(torch.cat([logp_gen, logq_gen], dim=1), dim=1, keepdim=True)

    v = ll_data.mean() + ll_gen.mean()

    r_x = torch.sigmoid(logp_x - logq_x)
    r_gen = torch.sigmoid(logq_gen - logp_gen)

    acc = ((r_x > 1/2).sum() + (r_gen > 1/2).sum()).cpu().numpy() / (len(x) + len(gen))


    return -v,  acc


#-------------------------------------------
# DATA
#-------------------------------------------
def get_data(args):
    dataset = sample_2d_data(dataset=args.dataset, n_samples=args.samples)
    dataloader  = DataLoader(dataset, batch_size=args.batch, shuffle=True)
    return dataset, dataloader
    

def sample_2d_data(dataset='8gaussians', n_samples=50000):
    
    z = torch.randn(n_samples, 2)

    if dataset == '8gaussians':
        scale = 4
        sq2 = 1/math.sqrt(2)
        centers = [(1,0), (-1,0), (0,1), (0,-1), (sq2,sq2), (-sq2,sq2), (sq2,-sq2), (-sq2,-sq2)]
        centers = torch.tensor([(scale * x, scale * y) for x,y in centers])
        return sq2 * (0.5 * z + centers[torch.randint(len(centers), size=(n_samples,))])

    elif dataset == '2spirals':
        n = torch.sqrt(torch.rand(n_samples // 2)) * 540 * (2 * math.pi) / 360
        d1x = - torch.cos(n) * n + torch.rand(n_samples // 2) * 0.5
        d1y =   torch.sin(n) * n + torch.rand(n_samples // 2) * 0.5
        x = torch.cat([torch.stack([ d1x,  d1y], dim=1),
                       torch.stack([-d1x, -d1y], dim=1)], dim=0) / 3
        return x + 0.1*z

    elif dataset == 'checkerboard':
        x1 = torch.rand(n_samples) * 4 - 2
        x2_ = torch.rand(n_samples) - torch.randint(0, 2, (n_samples,), dtype=torch.float) * 2
        x2 = x2_ + x1.floor() % 2
        return torch.stack([x1, x2], dim=1) * 2

    elif dataset == 'rings':
        n_samples4 = n_samples3 = n_samples2 = n_samples // 4
        n_samples1 = n_samples - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, set endpoint=False in np; here shifted by one
        linspace4 = torch.linspace(0, 2 * math.pi, n_samples4 + 1)[:-1]
        linspace3 = torch.linspace(0, 2 * math.pi, n_samples3 + 1)[:-1]
        linspace2 = torch.linspace(0, 2 * math.pi, n_samples2 + 1)[:-1]
        linspace1 = torch.linspace(0, 2 * math.pi, n_samples1 + 1)[:-1]

        circ4_x = torch.cos(linspace4)
        circ4_y = torch.sin(linspace4)
        circ3_x = torch.cos(linspace4) * 0.75
        circ3_y = torch.sin(linspace3) * 0.75
        circ2_x = torch.cos(linspace2) * 0.5
        circ2_y = torch.sin(linspace2) * 0.5
        circ1_x = torch.cos(linspace1) * 0.25
        circ1_y = torch.sin(linspace1) * 0.25

        x = torch.stack([torch.cat([circ4_x, circ3_x, circ2_x, circ1_x]),
                         torch.cat([circ4_y, circ3_y, circ2_y, circ1_y])], dim=1) * 3.0

        # random sample
        x = x[torch.randint(0, n_samples, size=(n_samples,))]

        # Add noise
        return x + torch.normal(mean=torch.zeros_like(x), std=0.08*torch.ones_like(x))

    elif dataset == "pinwheel":
        rng = np.random.RandomState()
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = n_samples // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = rng.randn(num_classes*num_per_class, 2) \
            * np.array([radial_std, tangential_std])
        features[:, 0] += 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))
        
        data = 2 * rng.permutation(np.einsum("ti,tij->tj", features, rotations))
        return torch.as_tensor(data, dtype=torch.float32)

    else:
        raise RuntimeError('Invalid `dataset` to sample from.')

# --------------------
# Plotting
# --------------------

@torch.no_grad()
def plot(dataset, energy, noise, epoch, device):
    n_pts = 1000
    range_lim = 4

    # construct test points
    test_grid = setup_grid(range_lim, n_pts, device)

    # plot
    fig, axs = plt.subplots(1, 3, figsize=(12,4.3), subplot_kw={'aspect': 'equal'})
    plot_samples(dataset, axs[0], range_lim, n_pts)
    plot_noise(noise, axs[1], test_grid, n_pts)
    plot_energy(energy, axs[2], test_grid, n_pts)

    # format
    for ax in plt.gcf().axes: format_ax(ax, range_lim)
    plt.tight_layout()

    # save
    print('Saving image to images/....')
    plt.savefig('images/epoch_{}.png'.format(epoch))
    plt.close()


def setup_grid(range_lim, n_pts, device):
    x = torch.linspace(-range_lim, range_lim, n_pts)
    xx, yy = torch.meshgrid((x, x))
    zz = torch.stack((xx.flatten(), yy.flatten()), dim=1)
    return xx, yy, zz.to(device)


def plot_samples(dataset, ax, range_lim, n_pts):
    samples = dataset.numpy()
    ax.hist2d(samples[:,0], samples[:,1], range=[[-range_lim, range_lim], [-range_lim, range_lim]], bins=n_pts, cmap=plt.cm.jet)
    ax.set_title('Target samples')


def plot_energy(energy, ax, test_grid, n_pts):
    xx, yy, zz = test_grid
    log_prob = energy(zz)
    prob = log_prob.exp().cpu()
    # plot
    ax.pcolormesh(xx, yy, prob.view(n_pts,n_pts), cmap=plt.cm.jet)
    ax.set_facecolor(plt.cm.jet(0.))
    ax.set_title('Energy density')

def plot_noise(noise, ax, test_grid, n_pts):
    xx, yy, zz = test_grid
    log_prob = noise.log_prob(zz)
    prob = log_prob.exp().cpu()
    # plot
    ax.pcolormesh(xx, yy, prob.view(n_pts,n_pts), cmap=plt.cm.jet)
    ax.set_facecolor(plt.cm.jet(0.))
    ax.set_title('Noise density')

def format_ax(ax, range_lim):
    ax.set_xlim(-range_lim, range_lim)
    ax.set_ylim(-range_lim, range_lim)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.invert_yaxis()