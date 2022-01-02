"""
Implementation of Noise Contrastive Estimation (NCE) on 2D dataset.
"""
import argparse
import os
import torch
import torch.distributions as D
from model import EBM
import util

device = torch.device('cuda')

parser = argparse.ArgumentParser(description='Noise Contrastive Estimation')
parser.add_argument('--epoch', default=100, type=int, help='number of training epochs')
parser.add_argument('--batch', default=100, type=int, help='batch size')
parser.add_argument('--dataset', default='8gaussians', type=str, choices=['8gaussians', '2spirals', 'checkerboard', 'rings', 'pinwheel'], help='2D dataset to use') 
parser.add_argument('--samples', default=10000, type=int, help='number of 2D samples for training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument('--resume', type=bool, default=False, help='Resume from checkpoint')
args = parser.parse_args()

# ------------------------------
# I. MODELS
# ------------------------------
energy = EBM(dim=2).to(device)
noise = D.MultivariateNormal(torch.zeros(2).to(device), 4.*torch.eye(2).to(device))
# ------------------------------
# II. OPTIMIZERS
# ------------------------------
optim_energy = torch.optim.Adam(energy.parameters(), lr=args.lr, betas=(args.b1, args.b2))
# ------------------------------
# III. DATA LOADER
# ------------------------------
dataset, dataloader = util.get_data(args)
# ------------------------------
# IV. TRAINING
# ------------------------------
def main(args):
    start_epoch = 0
# ----------------------------------------------------------------- #
    if args.resume:
        print('Resuming from checkpoint at ckpts/nce.pth.tar...')
        checkpoint = torch.load('ckpts/nce.pth.tar')
        energy.load_state_dict(checkpoint['energy'])
        start_epoch = checkpoint['epoch'] + 1
# ----------------------------------------------------------------- #
    for epoch in range(start_epoch, start_epoch + args.epoch):
        for i, x in enumerate(dataloader):           
            x = x.to(device)
            # -----------------------------
            #  Generate samples from noise
            # -----------------------------
            gen = noise.sample((args.batch,))
            # -----------------------------
            #  Train Energy-Based Model
            # -----------------------------
            optim_energy.zero_grad()

            loss_energy, acc = util.value(energy, noise, x, gen)

            loss_energy.backward()
            optim_energy.step()  

            print(
                "[Epoch %d/%d] [Batch %d/%d] [Value: %f] [Accuracy:%f]"
                % (epoch, start_epoch + args.epoch, i, len(dataloader), loss_energy.item(), acc)
            )


        # Save checkpoint
        print('Saving models...')
        state = {
        'energy': energy.state_dict(),
        'value': loss_energy,
        'epoch': epoch,
        }
        os.makedirs('ckpts', exist_ok=True)
        torch.save(state, 'ckpts/nce.pth.tar')

        # visualization
        util.plot(dataset, energy, noise, epoch, device)


if __name__ == '__main__':
    print(args)
    main(args)