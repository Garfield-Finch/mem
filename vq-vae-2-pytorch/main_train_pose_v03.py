import argparse
import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms, utils

from tqdm import tqdm

from vqvae import VQVAE
from scheduler import CycleScheduler

# from torch.utils.data import DataLoader
from tz_utils.dataloader_tz import iPERLoader


def train(epoch, loader, model, optimizer, scheduler, device):
    loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = 25

    mse_sum = 0
    mse_n = 0

    for i, (img, label) in enumerate(loader):
        model.zero_grad()

        img = img.to(device)

        out, latent_loss = model(img)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        mse_sum += recon_loss.item() * img.shape[0]
        mse_n += img.shape[0]

        lr = optimizer.param_groups[0]['lr']

        loader.set_description(
            (
                f'epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; '
                f'latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; '
                f'lr: {lr:.5f}'
            )
        )

        if i % 100 == 0:
            model.eval()

            sample = img[:sample_size]

            with torch.no_grad():
                out, _ = model(sample)

            # For the sake of visualization, add all 19 channels to 1
            sample = sample.sum(1) * 255
            out = out.sum(1) * 255
            sample = sample.unsqueeze(1)
            out = out.unsqueeze(1)

            utils.save_image(
                torch.cat([sample, out], 0),
                f'sample/pose_02/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png',
                nrow=sample_size,
                normalize=True,
                range=(-1, 1),
            )

            model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=560)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--sched', type=str)
    parser.add_argument('--path', type=str, default='/p300/dataset/iPER/')

    args = parser.parse_args()

    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    device = 'cuda'

    # TODO: Use a mini dataset set in the position of val_loader for sanity check of pose, to be changed back
    _, loader = iPERLoader(data_root=args.path, batch=8).data_load()

    model = VQVAE(in_channel=19, embed_dim=32, n_embed=64).to(device)
    # The below is the default parameters
    # in_channel = 3,
    # channel = 128,
    # n_res_block = 2,
    # n_res_channel = 32,
    # embed_dim = 64,
    # n_embed = 512,

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, args.lr, n_iter=len(loader) * args.epoch, momentum=None
        )

    for i in range(args.epoch):
        train(i, loader, model, optimizer, scheduler, device)
        torch.save(model.state_dict(), f'checkpoint/pose_02/vqvae_{str(i + 1).zfill(3)}.pt')
