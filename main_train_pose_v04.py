import argparse
import os
import visdom
from tqdm import tqdm
import socket
import numpy as np
from PIL import Image

import torch
from torch import nn, optim
# from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from utils.vqvae import VQVAE
from vq_vae_2_pytorch.scheduler import CycleScheduler
from utils.dataloader_v03 import iPERLoader


def train(epoch, loader, model, optimizer, scheduler, device):
    loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = 25

    mse_sum = 0
    mse_n = 0

    for i, (img, pose) in enumerate(loader):
        model.zero_grad()

        # Let the input be the pose skeleton
        img = pose

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

            img_save_name = f'sample/pose_04/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png'
            utils.save_image(
                torch.cat([sample, out], 0),
                img_save_name,
                nrow=sample_size,
                normalize=True,
                range=(-1, 1),
            )
            img_show = np.transpose(np.asarray(Image.open(img_save_name)), (2, 0, 1))
            viz.images(img_show, win='transfer', nrow=sample_size, opts={'title': 'gt-sample'})

            model.train()

        # increase the sequence of saving model
        if i % 200 == 0:
            torch.save(model.state_dict(), f'checkpoint/pose_04/vqvae_{str(epoch + 1).zfill(3)}.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=560)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--sched', type=str)
    parser.add_argument('--path', type=str, default='/p300/dataset/iPER/')
    parser.add_argument('--env', type=str, default='main')

    args = parser.parse_args()

    print(args)

    viz = visdom.Visdom(server='10.10.10.100', port=33241, env=args.env)

    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

    device = 'cuda'
    torch.backends.cudnn.benchmark = True

    transform = transforms.Compose(
        [
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    loader, _, _ = iPERLoader(data_root=args.path, batch=64, transform=transform).data_load()

    model = VQVAE().to(device)
    model = nn.DataParallel(model).cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, args.lr, n_iter=len(loader) * args.epoch, momentum=None
        )

    # print('Loading Model...', end='')
    # model.load_state_dict(torch.load('/p300/mem/mem_src/vq_vae_2_pytorch/checkpoint/app/vqvae_061.pt'))
    # model.eval()
    # print('Complete !')

    for i in range(args.epoch):
        train(i, loader, model, optimizer, scheduler, device)
        torch.save(model.state_dict(), f'checkpoint/pose_04/vqvae_{str(i + 1).zfill(3)}.pt')
