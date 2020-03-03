"""
With a discriminator in latent space
"""

import argparse
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from tqdm import tqdm
import visdom
import numpy as np

from scheduler import CycleScheduler

from tz_utils.dataloader_v02 import iPERLoader
from tz_utils.networks_v02 import TransferModel, VQVAE, DiscriminatorModel
# from tz_utils.vqvae_tz import VQVAE


def train_transfer(epoch, loader, model_transfer, model_img, model_cond, model_D_t, model_D_b,
                   optimizer, optimizer_D_t, optimizer_D_b, scheduler, device):
    loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = 6

    mse_sum = 0
    mse_n = 0

    model_img.train()
    model_cond.train()
    model_transfer.train()
    model_D_t.train()
    model_D_b.train()

    lst_loss_quant_recon = []
    lst_loss_image_recon = []
    lst_loss = []

    for i, (img, pose) in enumerate(loader):
        img = img.to(device)
        pose = pose.to(device)

        pose_out, pose_latent_loss, pose_quant_t, pose_quant_b = model_cond(pose)
        img_out, img_latent_loss, img_quant_t, img_quant_b = model_img(img)
        # quant_b.shape: [batch_size, 64, 64, 64]
        # quant_t.shape: [batch_size, 64, 32, 32]

        transfer_quant_t, transfer_quant_b = model_transfer(pose_quant_t, pose_quant_b)
        transfer_input = (transfer_quant_t, transfer_quant_b)
        transfer_out = model_img(transfer_input, mode='TRANSFER')
        discriminator_transfer_quant_t = model_D_t(transfer_quant_t)
        discriminator_img_quant_t = model_D_t(img_quant_t)
        discriminator_transfer_quant_b = model_D_b(transfer_quant_b)
        discriminator_img_quant_b = model_D_b(img_quant_b)

        #######################
        # calculate loss
        #######################

        # loss_quant_recon
        loss_quant_recon_t = criterion(transfer_quant_t, img_quant_t.clone().detach())
        loss_quant_recon_b = criterion(transfer_quant_b, img_quant_b.clone().detach())
        loss_quant_recon = loss_quant_recon_t + loss_quant_recon_b

        # loss_image_recon
        loss_image_recon = criterion(transfer_out, img)

        # utils to calculate loss GAN
        gt_D_t_false = torch.zeros(discriminator_transfer_quant_t.shape).cuda()
        gt_D_t_true = torch.ones(discriminator_transfer_quant_t.shape).cuda()
        gt_D_b_false = torch.zeros(discriminator_transfer_quant_b.shape).cuda()
        gt_D_b_true = torch.ones(discriminator_transfer_quant_b.shape).cuda()

        # loss_GAN
        loss_GAN = criterion(discriminator_transfer_quant_t, gt_D_t_true)\
                   + criterion(discriminator_transfer_quant_b, gt_D_b_true)

        # loss_discriminator
        loss_D_t = criterion(discriminator_transfer_quant_t, gt_D_t_false) \
                   + criterion(discriminator_img_quant_t, gt_D_t_true)
        loss_D_b = criterion(discriminator_transfer_quant_b, gt_D_b_false)\
                   + criterion(discriminator_img_quant_b, gt_D_b_true)

        # img_recon_loss = criterion(img_out, img)
        # img_latent_loss = img_latent_loss.mean()
        # img_loss = img_recon_loss + latent_loss_weight * img_latent_loss
        # img_loss.backward()

        if scheduler is not None:
            scheduler.step()

        # back propagation for transfer module
        optimizer.zero_grad()
        # TODO there are superparameters here
        loss = loss_quant_recon + loss_image_recon + loss_GAN
        loss.backward(retain_graph=True)
        optimizer.step()

        # back propagation for Discriminator
        optimizer_D_t.zero_grad()
        loss_D_t.backward(retain_graph=True)
        optimizer_D_t.step()

        optimizer_D_b.zero_grad()
        loss_D_b.backward()
        optimizer_D_b.step()

        # mse_sum += img_recon_loss.item() * img.shape[0]
        # mse_n += img.shape[0]

        lr = optimizer.param_groups[0]['lr']

        loader.set_description(
            (
                f'epoch: {epoch + 1}; '
                f'quant: {loss_quant_recon.item():.5f}; '
                f'image: {loss_image_recon.item():.3f}; '
                f'loss_GAN: {loss_GAN.item():.3f}; '
                f'D_t: {loss_D_t.item():.3f}; '
                f'D_b: {loss_D_b.item():.3f}; '
                f'lr: {lr:.5f}'
                # f'epoch: {epoch + 1}; mse: {img_recon_loss.item():.5f}; '
                # f'latent: {img_latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; '
                # f'lr: {lr:.5f}'
            )
        )

        # for visdom visualization
        lst_loss_quant_recon.append(loss_quant_recon.item())
        lst_loss_image_recon.append(loss_image_recon.item())
        lst_loss.append(loss.item())

        #########################
        # Evaluation
        #########################
        if i % 100 == 0:
            model_img.eval()
            sample = img[:sample_size]
            with torch.no_grad():
                out, _, _, _ = model_img(sample)

            # # viz recon-gt
            # img_show = torch.cat([img_out[:sample_size], img[:sample_size]]).to('cpu').detach().numpy()
            # img_show = (img_show * 0.5 + 0.5) * 255
            # viz.images(img_show, win='recon-gt', nrow=sample_size, opts={'title': 'img_out-gt'})

            # save image as file
            img_show = torch.cat([pose[:sample_size], pose_out[:sample_size], img_out[:sample_size],
                                  transfer_out[:sample_size], img[:sample_size]])
            utils.save_image(
                # torch.cat([sample, out], 0),
                img_show,
                f'sample/as_04/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png',
                nrow=sample_size,
                normalize=True,
                range=(-1, 1),
            )

            # viz pose-pose_recon-img_out-transfer_out-gt
            img_show = img_show.to('cpu').detach().numpy()
            img_show = (img_show * 0.5 + 0.5) * 255
            viz.images(img_show, win='transfer', nrow=sample_size, opts={
                'title': 'pose-img_out-transfer_out-gt',
            }
                       )

            model_img.train()
            model_transfer.train()

        #########################
        # increase the sequence of saving model
        #########################
        if i % 200 == 0:
            torch.save(model_transfer.state_dict(), f'checkpoint/as_04/vqvae_{str(epoch + 1).zfill(3)}.pt')

    #########################
    # Plot loss to visdom
    #########################
    for plot_y, line_title in [(sum(lst_loss_quant_recon) / len(lst_loss_quant_recon), 'loss_quant_recon'),
                               (sum(lst_loss_image_recon) / len(lst_loss_image_recon), 'loss_image_recon'),
                               (sum(lst_loss) / len(lst_loss), 'loss')]:
        viz.line(Y=np.array([plot_y]), X=np.array([epoch]),
                 name=line_title,
                 win='loss',
                 opts=dict(title='loss', showlegend=True),
                 update='append' if (epoch > 0) else None
                 )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=560)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--sched', type=str)
    parser.add_argument('--path', type=str, default='/p300/dataset/iPER/')
    parser.add_argument('--model_cond_path', type=str, default='/p300/mem/mem_src/vq-vae-2-pytorch/checkpoint/pose_04'
                                                               '/vqvae_075.pt')
    parser.add_argument('--model_img_path', type=str, default='/p300/mem/mem_src/vq-vae-2-pytorch/checkpoint/app'
                                                              '/vqvae_027.pt')
    parser.add_argument('--model_transfer_path', type=str, default='/p300/mem/mem_src/vq-vae-2-pytorch/checkpoint/as'
                                                                   '/vqvae_560.pt')
    parser.add_argument('--env', type=str, default='main')
    parser.add_argument('--gpu', type=str, default='0')

    args = parser.parse_args()

    print(args)

    viz = visdom.Visdom(server='10.10.10.100', port=33241, env=args.env)
    viz.text('file: main_v02.py;\n exp: as_04;\n', win='board')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda'

    BATCH_SIZE = 25
    transform = transforms.Compose(
        [
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    # TODO use a little set for sanity check
    _, loader = iPERLoader(data_root=args.path, batch=BATCH_SIZE, transform=transform).data_load()

    # model for image
    model_img = VQVAE().to(device)
    model_img.load_state_dict(torch.load(args.model_img_path))
    model_img.eval()
    # optimizer_img = optim.Adam(model_img.parameters(), lr=args.lr)

    # model for condition
    model_cond = VQVAE().to(device)
    model_cond.load_state_dict(torch.load(args.model_cond_path))
    model_cond.eval()
    # optimizer_cond = optim.Adam(model_cond.parameters(), lr=args.lr)

    # transfer model
    model_transfer = TransferModel().to(device)
    model_transfer.load_state_dict(torch.load(args.model_transfer_path))
    model_transfer.eval()
    optimizer = optim.Adam(model_transfer.parameters(), lr=args.lr)

    scheduler = None
    if args.sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, args.lr, n_iter=len(loader) * args.epoch, momentum=None
        )

    # Discriminator model
    model_D_t = DiscriminatorModel(in_channel=64).to(device)
    optimizer_D_t = optim.Adam(model_D_t.parameters(), lr=args.lr)

    model_D_b = DiscriminatorModel(in_channel=64).to(device)
    optimizer_D_b = optim.Adam(model_D_b.parameters(), lr=args.lr)

    NUM_BATCH = loader.dataset.__len__() // BATCH_SIZE

    for i in range(args.epoch):
        train_transfer(epoch=i, loader=loader, model_transfer=model_transfer, model_img=model_img,
                       model_cond=model_cond, model_D_t=model_D_t, model_D_b=model_D_b,
                       optimizer=optimizer, optimizer_D_t=optimizer_D_t, optimizer_D_b=optimizer_D_b,
                       scheduler=scheduler, device=device)
        torch.save(model_transfer.state_dict(), f'checkpoint/as_04/vqvae_{str(i + 1).zfill(3)}.pt')
