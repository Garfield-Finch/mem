import argparse
import os
import socket

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

    #############################
    # Hyper parameters
    #############################
    criterion = nn.MSELoss()

    weight_loss_GAN = 1
    weight_loss_recon = 100
    latent_loss_weight = 0.25
    sample_size = 6

    # mse_sum = 0
    # mse_n = 0

    model_img.train()
    model_cond.train()
    model_transfer.train()
    model_D_t.train()
    model_D_b.train()

    lst_loss_quant_recon = []
    lst_loss_image_recon = []
    lst_loss = []
    lst_loss_GAN_t = []
    lst_loss_GAN_b = []
    lst_loss_D_t = []
    lst_loss_D_b = []
    lst_loss_GAN_t_resamble = []
    lst_loss_GAN_b_resamble = []

    for i, (img, pose) in enumerate(loader):
        img = img.to(device)
        pose = pose.to(device)

        pose_out, pose_latent_loss, pose_quant_t, pose_quant_b = model_cond(pose)
        img_out, img_latent_loss, img_quant_t, img_quant_b = model_img(img)
        # quant_b.shape: [batch_size, 64, 64, 64]
        # quant_t.shape: [batch_size, 64, 32, 32]

        transfer_quant_t, transfer_quant_b = model_transfer(pose_quant_t, pose_quant_b)
        # transfer_input = (transfer_quant_t, transfer_quant_b)
        # transfer_out = model_img(transfer_input, mode='TRANSFER')
        transfer_out = model_img.seq2quant_decode(transfer_quant_t, transfer_quant_b)
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
        loss_GAN_t = criterion(discriminator_transfer_quant_t, gt_D_t_true)
        loss_GAN_b = criterion(discriminator_transfer_quant_b, gt_D_b_true)

        # loss_discriminator
        loss_D_t = criterion(discriminator_transfer_quant_t, gt_D_t_false) \
                   + criterion(discriminator_img_quant_t, gt_D_t_true)
        loss_D_b = criterion(discriminator_transfer_quant_b, gt_D_b_false)\
                   + criterion(discriminator_img_quant_b, gt_D_b_true)

        # loss_GAN_resamble: feature mapping loss
        loss_GAN_t_resamble = criterion(discriminator_transfer_quant_t, discriminator_img_quant_t)
        loss_GAN_b_resamble = criterion(discriminator_transfer_quant_b, discriminator_img_quant_b)

        # img_recon_loss = criterion(img_out, img)
        # img_latent_loss = img_latent_loss.mean()
        # img_loss = img_recon_loss + latent_loss_weight * img_latent_loss
        # img_loss.backward()

        if scheduler is not None:
            scheduler.step()

        # back propagation for transfer module
        optimizer.zero_grad()
        loss = weight_loss_recon * (loss_quant_recon + loss_image_recon)\
               + weight_loss_GAN * (loss_GAN_t + loss_GAN_b + loss_GAN_t_resamble + loss_GAN_b_resamble)
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
                f'quant: {loss_quant_recon.item():.3f}; '
                f'image: {loss_image_recon.item():.3f}; '
                f'G_t: {loss_GAN_t.item():.3f}; '
                f'G_b: {loss_GAN_b.item():.3f}; '
                f'\n'
                f'D_t: {loss_D_t.item():.3f}; '
                f'D_b: {loss_D_b.item():.3f}; '
                f'lr: {lr:.5f}'
                f'\n'
                # f'epoch: {epoch + 1}; mse: {img_recon_loss.item():.5f}; '
                # f'latent: {img_latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; '
                # f'lr: {lr:.5f}'
            )
        )

        # for visdom visualization
        lst_loss_quant_recon.append(loss_quant_recon.item())
        lst_loss_image_recon.append(loss_image_recon.item())
        lst_loss.append(loss.item())
        lst_loss_D_t.append(loss_D_t.item())
        lst_loss_D_b.append(loss_D_b.item())
        lst_loss_GAN_t.append(loss_GAN_t.item())
        lst_loss_GAN_b.append(loss_GAN_b.item())
        lst_loss_GAN_t_resamble.append((loss_GAN_t_resamble.item()))
        lst_loss_GAN_b_resamble.append((loss_GAN_b_resamble.item()))

        #########################
        # Evaluation
        #########################
        if i % 100 == 0:
            # save image as file
            img_show = torch.cat([pose[:sample_size], pose_out[:sample_size], img_out[:sample_size],
                                  transfer_out[:sample_size], img[:sample_size]])
            utils.save_image(
                img_show,
                f'sample/{EXPERIMENT_CODE}/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png',
                nrow=sample_size,
                normalize=True,
                range=(-1, 1),
            )

            # viz pose-pose_recon-img_out-transfer_out-gt
            img_show = img_show.to('cpu').detach().numpy()
            img_show = (img_show * 0.5 + 0.5) * 255
            viz.images(img_show, win='transfer', nrow=sample_size, opts={'title': 'pose-img_out-transfer_out-gt'})

        # increase the sequence of saving model
        if i % 200 == 0:
            torch.save(model_transfer.state_dict(), f'checkpoint/{EXPERIMENT_CODE}/vqvae_{str(epoch + 1).zfill(3)}.pt')

    #########################
    # Plot loss to visdom
    #########################
    for line_num, (plot_y, line_title) in enumerate(
            [(sum(lst_loss_quant_recon) / len(lst_loss_quant_recon), 'loss_quant_recon'),
             (sum(lst_loss_image_recon) / len(lst_loss_image_recon), 'loss_image_recon'),
             (sum(lst_loss) / len(lst_loss), 'loss'),
             ]):
        viz.line(Y=np.array([plot_y]), X=np.array([epoch]),
                 name=line_title,
                 win='loss',
                 opts=dict(title='loss', showlegend=True),
                 update=None if (epoch == 0 and line_num == 0) else 'append'
                 )
    for line_num, (lst, line_title) in enumerate(
            [(lst_loss_GAN_t, 'loss_GAN_t'),
             (lst_loss_GAN_b, 'loss_GAN_b'),
             (lst_loss_D_t, 'loss_D_t'),
             (lst_loss_D_b, 'loss_D_b')
             ]):
        viz.line(Y=np.array([sum(lst) / len(lst)]), X=np.array([epoch]),
                 name=line_title,
                 win='loss_GAN',
                 opts=dict(title='loss_GAN', showlegend=True),
                 update=None if (epoch == 0 and line_num == 0) else 'append'
                 )
    for line_num, (lst, line_title) in enumerate(
            [(lst_loss_GAN_t_resamble, 'loss_GAN_t_resamble'),
             (lst_loss_GAN_b_resamble, 'loss_GAN_b_resamble'),
             ]):
        viz.line(Y=np.array([sum(lst) / len(lst)]), X=np.array([epoch]),
                 name=line_title,
                 win='loss_feature_mapping',
                 opts=dict(title='feature mapping loss', showlegend=True),
                 update=None if (epoch == 0 and line_num == 0) else 'append'
                 )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=560)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--sched', type=str)
    parser.add_argument('--path', type=str, default='/p300/dataset/iPER/')
    parser.add_argument('--model_cond_path', type=str, default='/p300/mem/mem_src/vq-vae-2-pytorch/checkpoint/pose_04'
                                                               '/vqvae_101.pt')
    parser.add_argument('--model_img_path', type=str, default='/p300/mem/mem_src/vq-vae-2-pytorch/checkpoint/app'
                                                              '/vqvae_052.pt')
    parser.add_argument('--model_transfer_path', type=str, default='/p300/mem/mem_src/vq-vae-2-pytorch/checkpoint/as_07'
                                                                   '/vqvae_055.pt')
    parser.add_argument('--env', type=str, default='main')
    parser.add_argument('--gpu', type=str, default='0')

    args = parser.parse_args()

    print(args)

    DESCRIPTION = """
    With a discriminator in latent space, n_layers decreased to 1 and 2 for t and b respectively; 
    add weight for loss_GAN being 1 and other components of loss are amplified by 100 times
    add feature mapping loss
    use network_v02.py; 
    loss = weight_loss_recon * (loss_quant_recon + loss_image_recon) 
    + weight_loss_GAN * (loss_GAN_t + loss_GAN_b + loss_GAN_t_resamble + loss_GAN_b_resamble)
    """

    EXPERIMENT_CODE = 'as_13_DFWlFm'
    if not os.path.exists(f'checkpoint/{EXPERIMENT_CODE}/'):
        print(f'New EXPERIMENT_CODE:{EXPERIMENT_CODE}, creating saving directories ...', end='')
        os.mkdir(f'checkpoint/{EXPERIMENT_CODE}/')
        os.mkdir(f'sample/{EXPERIMENT_CODE}/')
        print('Done')
    else:
        print('EXPERIMENT_CODE already exits.')

    viz = visdom.Visdom(server='10.10.10.100', port=33241, env=args.env)
    viz.text(f'{DESCRIPTION}'
             f'Hostname: {socket.gethostname()}; '
             f'file: main_v08_1_DSWl.py;\n '
             f'Experiment_Code: {EXPERIMENT_CODE};\n', win='board')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda'

    BATCH_SIZE = 64 * 2
    transform = transforms.Compose(
        [
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    # # TODOn use a little set for sanity check
    # _, loader = iPERLoader(data_root=args.path, batch=BATCH_SIZE, transform=transform).data_load()
    loader, _ = iPERLoader(data_root=args.path, batch=BATCH_SIZE, transform=transform).data_load()

    # model for image
    model_img = VQVAE().to(device)
    print('Loading model_img ...', end='')
    model_img.load_state_dict(torch.load(args.model_img_path))
    print('Done')
    model_img.eval()
    model_img = nn.DataParallel(model_img).cuda()
    # optimizer_img = optim.Adam(model_img.parameters(), lr=args.lr)

    # model for condition
    model_cond = VQVAE().to(device)
    print('Loading model_cond ...', end='')
    model_cond.load_state_dict(torch.load(args.model_cond_path))
    print('Done')
    model_cond.eval()
    model_cond = nn.DataParallel(model_cond).cuda()
    # optimizer_cond = optim.Adam(model_cond.parameters(), lr=args.lr)

    # transfer model
    model_transfer = TransferModel().to(device)
    # print('Loading model_transfer ...', end='')
    # model_transfer.load_state_dict(torch.load(args.model_transfer_path))
    # model_transfer.eval()
    # print('Done')
    model_transfer = nn.DataParallel(model_transfer).cuda()
    optimizer = optim.Adam(model_transfer.parameters(), lr=args.lr)

    scheduler = None
    if args.sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, args.lr, n_iter=len(loader) * args.epoch, momentum=None
        )

    # Discriminator model
    model_D_t = DiscriminatorModel(in_channel=64, n_layers=1).to(device)
    model_D_t = nn.DataParallel(model_D_t).cuda()
    # print('Loading model_D_t ...', end='')
    # model_D_t.load_state_dict(torch.load(args.model_transfer_path.replace('vqvae', 'vqvae_Dt')))
    # model_D_t.eval()
    # print('Done')
    optimizer_D_t = optim.Adam(model_D_t.parameters(), lr=args.lr)

    model_D_b = DiscriminatorModel(in_channel=64, n_layers=2).to(device)
    # print('Loading model_D_b ...', end='')
    # model_D_b.load_state_dict(torch.load(args.model_transfer_path.replace('vqvae', 'vqvae_Db')))
    # model_D_b.eval()
    # print('Done')
    model_D_b = nn.DataParallel(model_D_b).cuda()
    optimizer_D_b = optim.Adam(model_D_b.parameters(), lr=args.lr)

    NUM_BATCH = loader.dataset.__len__() // BATCH_SIZE

    for i in range(args.epoch):
        viz.text(f'epoch: {i}', win='Epoch')
        train_transfer(epoch=i, loader=loader, model_transfer=model_transfer, model_img=model_img,
                       model_cond=model_cond, model_D_t=model_D_t, model_D_b=model_D_b,
                       optimizer=optimizer, optimizer_D_t=optimizer_D_t, optimizer_D_b=optimizer_D_b,
                       scheduler=scheduler, device=device)
        torch.save(model_transfer.state_dict(), f'checkpoint/{EXPERIMENT_CODE}/vqvae_{str(i + 1).zfill(3)}.pt')
        torch.save(model_D_t.state_dict(), f'checkpoint/{EXPERIMENT_CODE}/vqvae_Dt_{str(i + 1).zfill(3)}.pt')
        torch.save(model_D_b.state_dict(), f'checkpoint/{EXPERIMENT_CODE}/vqvae_Db_{str(i + 1).zfill(3)}.pt')
