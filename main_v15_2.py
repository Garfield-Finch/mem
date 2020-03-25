import argparse
import os
import socket

import torch
from torch import nn, optim
from torchvision import datasets, transforms, utils

from tqdm import tqdm
import visdom
import numpy as np
from PIL import Image

from vq_vae_2_pytorch.scheduler import CycleScheduler

from utils.dataloader_v04 import iPERLoader
from utils.networks_v10 import VQVAE, AppVQVAE
from utils.networks_transfer_v01_1 import TransferModel


def train(epoch, loader, dic_model, scheduler, device):
    loader = tqdm(loader)

    model_img = dic_model['model_img']
    model_cond = dic_model['model_cond']
    model_transfer = dic_model['model_transfer']
    optimizer_img = dic_model['optimizer_img']
    optimizer_cond = dic_model['optimizer_cond']
    optimizer_transfer = dic_model['optimizer_transfer']

    #############################
    # Hyper parameters
    #############################
    criterion = nn.MSELoss()

    weight_loss_GAN = 0.02
    weight_loss_recon = 1
    weight_latent_loss = 0.25
    sample_size = 6

    # mse_sum = 0
    # mse_n = 0

    model_img.train()
    model_cond.train()

    lst_loss_quant_recon = []
    lst_loss_quant_recon_t = []
    lst_loss_quant_recon_b = []
    lst_loss_image_recon = []
    lst_loss = []

    for i, (img_s, pose_s, img_t, pose_t) in enumerate(loader):
        img_s = img_s.to(device)
        img_t = img_t.to(device)
        pose_s = pose_s.to(device)
        pose_t = pose_t.to(device)

        pose_s_out, pose_s_latent_loss, pose_s_quant_t, pose_s_quant_b = model_cond(pose_s)
        pose_t_out, pose_t_latent_loss, pose_t_quant_t, pose_t_quant_b = model_cond(pose_t)
        img_s_out, img_s_latent_loss, img_s_quant_t, img_s_quant_b = model_img(img_s)
        img_t_out, img_t_latent_loss, img_t_quant_t, img_t_quant_b = model_img(img_t)
        # quant_b.shape: [batch_size, 64, 64, 64]
        # quant_t.shape: [batch_size, 64, 32, 32]

        transfer_quant_t, transfer_quant_b = model_transfer(pose_s_quant_t, pose_t_quant_t, img_s_quant_t,
                                                            pose_s_quant_b, pose_t_quant_b, img_s_quant_b)
        transfer_input = (transfer_quant_t, transfer_quant_b)
        img_transfer_out = model_img(transfer_input, mode='TRANSFER')

        #######################
        # calculate loss
        #######################

        # loss_quant_recon
        loss_quant_recon_t = criterion(transfer_quant_t, img_t_quant_t.clone().detach())
        loss_quant_recon_b = criterion(transfer_quant_b, img_t_quant_b.clone().detach())
        loss_quant_recon = loss_quant_recon_t + loss_quant_recon_b

        # loss_image_recon
        loss_image_recon = criterion(img_transfer_out, img_t) + criterion(img_s_out, img_s)
        loss_latent = img_s_latent_loss.mean()

        # # utils to calculate loss GAN
        def _cal_gan_loss(tsr_in, key=True):
            return criterion(tsr_in, torch.ones(tsr_in.shape).cuda()) if key is True \
                else criterion(tsr_in, torch.zeros(tsr_in.shape).cuda())

        if scheduler is not None:
            scheduler.step()

        # back propagation for transfer module
        optimizer_transfer.zero_grad()
        loss = weight_loss_recon * (loss_quant_recon + loss_image_recon + weight_latent_loss * loss_latent)
               # + weight_loss_GAN * (loss_GAN_img)
        loss.backward(retain_graph=True)
        optimizer_transfer.step()
        optimizer_img.step()
        optimizer_cond.step()

        # back propagation for Discriminator
        # optimizer_D_t.zero_grad()
        # loss_D_t.backward(retain_graph=True)
        # optimizer_D_t.step()
        #
        # optimizer_D_m.zero_grad()
        # loss_D_m.backward(retain_graph=True)
        # optimizer_D_m.step()
        #
        # optimizer_D_b.zero_grad()
        # loss_D_b.backward()
        # optimizer_D_b.step()
        # optimizer_D_img.zero_grad()
        # loss_D_img.backward()
        # optimizer_D_img.step()

        # mse_sum += img_recon_loss.item() * img.shape[0]
        # mse_n += img.shape[0]

        lr = optimizer_transfer.param_groups[0]['lr']

        loader.set_description(
            (
                f'epoch: {epoch + 1}; '
                f'quant: {loss_quant_recon.item():.3f}; '
                f'image: {loss_image_recon.item():.3f}; '
                # f'G_t: {loss_GAN_t.item():.3f}; '
                # f'G_m: {loss_GAN_m.item():.3f}; '
                # f'G_b: {loss_GAN_b.item():.3f}; '
                # f'D_t: {loss_D_t.item():.3f}; '
                # f'D_m: {loss_D_m.item():.3f}; '
                # f'D_b: {loss_D_b.item():.3f}; '
                f'lr: {lr:.5f}'
                # f'mse: {img_recon_loss.item():.5f}; '
                # f'latent: {img_latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; '
            )
        )

        # for visdom visualization
        lst_loss_quant_recon.append(loss_quant_recon.item())
        lst_loss_quant_recon_t.append(loss_quant_recon_t.item())
        lst_loss_quant_recon_b.append(loss_quant_recon_b.item())
        lst_loss_image_recon.append(loss_image_recon.item())
        lst_loss.append(loss.item())

        #########################
        # Evaluation
        #########################
        if i % 100 == 0:
            # save image as file
            img_show = torch.cat([pose_s[:sample_size], pose_s_out[:sample_size],
                                  pose_t[:sample_size], pose_t_out[:sample_size],
                                  img_s_out[:sample_size], img_s[:sample_size],
                                  img_t_out[:sample_size], img_transfer_out[:sample_size], img_t[:sample_size]
                                  ])
            img_save_name = f'sample/{EXPERIMENT_CODE}/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png'
            utils.save_image(
                img_show,
                img_save_name,
                nrow=sample_size,
                normalize=True,
                range=(-1, 1),
            )

            # viz pose-pose_recon-img_out-transfer_out-gt
            img_show = np.transpose(np.asarray(Image.open(img_save_name)), (2, 0, 1))
            viz.images(img_show, win='transfer', nrow=sample_size, opts={'title': 'pose-img_out-transfer_out-gt'})

        # increase the sequence of saving model
        if i % 200 == 0:
            torch.save(model_transfer.state_dict(),
                       f'checkpoint/{EXPERIMENT_CODE}/vqvae_trans_{str(epoch + 1).zfill(3)}.pt')
            # torch.save(model_D_img.state_dict(), f'checkpoint/{EXPERIMENT_CODE}/vqvae_Di_{str(i + 1).zfill(3)}.pt')

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
    # for line_num, (lst, line_title) in enumerate(
    #         [(lst_loss_GAN_t, 'loss_GAN_t'),
    #          (lst_loss_GAN_m, 'loss_GAN_m'),
    #          (lst_loss_GAN_b, 'loss_GAN_b'),
    #          (lst_loss_D_t, 'loss_D_t'),
    #          (lst_loss_D_m, 'loss_D_m'),
    #          (lst_loss_D_b, 'loss_D_b')
    #          ]):
    #     viz.line(Y=np.array([sum(lst) / len(lst)]), X=np.array([epoch]),
    #              name=line_title,
    #              win='loss_GAN',
    #              opts=dict(title='loss_GAN', showlegend=True),
    #              update=None if (epoch == 0 and line_num == 0) else 'append'
    #              )
    # for line_num, (lst, line_title) in enumerate(
    #         [(lst_loss_GAN_t_resamble, 'loss_GAN_t_resamble'),
    #          (lst_loss_GAN_m_resamble, 'loss_GAN_m_resamble'),
    #          (lst_loss_GAN_b_resamble, 'loss_GAN_b_resamble'),
    #          ]):
    #     viz.line(Y=np.array([sum(lst) / len(lst)]), X=np.array([epoch]),
    #              name=line_title,
    #              win='loss_feature_mapping',
    #              opts=dict(title='feature mapping loss', showlegend=True),
    #              update=None if (epoch == 0 and line_num == 0) else 'append'
    #              )
    for line_num, (lst, line_title) in enumerate(
            [(lst_loss_quant_recon_t, 'loss_quant_recon_t'),
             (lst_loss_quant_recon_b, 'loss_quant_recon_b'),
             ]):
        viz.line(Y=np.array([sum(lst) / len(lst)]), X=np.array([epoch]),
                 name=line_title,
                 win='loss_quant_recon',
                 opts=dict(title='loss_quant_recon', showlegend=True),
                 update=None if (epoch == 0 and line_num == 0) else 'append'
                 )
    # for line_num, (lst, line_title) in enumerate(
    #         [(lst_loss_D_img, 'loss_D_img'),
    #          (lst_loss_GAN_img, 'loss_GAN_img'),
    #          ]):
    #     viz.line(Y=np.array([sum(lst) / len(lst)]), X=np.array([epoch]),
    #              name=line_title,
    #              win='loss_GAN_img',
    #              opts=dict(title='loss_GAN_img', showlegend=True),
    #              update=None if (epoch == 0 and line_num == 0) else 'append'
    #              )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=560)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--sched', type=str)
    parser.add_argument('--path', type=str, default='/p300/dataset/iPER/')
    parser.add_argument('--model_cond_path', type=str, default='/p300/mem/mem_src/checkpoint/pose_04'
                                                               '/vqvae_166.pt')
    parser.add_argument('--model_img_path', type=str, default='/p300/mem/mem_src/checkpoint/app'
                                                              '/vqvae_164.pt')
    parser.add_argument('--model_transfer_path', type=str, default='/p300/mem/mem_src/checkpoint_exp/as_17_transfer'
                                                                   '/vqvae_trans_560.pt')
    parser.add_argument('--env', type=str, default='main')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=8)

    args = parser.parse_args()

    print(args)

    ##############################
    # Dash Board
    ##############################
    is_load_model_img = True
    is_load_model_cond = True
    is_load_model_transfer = False
    is_load_model_discriminator = False
    EXPERIMENT_CODE = 'as_30'
    if not os.path.exists(f'checkpoint/{EXPERIMENT_CODE}/'):
        print(f'New EXPERIMENT_CODE:{EXPERIMENT_CODE}, creating saving directories ...', end='')
        os.mkdir(f'checkpoint/{EXPERIMENT_CODE}/')
        os.mkdir(f'sample/{EXPERIMENT_CODE}/')
        print('Done')
    else:
        print('EXPERIMENT_CODE already exits.')

    viz = visdom.Visdom(server='10.10.10.100', port=33241, env=args.env)
    viz.text("""
        pretrained; 
        af+ae; 
        """
             f'Hostname: {socket.gethostname()}; '
             f'file: main_v15_2.py;\n '
             f'Experiment_Code: {EXPERIMENT_CODE};\n', win='board')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
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
    # TODO use a little set for sanity check
    # _, loader, _ = iPERLoader(data_root=args.path, batch=args.batch_size, transform=transform).data_load()
    _, _, loader = iPERLoader(data_root=args.path, batch=args.batch_size, transform=transform).data_load()

    # model for image
    model_img = AppVQVAE().to(device)
    model_img = nn.DataParallel(model_img).to(device)
    if is_load_model_img is True:
        print('Loading model_img ...', end='')
        model_img.load_state_dict(torch.load(args.model_img_path))
        model_img.eval()
        print('Done')
    else:
        print('model_img Initialized.')
    optimizer_img = optim.Adam(model_img.parameters(), lr=args.lr)

    # model for condition
    model_cond = VQVAE().to(device)
    model_cond = nn.DataParallel(model_cond).cuda()
    if is_load_model_cond is True:
        print('Loading model_cond ...', end='')
        model_cond.load_state_dict(torch.load(args.model_cond_path))
        model_cond.eval()
        print('Done')
    else:
        print('model_cond Initialized.')
    optimizer_cond = optim.Adam(model_cond.parameters(), lr=args.lr)

    # transfer model
    model_transfer = TransferModel().to(device)
    model_transfer = nn.DataParallel(model_transfer).to(device)
    if is_load_model_transfer is True:
        print('Loading model_transfer ...', end='')
        model_transfer.load_state_dict(torch.load(args.model_transfer_path))
        model_transfer.eval()
        print('Done')
    else:
        print('model_transfer Initialized.')
    optimizer_transfer = optim.Adam(model_transfer.parameters(), lr=args.lr)

    scheduler = None
    # if args.sched == 'cycle':
    #     scheduler = CycleScheduler(
    #         optimizer, args.lr, n_iter=len(loader) * args.epoch, momentum=None
    #     )

    # Discriminator model
    # model_D_t = DiscriminatorModel(in_channel=64, n_layers=1).to(device)
    # model_D_m = DiscriminatorModel(in_channel=64, n_layers=2).to(device)
    # model_D_b = DiscriminatorModel(in_channel=64, n_layers=2).to(device)
    # # model_D_img = MultiscaleDiscriminator(input_nc=3, num_D=1).to(device)
    # # model_D_img = nn.DataParallel(model_D_img).cuda()
    # if is_load_model_discriminator is True:
    #     # print('Loading model_D_t ...', end='')
    #     # model_D_t.load_state_dict(torch.load(args.model_transfer_path.replace('vqvae', 'vqvae_Dt')))
    #     # model_D_t.eval()
    #     # print('Done')
    #     #
    #     # print('Loading model_D_m ...', end='')
    #     # model_D_m.load_state_dict(torch.load(args.model_transfer_path.replace('vqvae', 'vqvae_Dm')))
    #     # model_D_m.eval()
    #     # print('Done')
    #     #
    #     # print('Loading model_D_b ...', end='')
    #     # model_D_b.load_state_dict(torch.load(args.model_transfer_path.replace('vqvae', 'vqvae_Db')))
    #     # model_D_b.eval()
    #     # print('Done')
    #
    #     print('Loading model_D_img ...', end='')
    #     model_D_img.load_state_dict(torch.load(args.model_transfer_path.replace('vqvae', 'vqvae_Db')))
    #     model_D_img.eval()
    #     print('Done')
    # else:
    #     print('model_discriminator Initialized.')
    # # model_D_t = nn.DataParallel(model_D_t).cuda()
    # # optimizer_D_t = optim.Adam(model_D_t.parameters(), lr=args.lr)
    # # model_D_m = nn.DataParallel(model_D_m).cuda()
    # # optimizer_D_m = optim.Adam(model_D_m.parameters(), lr=args.lr)
    # # model_D_b = nn.DataParallel(model_D_b).cuda()
    # # optimizer_D_b = optim.Adam(model_D_b.parameters(), lr=args.lr)
    # optimizer_D_img = optim.Adam(model_D_img.parameters(), lr=args.lr)

    dic_model = {'model_img': model_img, 'model_cond': model_cond, 'model_transfer': model_transfer,
                 'optimizer_img': optimizer_img, 'optimizer_cond': optimizer_cond,
                 'optimizer_transfer': optimizer_transfer}

    for i in range(args.epoch):
        viz.text(f'epoch: {i}', win='Epoch')
        train(epoch=i, loader=loader, dic_model=dic_model, scheduler=scheduler, device=device)
        torch.save(model_transfer.state_dict(), f'checkpoint/{EXPERIMENT_CODE}/vqvae_trans_{str(i + 1).zfill(3)}.pt')
        torch.save(model_img.state_dict(), f'checkpoint/{EXPERIMENT_CODE}/vqvae_img_{str(i + 1).zfill(3)}.pt')
        torch.save(model_cond.state_dict(), f'checkpoint/{EXPERIMENT_CODE}/vqvae_cond_{str(i + 1).zfill(3)}.pt')
        # torch.save(model_D_t.state_dict(), f'checkpoint/{EXPERIMENT_CODE}/vqvae_Dt_{str(i + 1).zfill(3)}.pt')
        # torch.save(model_D_m.state_dict(), f'checkpoint/{EXPERIMENT_CODE}/vqvae_Dm_{str(i + 1).zfill(3)}.pt')
        # torch.save(model_D_b.state_dict(), f'checkpoint/{EXPERIMENT_CODE}/vqvae_Db_{str(i + 1).zfill(3)}.pt')
        # torch.save(model_D_img.state_dict(), f'checkpoint/{EXPERIMENT_CODE}/vqvae_Di_{str(i + 1).zfill(3)}.pt')
