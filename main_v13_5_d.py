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

from utils.dataloader_v03 import iPERLoader
from utils.networks_v09 import TransferModel, VQVAE, MultiscaleDiscriminator


def train(epoch, loader, model_transfer, model_img, model_cond, model_D_img,
          optimizer, optimizer_D_img, scheduler, device):
    loader = tqdm(loader)

    #############################
    # Hyper parameters
    #############################
    criterion = nn.MSELoss()

    weight_loss_GAN = 1
    weight_loss_recon = 1
    latent_loss_weight = 0.25
    sample_size = 6

    # mse_sum = 0
    # mse_n = 0

    model_img.train()
    model_cond.train()
    model_transfer.train()
    # model_D_t.train()
    # model_D_m.train()
    # model_D_b.train()
    model_D_img.train()

    lst_loss_quant_recon = []
    lst_loss_quant_recon_t = []
    lst_loss_quant_recon_m = []
    lst_loss_quant_recon_b = []
    lst_loss_image_recon = []
    lst_loss = []
    # lst_loss_GAN_t = []
    # lst_loss_GAN_m = []
    # lst_loss_GAN_b = []
    # lst_loss_D_t = []
    # lst_loss_D_m = []
    # lst_loss_D_b = []
    # lst_loss_GAN_t_resamble = []
    # lst_loss_GAN_m_resamble = []
    # lst_loss_GAN_b_resamble = []
    lst_loss_D_img = []
    lst_loss_GAN_img = []

    for i, (img, pose) in enumerate(loader):
        img = img.to(device)
        pose = pose.to(device)

        pose_out, pose_latent_loss, pose_quant_t, pose_quant_m, pose_quant_b = model_cond(pose)
        img_out, img_latent_loss, img_quant_t, img_quant_m, img_quant_b = model_img(img)
        # quant_b.shape: [batch_size, 64, 128, 128]
        # quant_m.shape: [batch_size, 64, 64, 64]
        # quant_t.shape: [batch_size, 64, 32, 32]

        transfer_quant_t, transfer_quant_m, transfer_quant_b = model_transfer(pose_quant_t, pose_quant_m, pose_quant_b)
        transfer_input = (transfer_quant_t, transfer_quant_m, transfer_quant_b)
        transfer_out = model_img(transfer_input, mode='TRANSFER')

        # discriminator_transfer_quant_t = model_D_t(transfer_quant_t)
        # discriminator_img_quant_t = model_D_t(img_quant_t)
        #
        # discriminator_transfer_quant_m = model_D_m(transfer_quant_m)
        # discriminator_img_quant_m = model_D_m(img_quant_m)
        #
        # discriminator_transfer_quant_b = model_D_b(transfer_quant_b)
        # discriminator_img_quant_b = model_D_b(img_quant_b)

        lst_discriminator_transfer_out = model_D_img(transfer_out)
        lst_discriminator_img = model_D_img(img)

        #######################
        # calculate loss
        #######################

        # loss_quant_recon
        loss_quant_recon_t = criterion(transfer_quant_t, img_quant_t.clone().detach())
        loss_quant_recon_m = criterion(transfer_quant_m, img_quant_m.clone().detach())
        loss_quant_recon_b = criterion(transfer_quant_b, img_quant_b.clone().detach())
        loss_quant_recon = loss_quant_recon_t + loss_quant_recon_m + loss_quant_recon_b

        # loss_image_recon
        loss_image_recon = criterion(transfer_out, img)

        # # utils to calculate loss GAN
        def _cal_gan_loss(tsr_in, key=True):
            return criterion(tsr_in, torch.ones(tsr_in.shape).cuda()) if key is True \
                else criterion(tsr_in, torch.zeros(tsr_in.shape).cuda())

        # # loss_GAN
        # loss_GAN_t = criterion(discriminator_transfer_quant_t, gt_D_t_true)
        # loss_GAN_m = criterion(discriminator_transfer_quant_m, gt_D_m_true)
        # loss_GAN_b = criterion(discriminator_transfer_quant_b, gt_D_b_true)
        loss_GAN_img = _cal_gan_loss(lst_discriminator_transfer_out[0][0], True)
        loss_GAN_img += _cal_gan_loss(lst_discriminator_transfer_out[1][0], True)
        loss_GAN_img += _cal_gan_loss(lst_discriminator_transfer_out[2][0], True)

        #
        # # loss_discriminator
        # loss_D_t = criterion(discriminator_transfer_quant_t, gt_D_t_false) \
        #            + criterion(discriminator_img_quant_t, gt_D_t_true)
        # loss_D_m = criterion(discriminator_transfer_quant_m, gt_D_m_false) \
        #            + criterion(discriminator_img_quant_m, gt_D_m_true)
        # loss_D_b = criterion(discriminator_transfer_quant_b, gt_D_b_false)\
        #            + criterion(discriminator_img_quant_b, gt_D_b_true)
        loss_D_img = _cal_gan_loss(lst_discriminator_transfer_out[0][0], False) +\
                     _cal_gan_loss(lst_discriminator_img[0][0], True)
        for j in range(1, len(lst_discriminator_transfer_out)):
            loss_D_img += _cal_gan_loss(lst_discriminator_transfer_out[j][0], True)
            loss_D_img += _cal_gan_loss(lst_discriminator_img[j][0], True)
        #
        # # loss_GAN_resamble: feature mapping loss
        # loss_GAN_t_resamble = criterion(discriminator_transfer_quant_t, discriminator_img_quant_t)
        # loss_GAN_m_resamble = criterion(discriminator_transfer_quant_m, discriminator_img_quant_m)
        # loss_GAN_b_resamble = criterion(discriminator_transfer_quant_b, discriminator_img_quant_b)

        # img_recon_loss = criterion(img_out, img)
        # img_latent_loss = img_latent_loss.mean()
        # img_loss = img_recon_loss + latent_loss_weight * img_latent_loss
        # img_loss.backward()

        if scheduler is not None:
            scheduler.step()

        # back propagation for transfer module
        if i % 5 == 0:
            optimizer.zero_grad()
            loss = weight_loss_recon * (loss_quant_recon + loss_image_recon) \
                   + weight_loss_GAN * (loss_GAN_img)
            loss.backward(retain_graph=True)
            optimizer.step()
        else:
            loss = weight_loss_recon * (loss_quant_recon + loss_image_recon) \
                   + weight_loss_GAN * (loss_GAN_img)
            optimizer.zero_grad()

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
        optimizer_D_img.zero_grad()
        loss_D_img.backward()
        optimizer_D_img.step()

        # mse_sum += img_recon_loss.item() * img.shape[0]
        # mse_n += img.shape[0]

        lr = optimizer.param_groups[0]['lr']

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
                f'D_img: {loss_D_img.item():.3f}; '
                f'G_img: {loss_GAN_img.item():.3f}; '
                f'lr: {lr:.5f}'
                # f'epoch: {epoch + 1}; mse: {img_recon_loss.item():.5f}; '
                # f'latent: {img_latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; '
                # f'lr: {lr:.5f}'
            )
        )

        # for visdom visualization
        lst_loss_quant_recon.append(loss_quant_recon.item())
        lst_loss_quant_recon_t.append(loss_quant_recon_t.item())
        lst_loss_quant_recon_m.append(loss_quant_recon_m.item())
        lst_loss_quant_recon_b.append(loss_quant_recon_b.item())
        lst_loss_image_recon.append(loss_image_recon.item())
        lst_loss.append(loss.item())
        # lst_loss_D_t.append(loss_D_t.item())
        # lst_loss_D_m.append(loss_D_m.item())
        # lst_loss_D_b.append(loss_D_b.item())
        # lst_loss_GAN_t.append(loss_GAN_t.item())
        # lst_loss_GAN_m.append(loss_GAN_m.item())
        # lst_loss_GAN_b.append(loss_GAN_b.item())
        # lst_loss_GAN_t_resamble.append((loss_GAN_t_resamble.item()))
        # lst_loss_GAN_m_resamble.append((loss_GAN_m_resamble.item()))
        # lst_loss_GAN_b_resamble.append((loss_GAN_b_resamble.item()))
        lst_loss_D_img.append(loss_D_img.item())
        lst_loss_GAN_img.append(loss_GAN_img.item())

        #########################
        # Evaluation
        #########################
        if i % 100 == 0:
            # save image as file
            img_show = torch.cat([pose[:sample_size], pose_out[:sample_size], img_out[:sample_size],
                                  transfer_out[:sample_size], img[:sample_size]])
            img_save_name = f'sample/{EXPERIMENT_CODE}/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png'
            utils.save_image(
                img_show,
                img_save_name,
                nrow=sample_size,
                normalize=True,
                range=(-1, 1),
            )

            # viz pose-pose_recon-img_out-transfer_out-gt
            # img_show = img_show.to('cpu').detach().numpy()
            # img_show = (img_show * 0.5 + 0.5) * 255
            img_show = np.transpose(np.asarray(Image.open(img_save_name)), (2, 0, 1))
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
             (lst_loss_quant_recon_m, 'loss_quant_recon_m'),
             (lst_loss_quant_recon_b, 'loss_quant_recon_b'),
             ]):
        viz.line(Y=np.array([sum(lst) / len(lst)]), X=np.array([epoch]),
                 name=line_title,
                 win='loss_quant_recon',
                 opts=dict(title='loss_quant_recon', showlegend=True),
                 update=None if (epoch == 0 and line_num == 0) else 'append'
                 )
    for line_num, (lst, line_title) in enumerate(
            [(lst_loss_D_img, 'loss_D_img'),
             (lst_loss_GAN_img, 'loss_GAN_img'),
             ]):
        viz.line(Y=np.array([sum(lst) / len(lst)]), X=np.array([epoch]),
                 name=line_title,
                 win='loss_GAN_img',
                 opts=dict(title='loss_GAN_img', showlegend=True),
                 update=None if (epoch == 0 and line_num == 0) else 'append'
                 )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=560)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--sched', type=str)
    parser.add_argument('--path', type=str, default='/p300/dataset/iPER/')
    parser.add_argument('--model_cond_path', type=str, default='/p300/mem/mem_src/checkpoint_exp/as_17_transfer'
                                                               '/vqvae_cond_560.pt')
    parser.add_argument('--model_img_path', type=str, default='/p300/mem/mem_src/checkpoint_exp/as_17_transfer'
                                                              '/vqvae_img_560.pt')
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
    is_load_model_transfer = True
    is_load_model_discriminator = False
    EXPERIMENT_CODE = 'as_27'
    if not os.path.exists(f'checkpoint/{EXPERIMENT_CODE}/'):
        print(f'New EXPERIMENT_CODE:{EXPERIMENT_CODE}, creating saving directories ...', end='')
        os.mkdir(f'checkpoint/{EXPERIMENT_CODE}/')
        os.mkdir(f'sample/{EXPERIMENT_CODE}/')
        print('Done')
    else:
        print('EXPERIMENT_CODE already exits.')
    DESCRIPTION = """
        multiscale-discriminator; 
        mem3 VQ-VAE;  
        use network_v09.py; 
        loss = weight_loss_recon * (loss_quant_recon + loss_image_recon)
               + weight_loss_GAN * (loss_GAN_img)
        """

    viz = visdom.Visdom(server='10.10.10.100', port=33241, env=args.env)
    viz.text(f'{DESCRIPTION}'
             f'Hostname: {socket.gethostname()}; '
             f'file: main_v13_4.py;\n '
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
    _, loader, _ = iPERLoader(data_root=args.path, batch=args.batch_size, transform=transform).data_load()
    # _, _, loader = iPERLoader(data_root=args.path, batch=args.batch_size, transform=transform).data_load()

    # model for image
    model_img = VQVAE().to(device)
    model_img = nn.DataParallel(model_img).cuda()
    if is_load_model_img is True:
        print('Loading model_img ...', end='')
        model_img.load_state_dict(torch.load(args.model_img_path))
        model_img.eval()
        print('Done')
    else:
        print('model_img Initialized.')
    # optimizer_img = optim.Adam(model_img.parameters(), lr=args.lr)

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
    # optimizer_cond = optim.Adam(model_cond.parameters(), lr=args.lr)

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
    optimizer = optim.Adam(model_transfer.parameters(), lr=args.lr)

    scheduler = None
    if args.sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, args.lr, n_iter=len(loader) * args.epoch, momentum=None
        )

    # Discriminator model
    # model_D_t = DiscriminatorModel(in_channel=64, n_layers=1).to(device)
    # model_D_m = DiscriminatorModel(in_channel=64, n_layers=2).to(device)
    # model_D_b = DiscriminatorModel(in_channel=64, n_layers=2).to(device)
    model_D_img = MultiscaleDiscriminator(input_nc=3).to(device)
    model_D_img = nn.DataParallel(model_D_img).cuda()
    if is_load_model_discriminator is True:
        # print('Loading model_D_t ...', end='')
        # model_D_t.load_state_dict(torch.load(args.model_transfer_path.replace('vqvae', 'vqvae_Dt')))
        # model_D_t.eval()
        # print('Done')
        #
        # print('Loading model_D_m ...', end='')
        # model_D_m.load_state_dict(torch.load(args.model_transfer_path.replace('vqvae', 'vqvae_Dm')))
        # model_D_m.eval()
        # print('Done')
        #
        # print('Loading model_D_b ...', end='')
        # model_D_b.load_state_dict(torch.load(args.model_transfer_path.replace('vqvae', 'vqvae_Db')))
        # model_D_b.eval()
        # print('Done')

        print('Loading model_D_img ...', end='')
        model_D_img.load_state_dict(torch.load(args.model_transfer_path.replace('vqvae', 'vqvae_Db')))
        model_D_img.eval()
        print('Done')
    else:
        print('model_discriminator Initialized.')
    # model_D_t = nn.DataParallel(model_D_t).cuda()
    # optimizer_D_t = optim.Adam(model_D_t.parameters(), lr=args.lr)
    # model_D_m = nn.DataParallel(model_D_m).cuda()
    # optimizer_D_m = optim.Adam(model_D_m.parameters(), lr=args.lr)
    # model_D_b = nn.DataParallel(model_D_b).cuda()
    # optimizer_D_b = optim.Adam(model_D_b.parameters(), lr=args.lr)
    optimizer_D_img = optim.Adam(model_D_img.parameters(), lr=args.lr)

    for i in range(args.epoch):
        viz.text(f'epoch: {i}', win='Epoch')
        train(epoch=i, loader=loader, model_transfer=model_transfer, model_img=model_img,
              model_cond=model_cond, model_D_img=model_D_img,
              optimizer=optimizer, optimizer_D_img=optimizer_D_img,
              scheduler=scheduler, device=device)
        torch.save(model_transfer.state_dict(), f'checkpoint/{EXPERIMENT_CODE}/vqvae_trans_{str(i + 1).zfill(3)}.pt')
        torch.save(model_img.state_dict(), f'checkpoint/{EXPERIMENT_CODE}/vqvae_img_{str(i + 1).zfill(3)}.pt')
        torch.save(model_cond.state_dict(), f'checkpoint/{EXPERIMENT_CODE}/vqvae_cond_{str(i + 1).zfill(3)}.pt')
        # torch.save(model_D_t.state_dict(), f'checkpoint/{EXPERIMENT_CODE}/vqvae_Dt_{str(i + 1).zfill(3)}.pt')
        # torch.save(model_D_m.state_dict(), f'checkpoint/{EXPERIMENT_CODE}/vqvae_Dm_{str(i + 1).zfill(3)}.pt')
        # torch.save(model_D_b.state_dict(), f'checkpoint/{EXPERIMENT_CODE}/vqvae_Db_{str(i + 1).zfill(3)}.pt')
        torch.save(model_D_img.state_dict(), f'checkpoint/{EXPERIMENT_CODE}/vqvae_Di_{str(i + 1).zfill(3)}.pt')
