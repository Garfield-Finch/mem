"""
Use SPADE to decode
"""
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

# from utils.vqvae import VQVAE
from tz_networks_v12_spade import appVQVAE, VQVAE_SPADE, poseVQVAE, MultiscaleDiscriminator
# from vq_vae_2_pytorch.scheduler import CycleScheduler
from tz_dataloader_v04 import iPERLoader

from options.tz_train_options import TrainOptions
from models.networks.tz_generator import SPADEGenerator


def train(epoch, loader, dic_model, scheduler, device):
    loader = tqdm(loader)

    model_img = dic_model['model_img']
    optimizer_img = dic_model['optimizer_img']

    model_cond = dic_model['model_cond']
    optimizer_cond = dic_model['optimizer_cond']

    model_spade = dic_model['model_spade']
    optimizer_spade = dic_model['optimizer_spade']

    model_D = dic_model['model_D']
    optimizer_D = dic_model['optimizer_D']

    model_img.train()
    model_cond.train()
    model_spade.train()
    model_D.train()

    criterion = nn.MSELoss()

    # # utils to calculate loss GAN
    def _cal_gan_loss(tsr_in, key=True):
        return criterion(tsr_in, torch.ones(tsr_in.shape).cuda()) if key is True \
            else criterion(tsr_in, torch.zeros(tsr_in.shape).cuda())

    latent_loss_weight = 0.25
    weight_gan = 0.05
    sample_size = min(8, args.batch_size)

    mse_sum = 0
    mse_n = 0

    lst_loss = []
    lst_loss_G = []
    lst_loss_D = []

    # Obtain the average z
    z_ans = None
    for i, (img_0, pose_0, img, label) in enumerate(loader):
        if i < 8:
            img = img_0.to(device)
            img_out, _, _, _, img_seg = model_img(img)
            img_seg = img_seg.detach().clone()
            z_ans = img_seg if (z_ans is None) else z_ans + img_seg
        else:
            break
    z_ans = z_ans / 8

    for i, (img_0, pose_0, img, label) in enumerate(loader):
        img_0 = img_0.to(device)
        img = img.to(device)
        pose = label.to(device)

        pose_out, _, _, _, pose_seg = model_cond(pose)
        out = model_spade(pose_seg, z=z_ans)
        lst_D_img = model_D(img)
        lst_D_out = model_D(out)

        # Important, want to regres to appearance
        recon_loss = criterion(out, img)
        # latent_loss = latent_loss.mean()

        loss_G_img = _cal_gan_loss(lst_D_out[0][0], True)
        for j in range(1, len(lst_D_img)):
            loss_G_img += _cal_gan_loss(lst_D_out[j][0], True)
        loss_D_img = _cal_gan_loss(lst_D_out[0][0], False) + \
                     _cal_gan_loss(lst_D_img[0][0], True)
        for j in range(1, len(lst_D_out)):
            loss_D_img += _cal_gan_loss(lst_D_out[j][0], False)
            loss_D_img += _cal_gan_loss(lst_D_img[j][0], True)

        # loss = recon_loss + latent_loss_weight * latent_loss + weight_gan * loss_G_img
        loss = recon_loss #+ weight_gan * loss_G_img

        lst_loss.append(loss.item())
        lst_loss_G.append(loss_G_img.item())
        lst_loss_D.append(loss_D_img.item())

        if scheduler is not None:
            scheduler.step()

        optimizer_img.zero_grad()
        optimizer_cond.zero_grad()
        optimizer_D.zero_grad()
        loss.backward(retain_graph=True)
        optimizer_img.step()

        loss_D_img.backward()
        # optimizer_D.step()

        mse_sum += recon_loss.item() * img.shape[0]
        mse_n += img.shape[0]

        lr = optimizer_spade.param_groups[0]['lr']

        loader.set_description(
            (
                f'epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; '
                # f'latent: {latent_loss.item():.3f};'
                f' avg mse: {mse_sum / mse_n:.5f}; '
                f'lr: {lr:.5f}'
            )
        )

        if i % 100 == 0:
            # model.eval()

            # sample = img[:sample_size]

            # with torch.no_grad():
            #     out, _ = model(sample)
            # model.train()

            # img_show = (torch.cat([sample, out]).to('cpu').detach().numpy() * 0.5 + 0.5) * 255
            # viz.images(
            #     img_show,
            #     win='sample-out',
            #     nrow=sample_size,
            #     opts={
            #         'title': 'sample-out',
            #     }
            # )

            img_save_name = f'sample/{EXPERIMENT_CODE}/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png'
            utils.save_image(
                torch.cat([img_0[:sample_size], img[:sample_size],
                           out[:sample_size], pose[:sample_size], pose_out[:sample_size]], 0),
                img_save_name,
                nrow=sample_size,
                normalize=True,
                range=(-1, 1),
            )
            img_show = np.transpose(np.asarray(Image.open(img_save_name)), (2, 0, 1))
            viz.images(img_show, win='transfer', nrow=sample_size,
                       opts={'title': 'img_0-img_gt-img_out-pose_gt-pose_vq'})

        # # increase the sequence of saving model
        # if i % 200 == 0:
        #     torch.save(model.state_dict(), f'checkpoint/{EXPERIMENT_CODE}/vqvae_{str(epoch + 1).zfill(3)}.pt')

    for line_num, (lst, line_title) in enumerate(
            [(lst_loss, 'loss'),
             ([mse_sum / mse_n], 'MSE'),
             (lst_loss_D, 'loss_D'),
             (lst_loss_G, 'loss_G')
             ]):
        viz.line(Y=np.array([sum(lst) / len(lst)]), X=np.array([epoch]),
                 name=line_title,
                 win='loss',
                 opts=dict(title='loss', showlegend=True),
                 update=None if (epoch == 0 and line_num == 0) else 'append'
                 )


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    parser = TrainOptions().parse()


    # args = parser.parse_args()
    args = parser

    print(args)

    EXPERIMENT_CODE = 'as_86'
    if not os.path.exists(f'checkpoint/{EXPERIMENT_CODE}/'):
        print(f'New EXPERIMENT_CODE:{EXPERIMENT_CODE}, creating saving directories ...', end='')
        os.mkdir(f'checkpoint/{EXPERIMENT_CODE}/')
        os.mkdir(f'sample/{EXPERIMENT_CODE}/')
        print('Done')
    else:
        print('EXPERIMENT_CODE already exits.')

    viz = visdom.Visdom(server='10.10.10.100', port=33241, env=args.env)

    DESCRIPTION = """
        SPADE;Z=average img_0;Seg=pose;without Discriminator;
    """\
                  f'file: tz_main_v18_3_1_spade.py;\n '\
                  f'Hostname: {socket.gethostname()}; ' \
                  f'Experiment_Code: {EXPERIMENT_CODE};\n'

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

    _, _, loader = iPERLoader(data_root=args.path, batch=args.batch_size, transform=transform).data_load()

    # model = VQVAE_SPADE(embed_dim=128, parser=parser).to(device)
    # model = nn.DataParallel(model).cuda()


    model_img = poseVQVAE().to(device)
    model_img = nn.DataParallel(model_img).cuda()
    print('Loading Model...', end='')
    model_img.load_state_dict(torch.load('/p300/mem/mem_src/checkpoint/app/vqvae_264.pt'))
    model_img.eval()
    print('Complete !')
    optimizer_img = optim.Adam(model_img.parameters(), lr=args.lr)

    model_spade = SPADEGenerator(parser)
    model_spade = nn.DataParallel(model_spade).cuda()
    optimizer_spade = optim.Adam(model_spade.parameters(), lr=args.lr)

    scheduler = None

    model_cond = poseVQVAE().to(device)
    model_cond = nn.DataParallel(model_cond).cuda()
    print('Loading Model...', end='')
    model_cond.load_state_dict(torch.load('/p300/mem/mem_src/checkpoint/pose_04/vqvae_462.pt'))
    model_cond.eval()
    print('Complete !')
    optimizer_cond = optim.Adam(model_cond.parameters(), lr=args.lr)

    model_D = MultiscaleDiscriminator(input_nc=3).to(device)
    model_D = nn.DataParallel(model_D).cuda()
    optimizer_D = optim.Adam(model_D.parameters(), lr=args.lr)
    # if args.sched == 'cycle':
    #     scheduler = CycleScheduler(
    #         optimizer, args.lr, n_iter=len(loader) * args.epoch, momentum=None
    #     )

    dic_model = {'model_img': model_img, 'model_cond': model_cond, 'model_spade': model_spade,
                 # 'model_transfer': model_transfer,
                 'model_D': model_D,
                 'optimizer_img': optimizer_img, 'optimizer_cond': optimizer_cond,
                 'optimizer_spade': optimizer_spade,
                 # 'optimizer_transfer': optimizer_transfer
                 'optimizer_D': optimizer_D
                 }

    for i in range(args.epoch):
        viz.text(f'{DESCRIPTION} ##### Epoch: {i} #####', win='board')
        train(i, loader, dic_model, scheduler, device)
        torch.save(model_spade.state_dict(), f'checkpoint/{EXPERIMENT_CODE}/vqvae_{str(i + 1).zfill(3)}.pt')
