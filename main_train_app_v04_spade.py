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
from utils.networks_v11 import appVQVAE
from vq_vae_2_pytorch.scheduler import CycleScheduler
from utils.dataloader_v03 import iPERLoader

from SPADE.options.train_options import TrainOptions


def train(epoch, loader, model, optimizer, scheduler, device):
    loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = 8

    mse_sum = 0
    mse_n = 0

    lst_loss = []
    for i, (img, label) in enumerate(loader):
        model.zero_grad()

        img = img.to(device)

        out, latent_loss = model(img)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss

        lst_loss.append(loss.item())

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
            torch.save(model.state_dict(), f'checkpoint/{EXPERIMENT_CODE}/vqvae_{str(epoch + 1).zfill(3)}.pt')

    for line_num, (lst, line_title) in enumerate(
            [(lst_loss, 'loss'),
             ([mse_sum / mse_n], 'MSE')
             ]):
        viz.line(Y=np.array([sum(lst) / len(lst)]), X=np.array([epoch]),
                 name=line_title,
                 win='loss',
                 opts=dict(title='loss', showlegend=True),
                 update=None if (epoch == 0 and line_num == 0) else 'append'
                 )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=560)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--sched', type=str)
    parser.add_argument('--path', type=str, default='/p300/dataset/iPER/')
    parser.add_argument('--env', type=str, default='main')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=8)

    args = parser.parse_args()

    print(args)

    EXPERIMENT_CODE = 'app_v04'
    if not os.path.exists(f'checkpoint/{EXPERIMENT_CODE}/'):
        print(f'New EXPERIMENT_CODE:{EXPERIMENT_CODE}, creating saving directories ...', end='')
        os.mkdir(f'checkpoint/{EXPERIMENT_CODE}/')
        os.mkdir(f'sample/{EXPERIMENT_CODE}/')
        print('Done')
    else:
        print('EXPERIMENT_CODE already exits.')

    viz = visdom.Visdom(server='10.10.10.100', port=33241, env=args.env)

    DESCRIPTION = """
        SPADE
    """\
                  f'file: main_train_app_v04.py;\n '\
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

    _, loader, _ = iPERLoader(data_root=args.path, batch=args.batch_size, transform=transform).data_load()

    model = appVQVAE(embed_dim=128).to(device)
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
        viz.text(f'{DESCRIPTION} ##### Epoch: {i} #####', win='board')
        train(i, loader, model, optimizer, scheduler, device)
        torch.save(model.state_dict(), f'checkpoint/{EXPERIMENT_CODE}/vqvae_{str(i + 1).zfill(3)}.pt')
