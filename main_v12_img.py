import argparse
import os
import visdom
from tqdm import tqdm
import socket
import numpy as np

import torch
from torch import nn, optim
from torchvision import datasets, transforms, utils

from vq_vae_2_pytorch.scheduler import CycleScheduler
from utils.dataloader_v03 import iPERLoader
from utils.networks_v06 import VQVAE


def train(epoch, loader, model, optimizer, scheduler, device):
    loader = tqdm(loader)

    #############################
    # Hyper parameters
    #############################
    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = 6

    mse_sum = 0
    mse_n = 0

    lst_loss_recon = []
    lst_loss_latent = []
    lst_loss = []

    for i, (img, label) in enumerate(loader):
        model.zero_grad()

        img = img.to(device)

        out, latent_loss, _, _, _ = model(img)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        # for viz
        lst_loss_recon.append(recon_loss.item())
        lst_loss_latent.append(latent_loss.item())
        lst_loss.append(loss.item())

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

        #########################
        # Evaluation
        #########################
        if i % 100 == 0:
            # model.eval()

            # sample = img[:sample_size]

            # with torch.no_grad():
            #     out, _ = model(sample)

            img_show = torch.cat([img[:sample_size], out[:sample_size]], 0)
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
            viz.images(img_show, win='img', nrow=sample_size, opts={'title': 'gt-img_out'})

            model.train()

        # increase the sequence of saving model
        if i % 200 == 0:
            torch.save(model.state_dict(), f'checkpoint/{EXPERIMENT_CODE}/vqvae_{str(epoch + 1).zfill(3)}.pt')

    for line_num, (lst, line_title) in enumerate(
            [(lst_loss_recon, 'loss_recon'),
             (lst_loss_latent, 'loss_latent'),
             (lst_loss, 'loss'),
             ]):
        viz.line(Y=np.array([sum(lst) / len(lst)]), X=np.array([epoch]),
                 name=line_title,
                 win='loss',
                 opts=dict(title='loss', showlegend=True),
                 update=None if (epoch == 0 and line_num == 0) else 'append'
                 )
    viz.line(Y=np.array([mse_sum / mse_n]), X=np.array([epoch]),
             name='avg_mse',
             win='avg_mse',
             opts=dict(title='avg_mse', showlegend=True),
             update=None if (epoch == 0) else 'append'
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

    args = parser.parse_args()

    print(args)

    ##############################
    # Dash Board
    ##############################
    BATCH_SIZE = 32
    EXPERIMENT_CODE = 'as_15_mem3'
    if not os.path.exists(f'checkpoint/{EXPERIMENT_CODE}/'):
        print(f'New EXPERIMENT_CODE:{EXPERIMENT_CODE}, creating saving directories ...', end='')
        os.mkdir(f'checkpoint/{EXPERIMENT_CODE}/')
        os.mkdir(f'sample/{EXPERIMENT_CODE}/')
        print('Done')
    else:
        print('EXPERIMENT_CODE already exits.')
    DESCRIPTION = """
        Appearance VQ-VAE with 3 memory.
        """

    viz = visdom.Visdom(server='10.10.10.100', port=33241, env=args.env)
    viz.text(f'{DESCRIPTION}'
             f'Hostname: {socket.gethostname()}; '
             f'file: main_v12_img.py;\n '
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
    # _, _, loader = iPERLoader(data_root=args.path, batch=BATCH_SIZE, transform=transform).data_load()
    _, loader, _ = iPERLoader(data_root=args.path, batch=BATCH_SIZE, transform=transform).data_load()

    # model
    model = VQVAE().to(device)
    model = nn.DataParallel(model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, args.lr, n_iter=len(loader) * args.epoch, momentum=None
        )

    print('Loading Model...', end='')
    model.load_state_dict(torch.load('/p300/mem/mem_src/checkpoint/as_15_mem3/vqvae_061.pt'))
    model.eval()
    print('Complete !')

    for i in range(args.epoch):
        viz.text(f'epoch: {i}', win='Epoch')
        train(i, loader, model, optimizer, scheduler, device)
        torch.save(model.state_dict(), f'checkpoint/{EXPERIMENT_CODE}/vqvae_{str(i + 1).zfill(3)}.pt')
