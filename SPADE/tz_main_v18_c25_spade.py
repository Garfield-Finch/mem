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
from tz_networks_v12_spade import appVQVAE, VQVAE_SPADE, poseVQVAE
# from vq_vae_2_pytorch.scheduler import CycleScheduler
from tz_dataloader_v03 import iPERLoader

from options.tz_train_options import TrainOptions


def train(epoch, loader, dic_model, scheduler, device):
    loader = tqdm(loader)

    model_img = dic_model['model_img']
    optimizer_img = dic_model['optimizer_img']
    model_cond = dic_model['model_cond']
    optimizer_cond = dic_model['optimizer_cond']

    model_img.train()
    model_cond.train()

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = min(8, args.batch_size)

    mse_sum = 0
    mse_n = 0

    lst_loss = []
    for i, (img, label) in enumerate(loader):
        img = img.to(device)
        pose = label.to(device)

        pose_out, _, _, _, pose_seg = model_cond(pose)
        out, latent_loss = model_img(img, pose_seg)

        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss

        lst_loss.append(loss.item())

        if scheduler is not None:
            scheduler.step()

        optimizer_img.zero_grad()
        loss.backward()
        optimizer_img.step()

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
            # model.eval()

            # sample = img[:sample_size]

            # with torch.no_grad():
            #     out, _ = model(sample)

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
                torch.cat([pose[:sample_size], pose_out[:sample_size],
                           out[:sample_size], img[:sample_size]], 0),
                img_save_name,
                nrow=sample_size,
                normalize=True,
                range=(-1, 1),
            )
            img_show = np.transpose(np.asarray(Image.open(img_save_name)), (2, 0, 1))
            viz.images(img_show, win='transfer', nrow=sample_size, opts={'title': 'gt-sample'})

            model.train()

        # # increase the sequence of saving model
        # if i % 200 == 0:
        #     torch.save(model.state_dict(), f'checkpoint/{EXPERIMENT_CODE}/vqvae_{str(epoch + 1).zfill(3)}.pt')

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
    # parser = argparse.ArgumentParser()
    parser = TrainOptions().parse()


    # args = parser.parse_args()
    args = parser

    print(args)

    EXPERIMENT_CODE = 'as_75'
    if not os.path.exists(f'checkpoint/{EXPERIMENT_CODE}/'):
        print(f'New EXPERIMENT_CODE:{EXPERIMENT_CODE}, creating saving directories ...', end='')
        os.mkdir(f'checkpoint/{EXPERIMENT_CODE}/')
        os.mkdir(f'sample/{EXPERIMENT_CODE}/')
        print('Done')
    else:
        print('EXPERIMENT_CODE already exits.')

    viz = visdom.Visdom(server='10.10.10.100', port=33241, env=args.env)

    DESCRIPTION = """
        SPADE;Z=app;Seg=pose;TARGET;
    """\
                  f'file: tz_main_v18_c5_spade.py;\n '\
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

    model = VQVAE_SPADE(embed_dim=128, parser=parser).to(device)
    model = nn.DataParallel(model).cuda()
    # print('Loading Model...', end='')
    # model.load_state_dict(torch.load('/p300/mem/mem_src/SPADE/checkpoint/app_v04/vqvae_089.pt'))
    # model.eval()
    # print('Complete !')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None

    model_cond = poseVQVAE().to(device)
    model_cond = nn.DataParallel(model_cond).cuda()
    print('Loading Model...', end='')
    model_cond.load_state_dict(torch.load('/p300/mem/mem_src/checkpoint/pose_04/vqvae_462.pt'))
    model_cond.eval()
    print('Complete !')
    optimizer_cond = optim.Adam(model_cond.parameters(), lr=args.lr)
    # if args.sched == 'cycle':
    #     scheduler = CycleScheduler(
    #         optimizer, args.lr, n_iter=len(loader) * args.epoch, momentum=None
    #     )

    dic_model = {'model_img': model, 'model_cond': model_cond,
                 # 'model_transfer': model_transfer,
                 'optimizer_img': optimizer, 'optimizer_cond': optimizer_cond,
                 # 'optimizer_transfer': optimizer_transfer
                 }

    for i in range(args.epoch):
        viz.text(f'{DESCRIPTION} ##### Epoch: {i} #####', win='board')
        train(i, loader, dic_model, scheduler, device)
        torch.save(model.state_dict(), f'checkpoint/{EXPERIMENT_CODE}/vqvae_{str(i + 1).zfill(3)}.pt')
