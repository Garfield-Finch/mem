import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import PatchDiscriminator


class GlobalDiscriminator(nn.Module):
    def __init__(self, input_nc, global_nc=None, ndf=32, n_layers=3, max_nf_mult=8, norm_type='batch', use_sigmoid=False):
        super(GlobalDiscriminator, self).__init__()

        if global_nc is not None:
            self.global_model = PatchDiscriminator(global_nc, ndf=ndf, n_layers=n_layers, max_nf_mult=max_nf_mult,
                                                   norm_type=norm_type, use_sigmoid=use_sigmoid)
        self.local_model = PatchDiscriminator(input_nc, ndf=ndf, n_layers=n_layers - 1, max_nf_mult=max_nf_mult,
                                              norm_type=norm_type, use_sigmoid=use_sigmoid)

    def forward(self, global_x, local_x, body_rects, head_rects, get_avg=True):
        if global_x is None:
            local_outs = self.local_model(local_x)
            outs = [local_outs]
        else:
            global_outs = self.global_model(global_x)
            local_outs = self.local_model(local_x)
            outs = [global_outs, local_outs]

        if get_avg:
            return outs, self.reduce_tensor(outs)
        else:
            return outs

    @staticmethod
    def reduce_tensor(outs):
        with torch.no_grad():
            avg = 0.0
            num = len(outs)

            for out in outs:
                avg += torch.mean(out)

            avg /= num
            return avg

    @staticmethod
    def crop_body(imgs, rects):
        """
        :param imgs: (N, C, H, W)
        :return:
        """
        bs, _, ori_h, ori_w = imgs.shape
        head_imgs = []

        for i in range(bs):
            min_x, max_x, min_y, max_y = rects[i].detach()
            if (min_x != max_x) and (min_y != max_y):
                head = imgs[i:i+1, :, min_y:max_y, min_x:max_x]  # (1, c, h', w')
                head = F.interpolate(head, size=(ori_h // 2, ori_w // 2), mode='bilinear', align_corners=True)
                head_imgs.append(head)

        head_imgs = torch.cat(head_imgs, dim=0)

        return head_imgs


class GlobalLocalDiscriminator(nn.Module):
    def __init__(self, input_nc, global_nc=None, ndf=32, n_layers=3, max_nf_mult=8, norm_type='batch', use_sigmoid=False):
        super(GlobalLocalDiscriminator, self).__init__()

        if global_nc is None:
            global_nc = input_nc
        self.global_model = PatchDiscriminator(global_nc, ndf=ndf, n_layers=n_layers, max_nf_mult=max_nf_mult,
                                               norm_type=norm_type, use_sigmoid=use_sigmoid)
        self.local_model = PatchDiscriminator(input_nc, ndf=ndf, n_layers=n_layers - 1, max_nf_mult=max_nf_mult,
                                              norm_type=norm_type, use_sigmoid=use_sigmoid)

    def forward(self, global_x, local_x, body_rects, head_rects, get_avg=True, bs=1):
        if global_x is None:
            global_x = local_x

        glocal_outs = self.global_model(global_x)
        crop_imgs = self.crop_body(local_x, body_rects)
        outs = [glocal_outs]

        if len(crop_imgs) != 0:
            local_outs = self.local_model(crop_imgs)
            outs.append(local_outs)

        if get_avg:
            return outs, self.reduce_tensor(outs)
        else:
            return outs

    @staticmethod
    def reduce_tensor(outs):
        with torch.no_grad():
            avg = 0.0
            num = len(outs)

            for out in outs:
                avg += torch.mean(out)

            avg /= num
            return avg

    @staticmethod
    def crop_body(imgs, rects):
        """
        :param imgs: (N, C, H, W)
        :return:
        """
        bs, _, ori_h, ori_w = imgs.shape
        head_imgs = []

        for i in range(bs):
            min_x, max_x, min_y, max_y = rects[i].detach()
            if (min_x != max_x) and (min_y != max_y):
                head = imgs[i:i+1, :, min_y:max_y, min_x:max_x]  # (1, c, h', w')
                head = F.interpolate(head, size=(ori_h // 2, ori_w // 2), mode='bilinear', align_corners=True)
                head_imgs.append(head)

        if len(head_imgs) != 0:
            head_imgs = torch.cat(head_imgs, dim=0)

        return head_imgs


class GlobalBodyHeadDiscriminator(nn.Module):
    def __init__(self, input_nc, global_nc=None, ndf=32, n_layers=3, max_nf_mult=8, norm_type='batch', use_sigmoid=False):
        super(GlobalBodyHeadDiscriminator, self).__init__()

        if global_nc is None:
            global_nc = input_nc

        self.global_model = PatchDiscriminator(global_nc, ndf=ndf, n_layers=n_layers, max_nf_mult=max_nf_mult,
                                               norm_type=norm_type, use_sigmoid=use_sigmoid)
        self.body_model = PatchDiscriminator(input_nc, ndf=ndf, n_layers=n_layers, max_nf_mult=max_nf_mult,
                                             norm_type=norm_type, use_sigmoid=use_sigmoid)
        self.head_model = PatchDiscriminator(input_nc, ndf=ndf, n_layers=n_layers, max_nf_mult=max_nf_mult,
                                             norm_type=norm_type, use_sigmoid=use_sigmoid)

    def forward(self, global_x, local_x, body_rects, head_rects, get_avg=True, bs=1):
        if global_x is None:
            global_x = local_x

        body_imgs = self.crop_img(local_x, body_rects, fact=1)
        head_imgs = self.crop_img(local_x, head_rects, fact=4)

        global_outs = self.global_model(global_x)
        outs = [global_outs]

        if len(body_imgs) != 0:
            body_outs = self.body_model(body_imgs)
            outs.append(body_outs)

        if len(head_imgs) != 0:
            head_outs = self.head_model(head_imgs)
            outs.append(head_outs)

        if get_avg:
            return outs, self.reduce_tensor(outs)
        else:
            return outs

    @staticmethod
    def crop_img(imgs, rects, fact=2):
        """
        Args:
            imgs:
            rects:
            fact:

        Returns:

        """
        bs, _, ori_h, ori_w = imgs.shape
        crops = []
        for i in range(bs):
            min_x, max_x, min_y, max_y = rects[i].detach()
            if (min_x != max_x) and (min_y != max_y):
                _crop = imgs[i:i+1, :, min_y:max_y, min_x:max_x]  # (1, c, h', w')
                _crop = F.interpolate(_crop, size=(ori_h // fact, ori_w // fact), mode='bilinear', align_corners=True)
                crops.append(_crop)

        if len(crops) != 0:
            crops = torch.cat(crops, dim=0)

        return crops

    @staticmethod
    def reduce_tensor(outs):
        with torch.no_grad():
            avg = 0.0
            num = len(outs)

            for out in outs:
                avg += torch.mean(out)

            avg /= num
            return avg


class MultiScaleDiscriminator(nn.Module):

    def __init__(self, global_nc, input_nc, ndf=32, n_layers=3, max_nf_mult=8, norm_type='batch', use_sigmoid=False):
        super(MultiScaleDiscriminator, self).__init__()

        # low-res to high-res
        scale_models = nn.ModuleList()
        # scale_n_layers = [1, 1, 1, 1, 1]
        n_scales = 3
        for i in range(n_scales):
            # n_layers = scale_n_layers[i]
            model = PatchDiscriminator(input_nc, ndf=ndf, n_layers=n_layers, max_nf_mult=max_nf_mult,
                                       norm_type=norm_type, use_sigmoid=use_sigmoid)
            scale_models.append(model)

        self.n_scales = n_scales
        self.scale_models = scale_models

        if global_nc is not None:
            self.global_model = PatchDiscriminator(global_nc, ndf=ndf, n_layers=n_layers, max_nf_mult=max_nf_mult,
                                                   norm_type=norm_type, use_sigmoid=use_sigmoid)
        else:
            self.global_model = None

    def forward(self, global_x, local_x, body_rects, head_rects, get_avg=True):
        scale_outs = []

        if self.global_model is not None:
            outs = self.global_model(global_x)
            scale_outs.append(outs)

        _, _, ori_h, ori_w = local_x.shape
        x = local_x
        for i in range(0, self.n_scales):
            outs = self.scale_models[i](x)

            if i < self.n_scales - 1:
                fact = 2 ** (i + 1)
                x = F.interpolate(local_x, size=(ori_h // fact, ori_w // fact), mode='bilinear', align_corners=True)

            scale_outs.append(outs)

        if get_avg:
            return scale_outs, self.reduce_tensor(scale_outs)
        else:
            return scale_outs

    @staticmethod
    def reduce_tensor(outs):
        with torch.no_grad():
            avg = 0.0
            num = len(outs)

            for out in outs:
                avg += torch.mean(out)

            avg /= num
            return avg


class HeadDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=32, n_layers=3, max_nf_mult=8, norm_type='batch', use_sigmoid=False):
        super(HeadDiscriminator, self).__init__()

        # if global_nc is None:
        #     global_nc = input_nc
        #
        # self.global_model = PatchDiscriminator(global_nc, ndf=ndf, n_layers=n_layers, max_nf_mult=max_nf_mult,
        #                                        norm_type=norm_type, use_sigmoid=use_sigmoid)
        # self.body_model = PatchDiscriminator(input_nc, ndf=ndf, n_layers=n_layers, max_nf_mult=max_nf_mult,
        #                                      norm_type=norm_type, use_sigmoid=use_sigmoid)
        self.head_model = PatchDiscriminator(input_nc, ndf=ndf, n_layers=n_layers, max_nf_mult=max_nf_mult,
                                             norm_type=norm_type, use_sigmoid=use_sigmoid)

    def forward(self, head_imgs, get_avg=True, bs=1):
        # if global_x is None:
        #     global_x = local_x

        # body_imgs = self.crop_img(local_x, body_rects, fact=1)
        # head_imgs = self.crop_img(local_x, head_rects, fact=4)

        # global_outs = self.global_model(global_x)
        # outs = [global_outs]
        outs = []

        # if len(body_imgs) != 0:
        #     body_outs = self.body_model(body_imgs)
        #     outs.append(body_outs)

        if len(head_imgs) != 0:
            head_outs = self.head_model(head_imgs)
            outs.append(head_outs)

        if get_avg:
            return outs, self.reduce_tensor(outs)
        else:
            return outs

    @staticmethod
    def crop_img(imgs, rects, fact=2):
        """
        Args:
            imgs:
            rects:
            fact:

        Returns:

        """
        bs, _, ori_h, ori_w = imgs.shape
        crops = []
        for i in range(bs):
            min_x, max_x, min_y, max_y = rects[i].detach()
            if (min_x != max_x) and (min_y != max_y):
                _crop = imgs[i:i+1, :, min_y:max_y, min_x:max_x]  # (1, c, h', w')
                _crop = F.interpolate(_crop, size=(ori_h // fact, ori_w // fact), mode='bilinear', align_corners=True)
                crops.append(_crop)

        if len(crops) != 0:
            crops = torch.cat(crops, dim=0)

        return crops

    @staticmethod
    def reduce_tensor(outs):
        with torch.no_grad():
            avg = 0.0
            num = len(outs)

            for out in outs:
                avg += torch.mean(out)

            avg /= num
            return avg

