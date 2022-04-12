# -*- coding: utf-8 -*-
import torch
import numpy as np


class BatchImagePrompts:

    def __init__(self, pil_images, borders, vae, device='cpu', crop_first=False):
        self.device = device
        img = torch.cat([self._preprocess_img(pil_image) for pil_image in pil_images], dim=0)
        img = img.to(self.device, dtype=torch.float32)
        self.image_prompts_idx, self.image_prompts = self._get_image_prompts(img, borders, vae, crop_first)

    def _preprocess_img(self, pil_img):
        img = torch.tensor(np.array(pil_img.convert('RGB')).transpose(2, 0, 1)) / 255.
        img = img.unsqueeze(0)
        img = (2 * img) - 1
        return img

    def _get_image_prompts(self, img, borders, vae, crop_first):
        if crop_first:
            bs, _, img_h, img_w = img.shape
            vqg_img_w, vqg_img_h = img_w // 8, img_h // 8
            vqg_img = torch.zeros((bs, vqg_img_h, vqg_img_w), dtype=torch.int32, device=img.device)
            if borders['down'] != 0:
                down_border = borders['down'] * 8
                _, _, [_, _, down_vqg_img] = vae.model.encode(img[:, :, -down_border:, :])
                vqg_img[:, -borders['down']:, :] = down_vqg_img
            if borders['right'] != 0:
                right_border = borders['right'] * 8
                _, _, [_, _, right_vqg_img] = vae.model.encode(img[:, :, :, -right_border:])
                vqg_img[:, :, -borders['right']:] = right_vqg_img
            if borders['left'] != 0:
                left_border = borders['left'] * 8
                _, _, [_, _, left_vqg_img] = vae.model.encode(img[:, :, :, :left_border])
                vqg_img[:, :, :borders['left']] = left_vqg_img
            if borders['up'] != 0:
                up_border = borders['up'] * 8
                _, _, [_, _, up_vqg_img] = vae.model.encode(img[:, :, :up_border, :])
                vqg_img[:, :borders['up'], :] = up_vqg_img
        else:
            _, _, [_, _, vqg_img] = vae.model.encode(img)

        bs, vqg_img_h, vqg_img_w = vqg_img.shape
        mask = torch.zeros(vqg_img_h, vqg_img_w)
        if borders['up'] != 0:
            mask[:borders['up'], :] = 1.
        if borders['down'] != 0:
            mask[-borders['down']:, :] = 1.
        if borders['right'] != 0:
            mask[:, -borders['right']:] = 1.
        if borders['left'] != 0:
            mask[:, :borders['left']] = 1.
        mask = mask.reshape(-1).bool()

        image_prompts = vqg_img.reshape((bs, -1))
        image_prompts_idx = np.arange(vqg_img_w * vqg_img_h)
        image_prompts_idx = set(image_prompts_idx[mask])

        return image_prompts_idx, image_prompts
