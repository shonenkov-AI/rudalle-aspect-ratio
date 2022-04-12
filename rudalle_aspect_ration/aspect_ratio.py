# -*- coding: utf-8 -*-
import torch
import numpy as np
import more_itertools
import transformers
from tqdm import tqdm
from PIL import Image
from rudalle import utils
from einops import rearrange


from .image_prompts import BatchImagePrompts


class RuDalleAspectRation:

    def __init__(self, dalle, vae, tokenizer, aspect_ratio=1.0, window=128, image_size=256, bs=4,
                 device='cuda', quite=False):
        """
        :param float aspect_ratio: w / h
        :param int window: size of context window  for h_generations
        :param int image_size: image size, that is used by the rudalle model
        :param int bs: batch size
        :param bool quite: on/off tqdm
        """
        self.device = device
        self.dalle = dalle
        self.vae = vae
        self.tokenizer = tokenizer
        #
        self.vocab_size = self.dalle.get_param('vocab_size')
        self.text_seq_length = self.dalle.get_param('text_seq_length')
        self.image_seq_length = self.dalle.get_param('image_seq_length')
        self.total_seq_length = self.dalle.get_param('total_seq_length')
        self.image_tokens_per_dim = self.dalle.get_param('image_tokens_per_dim')
        #
        self.window = window
        self.image_size = image_size
        self.patch_size = image_size // self.image_tokens_per_dim
        self.bs = bs
        self.quite = quite
        if aspect_ratio <= 1:
            self.is_vertical = True
            self.w = image_size
            self.h = int(round(image_size / aspect_ratio))
        else:
            self.is_vertical = False
            self.h = image_size
            self.w = int(round(image_size * aspect_ratio))
        self.aspect_ratio = aspect_ratio

    def generate_images(self, text, top_k=1024, top_p=0.975, images_num=4, seed=None):
        if seed is not None:
            utils.seed_everything(seed)

        if self.is_vertical:
            codebooks = self.generate_h_codebooks(text, top_k=top_k, top_p=top_p, images_num=images_num)
            pil_images = self.decode_h_codebooks(codebooks)
        else:
            codebooks, pil_images = [], []
            image_prompts = None
            while (len(pil_images)+1)*self.window <= self.w:
                if pil_images:
                    image_prompts = self.prepare_w_image_prompt(pil_images[-1])
                _pil_images, _codebooks = self.generate_w_codebooks(
                    text, top_k=top_k, top_p=top_p, images_num=images_num,
                    image_prompts=image_prompts, use_cache=True,
                )
                codebooks.append(_codebooks)
                pil_images.append(_pil_images)

            pil_images = self.decode_w_codebooks(codebooks)
            codebooks = torch.cat([_codebooks for _codebooks in codebooks])

        result_images = [pil_img.crop((0, 0, self.w, self.h)) for pil_img in pil_images]
        return codebooks, result_images

    def generate_w_codebooks(self, text, top_k, top_p, images_num, image_prompts=None, temperature=1.0, use_cache=True):
        text = text.lower().strip()
        input_ids = self.tokenizer.encode_text(text, text_seq_length=self.text_seq_length)
        codebooks, pil_images = [], []
        for chunk in more_itertools.chunked(range(images_num), self.bs):
            chunk_bs = len(chunk)
            with torch.no_grad():
                attention_mask = torch.tril(
                    torch.ones((chunk_bs, 1, self.total_seq_length, self.total_seq_length), device=self.device)
                )
                out = input_ids.unsqueeze(0).repeat(chunk_bs, 1).to(self.device)
                has_cache = False
                if image_prompts is not None:
                    prompts_idx, prompts = image_prompts.image_prompts_idx, image_prompts.image_prompts
                range_out = range(out.shape[1], self.total_seq_length)
                if not self.quite:
                    range_out = tqdm(range_out)
                for idx in range_out:
                    idx -= self.text_seq_length
                    if image_prompts is not None and idx in prompts_idx:
                        out = torch.cat((out, prompts[:, idx].unsqueeze(1)), dim=-1)
                    else:
                        logits, has_cache = self.dalle(out, attention_mask,
                                                       has_cache=has_cache, use_cache=use_cache, return_loss=False)
                        logits = logits[:, -1, self.vocab_size:]
                        logits /= temperature
                        filtered_logits = transformers.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                        probs = torch.nn.functional.softmax(filtered_logits, dim=-1)
                        sample = torch.multinomial(probs, 1)
                        out = torch.cat((out, sample), dim=-1)

                _codebooks = out[:, -self.image_seq_length:]
                images = self.vae.decode(_codebooks)
                pil_images += utils.torch_tensors_to_pil_list(images)
                codebooks.append(_codebooks)

        codebooks = torch.cat(codebooks)
        return pil_images, codebooks

    def prepare_w_image_prompt(self, pil_images):
        changed_pil_images = []
        for pil_img in pil_images:
            np_img = np.array(pil_img)
            np_img[:, :self.window, :] = np_img[:, self.window:2*self.window, :]
            pil_img = Image.fromarray(np_img)
            changed_pil_images.append(pil_img)
        borders = {'up': 0, 'left': self.window // self.patch_size, 'right': 0, 'down': 0}
        return BatchImagePrompts(changed_pil_images, borders, self.vae, self.device, crop_first=True)

    def generate_h_codebooks(self, text, top_k, top_p, images_num, temperature=1.0,  use_cache=True):
        h_out = int(round(self.image_tokens_per_dim / self.aspect_ratio))
        text = text.lower().strip()
        input_ids = self.tokenizer.encode_text(text, text_seq_length=self.text_seq_length)
        codebooks = []
        for chunk in more_itertools.chunked(range(images_num), self.bs):
            chunk_bs = len(chunk)
            with torch.no_grad():
                attention_mask = torch.tril(
                    torch.ones((chunk_bs, 1, self.total_seq_length, self.total_seq_length), device=self.device)
                )
                full_context = input_ids.unsqueeze(0).repeat(chunk_bs, 1).to(self.device)
                range_h_out = range(h_out)
                if not self.quite:
                    range_h_out = tqdm(range_h_out)
                for i in range_h_out:
                    j = (self.image_tokens_per_dim * i) // h_out
                    out = torch.cat((
                        input_ids.unsqueeze(0).repeat(chunk_bs, 1).to(self.device),
                        full_context[:, self.text_seq_length:][:, -j * self.image_tokens_per_dim:]
                    ), dim=-1)

                    has_cache = False
                    for _ in range(self.image_tokens_per_dim):
                        logits, has_cache = self.dalle(out, attention_mask,
                                                       has_cache=has_cache, use_cache=use_cache, return_loss=False)
                        logits = logits[:, -1, self.vocab_size:]
                        logits /= temperature
                        filtered_logits = transformers.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                        probs = torch.nn.functional.softmax(filtered_logits, dim=-1)
                        sample = torch.multinomial(probs, 1)
                        out = torch.cat((out, sample), dim=-1)
                        full_context = torch.cat((full_context, sample), dim=-1)
            codebooks.append(full_context[:, self.text_seq_length:])

        return torch.cat(codebooks)

    def decode_h_codebooks(self, codebooks):
        with torch.no_grad():
            one_hot_indices = torch.nn.functional.one_hot(codebooks, num_classes=self.vae.num_tokens).float()
            z = (one_hot_indices @ self.vae.model.quantize.embed.weight)
            z = rearrange(z, 'b (h w) c -> b c h w', w=self.image_tokens_per_dim)
            img = self.vae.model.decode(z)
            img = (img.clamp(-1., 1.) + 1) * 0.5
        return utils.torch_tensors_to_pil_list(img)

    def decode_w_codebooks(self, codebooks):
        with torch.no_grad():
            final_z = []
            for i, img_seq in enumerate(codebooks):
                one_hot_indices = torch.nn.functional.one_hot(img_seq, num_classes=self.vae.num_tokens).float()
                z = (one_hot_indices @ self.vae.model.quantize.embed.weight)
                z = rearrange(z, 'b (h w) c -> b c h w', h=self.image_tokens_per_dim)
                if i < len(codebooks)-1:
                    final_z.append(z[:, :, :, :self.window//self.patch_size])
                else:
                    final_z.append(z)
            z = torch.cat(final_z, -1)
            img = self.vae.model.decode(z)
            img = (img.clamp(-1., 1.) + 1) * 0.5
        return utils.torch_tensors_to_pil_list(img)
