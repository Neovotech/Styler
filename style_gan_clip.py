import sys
sys.path.append('./CLIP')
sys.path.append('./stylegan3')

import io
import os, time, glob
import pickle
import shutil
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import requests
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import clip
import unicodedata
import re
from tqdm.notebook import tqdm
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from IPython.display import display
from einops import rearrange
import cv2
# from google.colab import files

device = torch.device('cuda:0')
print('Using device:', device, file=sys.stderr)

#@markdown #**Define necessary functions** 🛠️

def main(concatenated_input, number_of_clothes):
    img_list = []
    def fetch(url_or_path):
        if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
            r = requests.get(url_or_path)
            r.raise_for_status()
            fd = io.BytesIO()
            fd.write(r.content)
            fd.seek(0)
            return fd
        return open(url_or_path, 'rb')

    def fetch_model(url_or_path):
        if os.path.exists(basename):
            return basename
        else:
            if "drive.google" not in url_or_path:
                # !wget -c '{url_or_path}'
                basename = url_or_path
            else:
                # path_id = url_or_path.split("id=")[-1]
                # !gdown --id '{path_id}'
                basename = url_or_path
            return basename

    def slugify(value, allow_unicode=False):
        """
        Taken from https://github.com/django/django/blob/master/django/utils/text.py
        Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
        dashes to single dashes. Remove characters that aren't alphanumerics,
        underscores, or hyphens. Convert to lowercase. Also strip leading and
        trailing whitespace, dashes, and underscores.
        """
        value = str(value)
        if allow_unicode:
            value = unicodedata.normalize('NFKC', value)
        else:
            value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
        value = re.sub(r'[^\w\s-]', '', value.lower())
        return re.sub(r'[-\s]+', '-', value).strip('-_')

    def norm1(prompt):
        "Normalize to the unit sphere."
        return prompt / prompt.square().sum(dim=-1,keepdim=True).sqrt()

    def spherical_dist_loss(x, y):
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

    def prompts_dist_loss(x, targets, loss):
        if len(targets) == 1:
            return loss(x, targets[0])
        distances = [loss(x, target) for target in targets]
        return torch.stack(distances, dim=-1).sum(dim=-1)  

    class MakeCutouts(torch.nn.Module):
        def __init__(self, cut_size, cutn, cut_pow=1.):
            super().__init__()
            self.cut_size = cut_size
            self.cutn = cutn
            self.cut_pow = cut_pow

        def forward(self, input):
            sideY, sideX = input.shape[2:4]
            max_size = min(sideX, sideY)
            min_size = min(sideX, sideY, self.cut_size)
            cutouts = []
            for _ in range(self.cutn):
                size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
                cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
            return torch.cat(cutouts)

    make_cutouts = MakeCutouts(224, 32, 0.5)

    def embed_image(image):
        n = image.shape[0]
        cutouts = make_cutouts(image)
        embeds = clip_model.embed_cutout(cutouts)
        embeds = rearrange(embeds, '(cc n) c -> cc n c', n=n)
        return embeds

    def embed_url(url):
        image = Image.open(fetch(url)).convert('RGB')
        return embed_image(TF.to_tensor(image).to(device).unsqueeze(0)).mean(0).squeeze(0)

    class CLIP(object):
        def __init__(self):
            clip_model = "ViT-B/32"
            self.model, _ = clip.load(clip_model)
            self.model = self.model.requires_grad_(False)
            self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])

        @torch.no_grad()
        def embed_text(self, prompt):
            "Normalized clip text embedding."
            return norm1(self.model.encode_text(clip.tokenize(prompt).to(device)).float())

        def embed_cutout(self, image):
            "Normalized clip image embedding."
            return norm1(self.model.encode_image(self.normalize(image)))

    clip_model = CLIP()

    with open(fetch_model("./styler"), 'rb') as fp:
        G = pickle.load(fp)['G_ema'].to(device)

    zs = torch.randn([10000, G.mapping.z_dim], device=device)
    w_stds = G.mapping(zs, None).std(0)

    texts = concatenated_input #"male black suit white shirt upper body"#@param {type:"string"}
    steps = 61 #@param {type:"number"}
    seed = 61570 #@param {type:"number"}

    #@markdown ---

    if seed == -1:
        seed = np.random.randint(0,9e9)
        print(f"Your random seed is: {seed}")

    texts = [frase.strip() for frase in texts.split("|") if frase]

    targets = [clip_model.embed_text(text) for text in texts]

    # do the run

    tf = Compose([
        Resize(224),
        lambda x: torch.clamp((x+1)/2,min=0,max=1),
        ])

    def run(timestring):
        torch.manual_seed(seed)

        # Init
        # Sample 32 inits and choose the one closest to prompt

        with torch.no_grad():
            qs = []
            losses = []
            for _ in range(8):
                q = (G.mapping(torch.randn([4,G.mapping.z_dim], device=device), None, truncation_psi=1.0) - G.mapping.w_avg) / w_stds
                images = G.synthesis(q * w_stds + G.mapping.w_avg)
                embeds = embed_image(images.add(1).div(2))
                loss = prompts_dist_loss(embeds, targets, spherical_dist_loss).mean(0)
                i = torch.argmin(loss)
                qs.append(q[i])
                losses.append(loss[i])
            qs = torch.stack(qs)
            losses = torch.stack(losses)
            i = torch.argmin(losses)
            q = qs[i].unsqueeze(0).requires_grad_()

        # Sampling loop
        q_ema = q
        opt = torch.optim.AdamW([q], lr=0.03, betas=(0.0,0.999))
        loop = tqdm(range(steps))
        for i in loop:
            opt.zero_grad()
            w = q * w_stds
            image = G.synthesis(w + G.mapping.w_avg, noise_mode='const')
            embed = embed_image(image.add(1).div(2))
            loss = prompts_dist_loss(embed, targets, spherical_dist_loss).mean()
            loss.backward()
            opt.step()
            loop.set_postfix(loss=loss.item(), q_magnitude=q.std().item())

            q_ema = q_ema * 0.9 + q * 0.1
            image = G.synthesis(q_ema * w_stds + G.mapping.w_avg, noise_mode='const')

            if i % 5 == 0:
                img = TF.to_pil_image(tf(image)[0])
                np_img = np.asarray(img)
                img_list.append(np_img)
                print(f"Input Text: {texts} | Image {i}/{steps} | Current loss: {loss}")
            pil_image = TF.to_pil_image(image[0].add(1).div(2).clamp(0,1))
            os.makedirs(f'samples/{timestring}', exist_ok=True)
            pil_image.save(f'samples/{timestring}/{i:04}.jpg')

    try:
        timestring = time.strftime('%Y%m%d%H%M%S')
        run(timestring)
    except KeyboardInterrupt:
        pass
    
    return img_list[4:]
