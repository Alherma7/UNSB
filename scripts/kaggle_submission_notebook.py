"""Self-contained Kaggle Notebook script for the "I'm Something of a Painter
Myself" (gan-getting-started) competition -- a *code competition*: grading
requires this to run inside a committed Kaggle Notebook and produce
/kaggle/working/images.zip as the notebook's output ("Save & Run All
(Commit)"). A locally-generated zip cannot be submitted directly.

Paste this whole file into a single Kaggle Notebook cell (or split into
cells at the section markers below). It has NO dependency on the UNSB repo
or its options/argparse machinery -- only torch/torchvision/PIL/numpy,
which are preinstalled in Kaggle's GPU notebook image.

Setup required before running:
  1. Train locally with UNSB/train.py --mode sb (see the repo's plan/README),
     pick the best checkpoint via UNSB/scripts/eval_checkpoints.py.
  2. Upload just that checkpoint (`<epoch>_net_G.pth` -- F/D/E are
     training-only and not needed here) as a private Kaggle Dataset, e.g.
     named "monet-sb-generator-checkpoint".
  3. In the Notebook's "Add data" panel, attach both that dataset and the
     competition data (gan-getting-started).
  4. Enable a GPU accelerator in Notebook settings.
  5. Update CHECKPOINT_PATH below to match the uploaded file's path.
  6. First do a plain "Run All" with LIMIT set to ~20 to confirm the
     checkpoint loads and inference runs in this environment. Only then set
     LIMIT = None and "Save & Run All (Commit)" for the real, graded run.

Architecture ported below (verbatim logic, only .cuda() generalized to
.to(device)) from the local UNSB repo:
    models/ncsn_networks.py  -- PixelNorm, AdaptiveLayer, ResnetBlock_cond,
                                 get_timestep_embedding, Downsample, Upsample,
                                 get_filter, get_pad_layer, ResnetGenerator_ncsn
    models/networks.py       -- get_norm_layer
    models/sb_model.py:234-264 -- the test-time sampling loop (sample_fake_B)
    util/util.py             -- tensor2im (inlined as tensor_to_uint8)
"""
import functools
import glob
import math
import os
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

# ============================== Configuration ==============================
CHECKPOINT_PATH = '/kaggle/input/monet-sb-generator-checkpoint/latest_net_G.pth'
PHOTO_DIR = '/kaggle/input/gan-getting-started/photo_jpg'
OUT_DIR = '/kaggle/working/images'
ZIP_PATH = '/kaggle/working/images.zip'

NGF = 64          # must match the --ngf the checkpoint was trained with
N_MLP = 3         # must match the --n_mlp the checkpoint was trained with
NUM_TIMESTEPS = 5  # must match the --num_timesteps the checkpoint was trained with
TAU = 0.01        # must match the --tau the checkpoint was trained with
SEED = 42

LIMIT = 20  # set to None for the real, graded run over the full photo set
JPEG_QUALITY = 95

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ======================= Ported model architecture ========================
# From models/ncsn_networks.py


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class AdaptiveLayer(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()
        self.style_net = nn.Linear(style_dim, in_channel * 2)
        self.style_net.bias.data[:in_channel] = 1
        self.style_net.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style_net(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)
        return gamma * input + beta


class ResnetBlock_cond(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, temb_dim, z_dim):
        super(ResnetBlock_cond, self).__init__()
        self.conv_block, self.adaptive, self.conv_fin = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias, temb_dim, z_dim)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, temb_dim, z_dim):
        self.conv_block = nn.ModuleList()
        self.conv_fin = nn.ModuleList()
        p = 0
        if padding_type == 'reflect':
            self.conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            self.conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        self.conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]
        self.adaptive = AdaptiveLayer(dim, z_dim)
        self.conv_fin += [nn.ReLU(True)]
        if use_dropout:
            self.conv_fin += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            self.conv_fin += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            self.conv_fin += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        self.conv_fin += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        self.Dense_time = nn.Linear(temb_dim, dim)
        nn.init.zeros_(self.Dense_time.bias)

        self.style = nn.Linear(z_dim, dim * 2)
        self.style.bias.data[:dim] = 1
        self.style.bias.data[dim:] = 0

        return self.conv_block, self.adaptive, self.conv_fin

    def forward(self, x, time_cond, z):
        time_input = self.Dense_time(time_cond)
        for n, layer in enumerate(self.conv_block):
            out = layer(x)
            if n == 0:
                out += time_input[:, :, None, None]
        out = self.adaptive(out, z)
        for layer in self.conv_fin:
            out = layer(out)
        out = x + out
        return out


def get_filter(filt_size=3):
    if filt_size == 1:
        a = np.array([1., ])
    elif filt_size == 2:
        a = np.array([1., 1.])
    elif filt_size == 3:
        a = np.array([1., 2., 1.])
    elif filt_size == 4:
        a = np.array([1., 3., 3., 1.])
    elif filt_size == 5:
        a = np.array([1., 4., 6., 4., 1.])
    elif filt_size == 6:
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif filt_size == 7:
        a = np.array([1., 6., 15., 20., 15., 6., 1.])
    filt = torch.Tensor(a[:, None] * a[None, :])
    filt = filt / torch.sum(filt)
    return filt


def get_pad_layer(pad_type):
    if pad_type in ['refl', 'reflect']:
        return nn.ReflectionPad2d
    elif pad_type in ['repl', 'replicate']:
        return nn.ReplicationPad2d
    elif pad_type == 'zero':
        return nn.ZeroPad2d
    else:
        raise NotImplementedError('Pad type [%s] not recognized' % pad_type)


class Downsample(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)),
                           int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))
        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class Upsample(nn.Module):
    def __init__(self, channels, pad_type='repl', filt_size=4, stride=2):
        super(Upsample, self).__init__()
        self.filt_size = filt_size
        self.filt_odd = np.mod(filt_size, 2) == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size) * (stride ** 2)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))
        self.pad = get_pad_layer(pad_type)([1, 1, 1, 1])

    def forward(self, inp):
        ret_val = F.conv_transpose2d(self.pad(inp), self.filt, stride=self.stride,
                                      padding=1 + self.pad_size, groups=inp.shape[1])[:, :, 1:, 1:]
        if self.filt_odd:
            return ret_val
        else:
            return ret_val[:, :, :-1, :-1]


def get_norm_layer(norm_type='instance'):
    if norm_type == 'instance':
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'batch':
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)


class ResnetGenerator_ncsn(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9,
                 padding_type='reflect', no_antialias=False, no_antialias_up=False, n_mlp=3):
        assert n_blocks >= 0
        super(ResnetGenerator_ncsn, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        self.ngf = ngf
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            if no_antialias:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2), nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2), nn.ReLU(True), Downsample(ngf * mult * 2)]

        self.model_res = nn.ModuleList()
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            self.model_res += [ResnetBlock_cond(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                                 use_dropout=use_dropout, use_bias=use_bias,
                                                 temb_dim=4 * ngf, z_dim=4 * ngf)]

        model_upsample = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            if no_antialias_up:
                model_upsample += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2,
                                                       padding=1, output_padding=1, bias=use_bias),
                                    norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
            else:
                model_upsample += [
                    Upsample(ngf * mult),
                    nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias),
                    norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
        model_upsample += [nn.ReflectionPad2d(3)]
        model_upsample += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model_upsample += [nn.Tanh()]

        self.model = nn.Sequential(*model)
        self.model_upsample = nn.Sequential(*model_upsample)

        mapping_layers = [PixelNorm(), nn.Linear(self.ngf * 4, self.ngf * 4), nn.LeakyReLU(0.2)]
        for _ in range(n_mlp):
            mapping_layers.append(nn.Linear(self.ngf * 4, self.ngf * 4))
            mapping_layers.append(nn.LeakyReLU(0.2))
        self.z_transform = nn.Sequential(*mapping_layers)

        modules_emb = [nn.Linear(self.ngf, self.ngf * 4)]
        nn.init.zeros_(modules_emb[-1].bias)
        modules_emb += [nn.LeakyReLU(0.2)]
        modules_emb += [nn.Linear(self.ngf * 4, self.ngf * 4)]
        nn.init.zeros_(modules_emb[-1].bias)
        modules_emb += [nn.LeakyReLU(0.2)]
        self.time_embed = nn.Sequential(*modules_emb)

    def forward(self, x, time_cond, z):
        z_embed = self.z_transform(z)
        temb = get_timestep_embedding(time_cond, self.ngf)
        time_embed = self.time_embed(temb)
        out = self.model(x)
        for layer in self.model_res:
            out = layer(out, time_embed, z_embed)
        out = self.model_upsample(out)
        return out


# ==================== Ported test-time sampling loop =======================
# From models/sb_model.py:234-264, netG-only, .cuda() generalized to device.

def sample_fake_B(net, real_A, num_timesteps, tau, ngf, device):
    incs = np.array([0] + [1 / (i + 1) for i in range(num_timesteps - 1)])
    times = np.cumsum(incs)
    times = times / times[-1]
    times = 0.5 * times[-1] + 0.5 * times
    times = np.concatenate([np.zeros(1), times])
    times = torch.tensor(times).float().to(device)

    bs = real_A.shape[0]
    Xt, Xt_1 = None, None
    with torch.no_grad():
        net.eval()
        for t in range(num_timesteps):
            if t > 0:
                delta = times[t] - times[t - 1]
                denom = times[-1] - times[t - 1]
                inter = (delta / denom).reshape(-1, 1, 1, 1)
                scale = (delta * (1 - delta / denom)).reshape(-1, 1, 1, 1)
                Xt = (1 - inter) * Xt + inter * Xt_1.detach() + (scale * tau).sqrt() * torch.randn_like(Xt).to(device)
            else:
                Xt = real_A
            time_idx = (t * torch.ones(size=[bs])).long().to(device)
            z = torch.randn(size=[bs, 4 * ngf]).to(device)
            Xt_1 = net(Xt, time_idx, z)
    return Xt_1


def tensor_to_uint8(tensor):
    """Inline equivalent of util/util.py's tensor2im."""
    image_numpy = tensor[0].clamp(-1.0, 1.0).detach().cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(np.uint8)


# Equivalent of get_transform() with --preprocess none --no_flip (inference,
# no augmentation): ToTensor + Normalize(0.5, 0.5).
infer_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


# ================================ Main ======================================

def main():
    torch.manual_seed(SEED)

    norm_layer = get_norm_layer('instance')
    net = ResnetGenerator_ncsn(
        input_nc=3, output_nc=3, ngf=NGF, norm_layer=norm_layer,
        use_dropout=False, n_blocks=9, no_antialias=False, no_antialias_up=False, n_mlp=N_MLP,
    ).to(DEVICE)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()

    photo_paths = sorted(glob.glob(os.path.join(PHOTO_DIR, '*.jpg')))
    if LIMIT is not None:
        photo_paths = photo_paths[:LIMIT]
    assert photo_paths, f'No images found in {PHOTO_DIR}'
    print(f'Generating {len(photo_paths)} images...')

    os.makedirs(OUT_DIR, exist_ok=True)
    for i, path in enumerate(photo_paths):
        img = Image.open(path).convert('RGB')
        real_A = infer_transform(img).unsqueeze(0).to(DEVICE)
        fake_B = sample_fake_B(net, real_A, NUM_TIMESTEPS, TAU, NGF, DEVICE)
        im = tensor_to_uint8(fake_B)
        out_name = os.path.splitext(os.path.basename(path))[0] + '.jpg'
        Image.fromarray(im).save(os.path.join(OUT_DIR, out_name), quality=JPEG_QUALITY)
        if i % 500 == 0:
            print(f'{i}/{len(photo_paths)}')

    with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(OUT_DIR):
            zf.write(os.path.join(OUT_DIR, fname), arcname=fname)

    n = len(os.listdir(OUT_DIR))
    print(f'Wrote {ZIP_PATH} with {n} images')
    if LIMIT is None and not (7000 <= n <= 10000):
        print(f'WARNING: Kaggle requires 7,000-10,000 images; got {n}')


if __name__ == '__main__':
    main()
