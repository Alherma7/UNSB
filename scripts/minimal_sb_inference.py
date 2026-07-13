"""Minimal, netG-only inference path for the UNSB `sb` model.

Bypasses `test.py` / `SBModel.forward()`, which unconditionally touch
`real_B`/`real_A2` (a second dataset) even though the actual test-time
output only ever depends on `real_A` (see `models/sb_model.py:234-264`).
Used by `scripts/eval_checkpoints.py` and `scripts/generate_submission.py`.

Can be invoked either as `python scripts/eval_checkpoints.py` from UNSB/ or
via its full path from anywhere; the UNSB repo root is added to sys.path
below so `models.*`/`data.*` are importable regardless of cwd (mirrors the
import-path fix needed for vgg_sb/evaluations/fid_score.py).
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data.base_dataset import get_transform
from models.networks import get_norm_layer, init_net
from models.ncsn_networks import ResnetGenerator_ncsn
import util.util as util


def build_netG(ngf=64, n_mlp=3, gpu_ids=None, init_type='xavier', init_gain=0.02):
    """Construct the same generator define_G() builds for --netG resnet_9blocks_cond
    (the default), without needing a full argparse options object."""
    gpu_ids = gpu_ids or []
    norm_layer = get_norm_layer(norm_type='instance')
    opt_stub = SimpleNamespace(n_mlp=n_mlp)  # only opt field ResnetGenerator_ncsn.__init__ reads
    net = ResnetGenerator_ncsn(
        input_nc=3, output_nc=3, ngf=ngf, norm_layer=norm_layer,
        use_dropout=False, n_blocks=9,
        no_antialias=False, no_antialias_up=False, opt=opt_stub,
    )
    # initialize_weights=False: random init is pointless, load_weights() overwrites immediately.
    return init_net(net, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, initialize_weights=False)


def load_weights(net, checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()
    return net


def get_infer_transform():
    """Equivalent of get_transform() with --preprocess none --no_flip: for
    already-256x256 images this is just ToTensor + Normalize(0.5, 0.5)."""
    opt_stub = SimpleNamespace(preprocess='none', no_flip=True, dataroot='')
    return get_transform(opt_stub, grayscale=False)


def load_image_tensor(path, transform, device):
    img = Image.open(path).convert('RGB')
    return transform(img).unsqueeze(0).to(device)


def sample_fake_B(net, real_A, num_timesteps, tau, ngf, device):
    """Line-for-line reimplementation of SBModel.forward()'s test-time
    sampling loop (models/sb_model.py:234-264), generalized to `device` and
    operating on netG alone. Returns the final refined image, fake_{num_timesteps}.
    """
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
    return util.tensor2im(tensor)
