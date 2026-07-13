"""Periodic checkpoint evaluator for a UNSB `sb`-mode training run.

For every saved `<epoch>_net_G.pth` checkpoint not yet logged, generates
images for a held-out photo set (testA) using ONLY netG (see
scripts/minimal_sb_inference.py), computes FID against the real Monet
reference set (dataset_monet) via pytorch-fid, and appends epoch,fid to a
CSV log. This is a proxy for MiFID (no memorization/near-duplicate penalty
term is computed here) -- good enough to discard bad checkpoints and catch
divergence/mode collapse, not a substitute for a final visual QA pass.

Usage:
    python scripts/eval_checkpoints.py \
        --checkpoints_dir ./checkpoints/monet_sb_v1 \
        --val_dir ./datasets/monet2photo/testA \
        --monet_ref_dir ../dataset_monet \
        --out_csv ./checkpoints/monet_sb_v1/fid_log.csv \
        --tmp_dir ./checkpoints/monet_sb_v1/eval_tmp
"""
import argparse
import csv
import glob
import os
import re
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.minimal_sb_inference import (
    build_netG, load_weights, get_infer_transform, load_image_tensor,
    sample_fake_B, tensor_to_uint8,
)
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints_dir', required=True, help='e.g. ./checkpoints/monet_sb_v1')
    parser.add_argument('--val_dir', required=True, help='held-out photo folder (testA)')
    parser.add_argument('--monet_ref_dir', required=True, help='real Monet reference folder (dataset_monet)')
    parser.add_argument('--out_csv', required=True)
    parser.add_argument('--tmp_dir', required=True, help='where generated images per epoch are written')
    parser.add_argument('--num_timesteps', type=int, default=5)
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--n_mlp', type=int, default=3)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--fid_batch_size', type=int, default=50)
    return parser.parse_args()


def find_checkpoints(checkpoints_dir):
    """Return sorted [(epoch:int, path)] for <epoch>_net_G.pth, excluding 'latest'."""
    out = []
    for p in glob.glob(os.path.join(checkpoints_dir, '*_net_G.pth')):
        m = re.match(r'(\d+)_net_G\.pth$', os.path.basename(p))
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def load_already_logged(out_csv):
    done = set()
    if os.path.exists(out_csv):
        with open(out_csv, newline='') as f:
            for row in csv.reader(f):
                if row and row[0].isdigit():
                    done.add(int(row[0]))
    return done


def append_csv(out_csv, epoch, fid):
    write_header = not os.path.exists(out_csv)
    with open(out_csv, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['epoch', 'fid'])
        w.writerow([epoch, f'{fid:.4f}'])


def _patch_pytorch_fid_scipy_compat():
    """pytorch-fid==0.3.0 (last PyPI release, 2020) calls
    scipy.linalg.sqrtm(A, disp=False) and unpacks a (sqrtm, errest) tuple.
    Modern scipy dropped the disp/blocksize kwargs and always returns a bare
    array. Shim the old calling convention back in, rather than pinning an
    old scipy that would fight the modern numpy torch already depends on.
    """
    from pytorch_fid import fid_score as _fs
    _orig_sqrtm = _fs.linalg.sqrtm

    def _sqrtm_compat(A, disp=True, blocksize=64):
        result = _orig_sqrtm(A)
        return result if disp else (result, 0)

    _fs.linalg.sqrtm = _sqrtm_compat


def main():
    args = parse_args()
    gpu_ids = [int(x) for x in args.gpu_ids.split(',') if int(x) >= 0]
    device = torch.device(f'cuda:{gpu_ids[0]}' if gpu_ids else 'cpu')

    from pytorch_fid.fid_score import calculate_fid_given_paths
    _patch_pytorch_fid_scipy_compat()

    net = build_netG(ngf=args.ngf, n_mlp=args.n_mlp, gpu_ids=gpu_ids)
    transform = get_infer_transform()
    val_paths = sorted(
        os.path.join(args.val_dir, f) for f in os.listdir(args.val_dir)
        if f.lower().endswith(('.jpg', '.jpeg'))
    )
    assert val_paths, f'No images found in {args.val_dir}'

    already_done = load_already_logged(args.out_csv)
    checkpoints = find_checkpoints(args.checkpoints_dir)
    if not checkpoints:
        print(f'No <epoch>_net_G.pth checkpoints found in {args.checkpoints_dir}')
        return

    fid_batch_size = min(args.fid_batch_size, len(val_paths))
    for epoch, ckpt_path in checkpoints:
        if epoch in already_done:
            continue
        load_weights(net, ckpt_path, device)
        gen_dir = os.path.join(args.tmp_dir, f'epoch_{epoch}')
        os.makedirs(gen_dir, exist_ok=True)
        for p in val_paths:
            real_A = load_image_tensor(p, transform, device)
            fake_B = sample_fake_B(net, real_A, args.num_timesteps, args.tau, args.ngf, device)
            im = tensor_to_uint8(fake_B)
            Image.fromarray(im).save(os.path.join(gen_dir, os.path.basename(p)))

        fid = calculate_fid_given_paths(
            [gen_dir, args.monet_ref_dir], batch_size=fid_batch_size, device=device, dims=2048,
        )
        print(f'epoch {epoch}: FID={fid:.4f}')
        append_csv(args.out_csv, epoch, fid)

    all_results = []
    with open(args.out_csv, newline='') as f:
        for row in csv.reader(f):
            if row and row[0].isdigit():
                all_results.append((int(row[0]), float(row[1])))
    if all_results:
        best = sorted(all_results, key=lambda x: x[1])[:5]
        print('\nBest epochs by FID (lower is better):')
        for epoch, fid in best:
            print(f'  epoch {epoch}: FID={fid:.4f}')


if __name__ == '__main__':
    main()
