"""Local dry-run submission generator for the Kaggle "I'm Something of a
Painter Myself" (gan-getting-started) competition.

Runs a chosen checkpoint over the photo set and writes 256x256 JPEGs into a
flat `images/` folder, zipped as `images.zip`. This validates output
format/count/timing locally -- it is NOT the graded submission: this is a
code competition, so the real submission must be produced by a committed
Kaggle Notebook (see the plan's step 5 / the ported notebook script).

Usage:
    python scripts/generate_submission.py \
        --photo_dir ../dataset_paisajes \
        --checkpoint ./checkpoints/monet_sb_v1/<BEST_EPOCH>_net_G.pth \
        --out_dir ./submission_dryrun/images \
        --zip_path ./submission_dryrun/images.zip \
        [--limit 20]
"""
import argparse
import os
import sys
import zipfile

import torch
from PIL import Image

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.minimal_sb_inference import (
    build_netG, load_weights, get_infer_transform, load_image_tensor,
    sample_fake_B, tensor_to_uint8,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--photo_dir', required=True)
    parser.add_argument('--checkpoint', required=True, help='path to <epoch>_net_G.pth')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--zip_path', required=True)
    parser.add_argument('--num_timesteps', type=int, default=5)
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--n_mlp', type=int, default=3)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--jpeg_quality', type=int, default=95)
    parser.add_argument('--limit', type=int, default=None, help='only process the first N photos (smoke testing)')
    return parser.parse_args()


def main():
    args = parse_args()
    gpu_ids = [int(x) for x in args.gpu_ids.split(',') if int(x) >= 0]
    device = torch.device(f'cuda:{gpu_ids[0]}' if gpu_ids else 'cpu')

    net = build_netG(ngf=args.ngf, n_mlp=args.n_mlp, gpu_ids=gpu_ids)
    load_weights(net, args.checkpoint, device)
    transform = get_infer_transform()

    photo_paths = sorted(
        os.path.join(args.photo_dir, f) for f in os.listdir(args.photo_dir)
        if f.lower().endswith(('.jpg', '.jpeg'))
    )
    if args.limit is not None:
        photo_paths = photo_paths[:args.limit]
    assert photo_paths, f'No images found in {args.photo_dir}'

    os.makedirs(args.out_dir, exist_ok=True)
    for i, p in enumerate(photo_paths):
        real_A = load_image_tensor(p, transform, device)
        fake_B = sample_fake_B(net, real_A, args.num_timesteps, args.tau, args.ngf, device)
        im = tensor_to_uint8(fake_B)
        out_name = os.path.splitext(os.path.basename(p))[0] + '.jpg'
        Image.fromarray(im).save(os.path.join(args.out_dir, out_name), quality=args.jpeg_quality)
        if i % 500 == 0:
            print(f'{i}/{len(photo_paths)}')

    os.makedirs(os.path.dirname(args.zip_path) or '.', exist_ok=True)
    with zipfile.ZipFile(args.zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(args.out_dir):
            zf.write(os.path.join(args.out_dir, fname), arcname=fname)

    n = len(os.listdir(args.out_dir))
    print(f'Wrote {args.zip_path} with {n} images')
    if not (7000 <= n <= 10000) and args.limit is None:
        print(f'WARNING: Kaggle requires 7,000-10,000 images; got {n}')


if __name__ == '__main__':
    main()
