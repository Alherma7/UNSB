"""Build a UNSB-compatible trainA/trainB/testA/testB dataroot from the flat
dataset_monet / dataset_paisajes folders used for the Kaggle
"I'm Something of a Painter Myself" (gan-getting-started) competition.

Usage:
    python datasets/prepare_monet_dataset.py \
        --monet_dir ../dataset_monet --photo_dir ../dataset_paisajes \
        --out_dir ./datasets/monet2photo --val_size 24 --seed 42
"""
import argparse
import json
import os
import random
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--monet_dir', required=True, help='path to flat folder of Monet paintings (domain B)')
    parser.add_argument('--photo_dir', required=True, help='path to flat folder of photos (domain A)')
    parser.add_argument('--out_dir', required=True, help='dataroot to create (trainA/trainB/testA/testB)')
    parser.add_argument('--val_size', type=int, default=24, help='# of photos held out into testA for FID monitoring; also the size of the testB placeholder folder')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--expected_monet_count', type=int, default=300)
    parser.add_argument('--expected_photo_count', type=int, default=7038)
    return parser.parse_args()


def list_jpgs(d):
    return sorted(
        os.path.join(d, f) for f in os.listdir(d)
        if f.lower().endswith(('.jpg', '.jpeg'))
    )


def copy_all(paths, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    for p in paths:
        shutil.copy2(p, os.path.join(dest_dir, os.path.basename(p)))


def main():
    args = parse_args()
    random.seed(args.seed)

    monet_files = list_jpgs(args.monet_dir)
    photo_files = list_jpgs(args.photo_dir)

    assert len(monet_files) == args.expected_monet_count, (
        f"Expected {args.expected_monet_count} Monet images in {args.monet_dir}, found {len(monet_files)}"
    )
    assert len(photo_files) == args.expected_photo_count, (
        f"Expected {args.expected_photo_count} photos in {args.photo_dir}, found {len(photo_files)}"
    )
    assert args.val_size < len(photo_files) and args.val_size <= len(monet_files)

    # trainB: full Monet set. Kaggle's own MiFID reference set IS this same
    # 300-image folder, so there's no benefit to holding any of it out.
    copy_all(monet_files, os.path.join(args.out_dir, 'trainB'))

    # testB: small arbitrary non-empty folder so UnalignedDataset doesn't
    # crash at test time. Its content does not influence the generated
    # output (SBModel's test-time sampling loop only reads real_A).
    copy_all(monet_files[:args.val_size], os.path.join(args.out_dir, 'testB'))

    # trainA / testA: random split of the photos, held out disjointly so the
    # periodic FID-monitoring set (testA) is never seen during training.
    shuffled_photos = photo_files[:]
    random.shuffle(shuffled_photos)
    val_photos = shuffled_photos[:args.val_size]
    train_photos = shuffled_photos[args.val_size:]

    copy_all(train_photos, os.path.join(args.out_dir, 'trainA'))
    copy_all(val_photos, os.path.join(args.out_dir, 'testA'))

    manifest = {
        'seed': args.seed,
        'val_size': args.val_size,
        'trainA_count': len(train_photos),
        'trainB_count': len(monet_files),
        'testA_count': len(val_photos),
        'testB_count': args.val_size,
        'testA_files': [os.path.basename(p) for p in val_photos],
        'testB_files': [os.path.basename(p) for p in monet_files[:args.val_size]],
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'split_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"trainA: {manifest['trainA_count']} photos")
    print(f"trainB: {manifest['trainB_count']} monet paintings")
    print(f"testA:  {manifest['testA_count']} held-out photos")
    print(f"testB:  {manifest['testB_count']} monet paintings (placeholder, not used to shape output)")
    print(f"manifest written to {os.path.join(args.out_dir, 'split_manifest.json')}")


if __name__ == '__main__':
    main()
