#!/usr/bin/env python3
"""
LDM training entrypoint.

Stage 1 + Stage 2:
    python train_ldm.py --config configs/ldm.json

Skip VAE (already trained):
    python train_ldm.py --config configs/ldm.json --skip_vae --vae_ckpt runs/mwir2lwir_ldm/stage1_vae/vae_final.pt

Stage 1 only:
    python train_ldm.py --config configs/ldm.json --stage1_only
"""

import argparse
import json
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from training.ldm_trainer import train_ldm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/ldm.json')
    parser.add_argument('--skip_vae', action='store_true')
    parser.add_argument('--vae_ckpt', default=None)
    parser.add_argument('--stage1_only', action='store_true')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    if args.stage1_only:
        from training.ldm_trainer import VAETrainer
        trainer = VAETrainer(config)
        trainer.train()
    else:
        train_ldm(config, skip_vae=args.skip_vae, vae_ckpt=args.vae_ckpt)


if __name__ == '__main__':
    main()
