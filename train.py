#!/usr/bin/env python3
"""
Train entrypoint. Example:

    python train.py --config configs/base.json
    python train.py --config configs/base.json --resume runs/mwir2lwir/checkpoints/ckpt_step_0050000.pt
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/base.json')
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    trainer = Trainer(config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == '__main__':
    main()
