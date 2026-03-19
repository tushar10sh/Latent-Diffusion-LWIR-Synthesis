import torch, sys
sys.path.insert(0, '.')
from models.ldm.vae import IRVAE
from data.dataset import build_dataloaders

ckpt = torch.load('runs/mwir2lwir_ldm/stage1_vae/vae_final.pt')
cfg  = ckpt['config']
vae  = IRVAE(
    in_channels=cfg.get('lwir_channels', 1),
    ch=cfg.get('vae_ch', 128),
    ch_mult=tuple(cfg.get('vae_ch_mult', [1, 2, 4])),
    z_channels=cfg.get('z_channels', 4),
)
vae.load_state_dict(ckpt['vae'])
vae.to('cuda')  # model must be on GPU before compute_scale_factor

train_loader, _ = build_dataloaders(
    root=cfg['data_root'], image_size=cfg.get('image_size', 256),
    batch_size=16, num_workers=4, file_ext=cfg.get('file_ext', 'npy'),
)
vae.compute_scale_factor(train_loader, n_batches=100)

torch.save({
    'vae':          vae.state_dict(),
    'scale_factor': vae.scale_factor,  # list[float] of len 4
    'latent_mean':  vae.latent_mean,   # list[float] of len 4
    'config':       cfg,
}, 'runs/mwir2lwir_ldm/stage1_vae/vae_final_recal.pt')
print("Saved. Run --inspect to verify.")