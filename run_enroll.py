"""
Stage 3: Writer Enrollment Script

Loads the fine-tuned encoder from a specified checkpoint, then enrolls
every writer in the dataset using their genuine reference signatures.

For each writer the following are computed and cached (no gradient updates):
  F_refs      : (R, d)      — GAP patch features per reference
  mu          : (d,)        — writer prototype
  sigma_tilde : (d,d)/(d,)  — regularised covariance

Regularisation branch is chosen automatically by R:
  R <= 10      : isotropic  (Sigma_hat + eps * I)
  10 < R <= 30 : Ledoit-Wolf shrinkage via sklearn
  R > 30       : diagonal   diag(var_j + eps)

Usage:
    python run_enroll.py
"""

import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

# ── Import shared encoder architecture ───────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'osv_finetuning'))
from vit import ViTEncoder

from enroll import WriterEnrollment


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

CHECKPOINT_PATH = (
    r"C:\Users\Himanshu Singhal\Desktop\BTP"
    r"\osv_finetuning\finetune_runs\run_20260219_102652"
    r"\finetuned_full_weights.pth"
)

DATA_PATH = (
    r"C:\Users\Himanshu Singhal\Desktop\BTP"
    r"\vit_pretraining\deepsigndb_asymmetric_gasf_test.npz"
)

OUTPUT_DIR = r"C:\Users\Himanshu Singhal\Desktop\BTP\writer_enrollment"

# R_ENROLL : genuine references to use per writer for enrollment.
#            The first R_ENROLL genuine samples are used; the remainder
#            are held out as positive queries for evaluation (Stage 4).
#            Set to 5 to match the training episodic R and leave enough
#            genuine samples for query testing (min genuine count = 8).
R_ENROLL = 15

# Regularisation epsilon for isotropic and diagonal branches. Must be in [0.01, 0.1].
EPSILON = 0.05

# ══════════════════════════════════════════════════════════════════════════════


def _is_genuine(fname: str) -> bool:
    parts = str(fname).lower().split('_')
    if len(parts) >= 2:
        if parts[1] == 'g':
            return True
        if parts[1] == 's':
            return False
    return 'forgery' not in str(fname).lower() and 'skilled' not in str(fname).lower()


def load_encoder(checkpoint_path: str, device: torch.device) -> ViTEncoder:
    """
    Load the fine-tuned ViTEncoder from finetuned_full_weights.pth.
    Architecture parameters are read from the args.json in the same directory.
    """
    run_dir  = os.path.dirname(checkpoint_path)
    args_path = os.path.join(run_dir, 'args.json')
    with open(args_path) as f:
        args = json.load(f)

    encoder = ViTEncoder(
        img_size    = args['img_size'],
        patch_size  = args['patch_size'],
        in_channels = args['in_channels'],
        embed_dim   = args['embed_dim'],
        num_layers  = 12,
        num_heads   = 12,
        mlp_dim     = args['embed_dim'] * 4,
    )

    weights = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(weights['encoder'])
    encoder.eval()
    encoder.to(device)

    print(f"Encoder loaded  (embed_dim={args['embed_dim']}, "
          f"patch_size={args['patch_size']}, "
          f"log_tau={weights['log_tau']:.4f}, "
          f"tau_agg={weights['tau_agg']:.4f})")
    return encoder, args['embed_dim']


def build_writer_index(data_path: str):
    """
    Load the .npz and build a mapping: writer_id -> list of genuine array indices.
    Images are returned already normalised to [-1, 1].
    """
    data        = np.load(data_path, allow_pickle=True)
    images      = data['gasf_data'].astype(np.float32)
    file_names  = data['file_names']

    # Normalise to [-1, 1]
    img_min, img_max = images.min(), images.max()
    if img_max - img_min > 1e-6:
        images = 2.0 * (images - img_min) / (img_max - img_min) - 1.0

    writer_to_indices: dict = defaultdict(list)
    for i, fname in enumerate(file_names):
        if _is_genuine(fname):
            writer_id = str(fname).split('_')[0]
            writer_to_indices[writer_id].append(i)

    return images, dict(writer_to_indices)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    # ── Load encoder ──────────────────────────────────────────────────────────
    print(f"\nCheckpoint : {CHECKPOINT_PATH}")
    encoder, embed_dim = load_encoder(CHECKPOINT_PATH, device)

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"\nDataset    : {DATA_PATH}")
    images, writer_to_indices = build_writer_index(DATA_PATH)
    writers = sorted(writer_to_indices.keys())
    print(f"Writers to enroll : {len(writers)}")

    # Show the R distribution (and which covariance branch each writer will hit)
    r_counts = Counter(
        min(len(writer_to_indices[w]), R_ENROLL) if R_ENROLL else len(writer_to_indices[w])
        for w in writers
    )
    print("R distribution:")
    for r, n in sorted(r_counts.items()):
        if r <= 10:
            branch = 'isotropic (full)'
        elif r <= 30:
            branch = 'Ledoit-Wolf (full)'
        else:
            branch = 'diagonal'
        print(f"  R={r:2d} : {n:3d} writers  ->  {branch}")

    # ── Enroll ────────────────────────────────────────────────────────────────
    enroller    = WriterEnrollment(encoder, embed_dim=embed_dim, epsilon=EPSILON)
    branch_tally = Counter()

    for writer_id in tqdm(writers, desc="Enrolling"):
        indices = writer_to_indices[writer_id]
        if R_ENROLL is not None:
            indices = indices[:R_ENROLL]

        ref_images = torch.from_numpy(images[indices])   # (R, C, H, W)
        enroller.enroll_writer(writer_id, ref_images, device)
        branch_tally[enroller.registry[writer_id]['sigma_type']] += 1

    print(f"\nEnrollment complete  ({len(enroller)} writers)")
    print(f"  full matrices : {branch_tally['full']}")
    print(f"  diagonal vecs : {branch_tally['diag']}")

    # Sanity-check one writer
    sample_wid  = writers[0]
    sample_rec  = enroller.get(sample_wid)
    sigma_shape = sample_rec['sigma_tilde'].shape
    print(f"\n  Sample [{sample_wid}]  R={sample_rec['R']}  "
          f"sigma_type={sample_rec['sigma_type']}  "
          f"sigma_tilde.shape={sigma_shape}  "
          f"mu_norm={np.linalg.norm(sample_rec['mu']):.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    r_tag     = f'R{R_ENROLL}' if R_ENROLL is not None else 'Rall'
    save_stem = os.path.join(OUTPUT_DIR, f'enrollment_{r_tag}_eps{EPSILON}_{timestamp}')

    enroller.save(save_stem)   # writes  save_stem.npz

    # Save config alongside
    cfg = dict(
        checkpoint  = CHECKPOINT_PATH,
        data_path   = DATA_PATH,
        R_enroll    = R_ENROLL,
        epsilon     = EPSILON,
        n_writers   = len(enroller),
        branch_full = branch_tally['full'],
        branch_diag = branch_tally['diag'],
        timestamp   = timestamp,
    )
    with open(save_stem + '_config.json', 'w') as f:
        json.dump(cfg, f, indent=2)

    print(f"\nConfig saved to : {save_stem}_config.json")


if __name__ == '__main__':
    main()
