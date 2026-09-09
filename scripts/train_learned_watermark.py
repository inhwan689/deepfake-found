"""Train the learned watermark against the swap-shaped distortion layer.

Two settings here were chosen by measurement rather than taste.

The distortion strength is ramped from zero over the first part of training. A
first run applied it at full strength from step one and stalled at 0.614 bit
accuracy by step 1000; with the ramp the same step reached 0.623 and kept
climbing to 0.809. Starting hard leaves the model in a shallow solution.

The message and image losses are weighted against each other because they pull in
opposite directions: the residual has to be strong enough to survive resampling
and faint enough to stay invisible. ``--img-weight`` is the knob, and the PSNR
reported each validation is what it buys.

Validation reports bit accuracy at three distortion strengths. The clean column
is the model's own ceiling - what it can do with nothing done to the carrier -
and no amount of robustness work raises it.

``--vae-roundtrip`` adds a second distortion that stands in for diffusion
regeneration. Diffusion itself cannot go in the loop - thirty denoising steps are
not practically differentiable and cost seconds per image - but every img2img
output passes through Stable Diffusion's VAE decoder, and its eight-fold spatial
compression is the bottleneck that erases the mark: the swap-only model drops from
0.879 to 0.580 bit accuracy through the VAE round trip alone, with no denoising.
Encode-decode is one differentiable forward pass, so on every second step a
random quarter of the batch is passed through it after the swap distortion. The
VAE weights are frozen and gradients flow through them into the encoder. A model
trained this way (v3) was measured against real thirty-step img2img and held its
attribution where the swap-only model lost it; the numbers are in the README.

A run of this length outlives a laptop lid, so the state needed to continue one -
both networks, the optimiser, the schedule and the step - is written every
``--checkpoint-every`` steps. The write goes to a temporary file and is renamed
over the destination, because a process killed midway through writing a
checkpoint would otherwise leave a truncated one where the good one used to be.
``--resume`` continues from whatever the checkpoint last recorded.

Usage:
    python scripts/train_learned_watermark.py --data data/results/face_crops_128.npy \
        --out models/learned_watermark.pt
    python scripts/train_learned_watermark.py --data data/results/face_crops_128.npy \
        --out models/learned_watermark.pt --resume
    python scripts/train_learned_watermark.py --data data/results/face_crops_128.npy \
        --out models/learned_watermark_v3.pt --steps 20000 --vae-roundtrip
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from learned_watermark import BITS, Decoder, Encoder, distort, pick_device

VALIDATION = 512
VALIDATION_CHUNK = 64
WARMUP_FRACTION = 0.4
VAE_REPOSITORY = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def load_vae(device: str) -> Any:
    """Return Stable Diffusion's VAE, frozen, as a differentiable distortion."""
    try:
        from diffusers import AutoencoderKL
    except ImportError as exc:
        raise SystemExit(
            "--vae-roundtrip needs diffusers; pip install diffusers transformers"
        ) from exc
    vae = AutoencoderKL.from_pretrained(VAE_REPOSITORY, subfolder="vae").to(device)
    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    if device == "cuda":
        # A 16 GB T4 cannot hold the round trip's activations for eight
        # 128-pixel images; recomputing them in the backward pass can.
        vae.enable_gradient_checkpointing()
    return vae


def vae_roundtrip(vae: Any, image: torch.Tensor) -> torch.Tensor:
    """Encode to the latent and decode back, keeping the graph for backprop.

    On CUDA the VAE runs under fp16 autocast, which halves its activation
    memory; the encoder and decoder being trained stay in fp32.
    """
    with torch.autocast("cuda", dtype=torch.float16, enabled=bool(image.is_cuda)):
        latent = vae.encode(image).latent_dist.mean
        decoded: torch.Tensor = vae.decode(latent).sample
    return decoded.float().clamp(-1, 1)


def peak_snr(marked: torch.Tensor, original: torch.Tensor) -> float:
    """Return PSNR in decibels for tensors scaled to [-1, 1]."""
    error = ((marked - original) ** 2).mean().item()
    return float(10 * np.log10(4.0 / max(error, 1e-12)))


def main(argv: list[str] | None = None) -> int:
    """Train the encoder and decoder jointly and save both."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=9000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--img-weight", type=float, default=1.5)
    parser.add_argument("--res-scale", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument(
        "--resume", action="store_true", help="continue from the checkpoint at --out"
    )
    parser.add_argument(
        "--vae-roundtrip",
        action="store_true",
        help="also pass a random quarter of every second batch through the SD VAE",
    )
    parser.add_argument("--vae-fraction", type=float, default=0.25)
    parser.add_argument("--vae-every", type=int, default=2)
    args = parser.parse_args(argv)

    if not args.data.is_file():
        raise SystemExit(f"missing {args.data}; run scripts/prepare_face_crops.py first")

    device = pick_device()
    crops = np.load(args.data)
    if crops.shape[0] <= VALIDATION:
        raise SystemExit(
            f"{args.data} holds {crops.shape[0]} crops, too few to hold out {VALIDATION}"
        )
    held_out = torch.from_numpy(crops[:VALIDATION]).permute(0, 3, 1, 2)
    held_out = held_out.float().div(127.5).sub(1).to(device)
    training = torch.from_numpy(crops[VALIDATION:]).permute(0, 3, 1, 2)
    print(f"device={device} train={tuple(training.shape)} val={tuple(held_out.shape)}", flush=True)

    encoder, decoder = Encoder().to(device), Decoder().to(device)
    vae = load_vae(device) if args.vae_roundtrip else None
    parameters = list(encoder.parameters()) + list(decoder.parameters())
    optimiser = torch.optim.Adam(parameters, lr=args.lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, args.steps)

    first_step = 1
    if args.resume:
        if not args.out.is_file():
            raise SystemExit(f"--resume given but {args.out} does not exist")
        saved = torch.load(args.out, map_location=device)
        encoder.load_state_dict(saved["encoder"])
        decoder.load_state_dict(saved["decoder"])
        if "optimiser" in saved:
            optimiser.load_state_dict(saved["optimiser"])
            schedule.load_state_dict(saved["schedule"])
            first_step = saved["step"] + 1
            print(f"resuming at step {first_step}", flush=True)
        else:
            print(f"{args.out} holds weights only; restarting the schedule", flush=True)

    def save(step: int) -> None:
        """Write the checkpoint atomically so a kill cannot truncate it."""
        args.out.parent.mkdir(parents=True, exist_ok=True)
        staging = args.out.with_suffix(args.out.suffix + ".partial")
        torch.save(
            {
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "optimiser": optimiser.state_dict(),
                "schedule": schedule.state_dict(),
                "step": step,
                "steps": args.steps,
                "bits": BITS,
                "res_scale": args.res_scale,
                "vae_roundtrip": bool(vae is not None),
                "orientation": "upright",
            },
            staging,
        )
        staging.replace(args.out)

    started = time.time()
    for step in range(first_step, args.steps + 1):
        warmed = min(1.0, step / (args.steps * WARMUP_FRACTION))
        batch = training[torch.randint(0, training.size(0), (args.batch,))]
        images = batch.float().div(127.5).sub(1).to(device)
        message = torch.randint(0, 2, (args.batch, BITS), device=device).float()
        marked = (images + encoder(images, message) * args.res_scale).clamp(-1, 1)
        distorted = distort(marked, warmed)
        if vae is not None and step % args.vae_every == 0:
            subset = max(1, int(round(args.batch * args.vae_fraction)))
            chosen = torch.randperm(args.batch, device=device)[:subset]
            distorted = distorted.clone()
            distorted[chosen] = vae_roundtrip(vae, distorted[chosen])
        logits = decoder(distorted)
        message_loss = F.binary_cross_entropy_with_logits(logits, message)
        image_loss = F.mse_loss(marked, images)
        loss = message_loss + args.img_weight * image_loss
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        schedule.step()

        if step % 250 == 0 or step == 1:
            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                probe = torch.randint(0, 2, (VALIDATION, BITS), device=device).float()
                # Validation runs in chunks for the same reason the VAE round trip
                # below does: one 512-image forward pass at 128 pixels asks a 16 GB
                # card for two gigabytes in a single activation and loses the run to
                # an out-of-memory error. Both networks are in eval mode, so their
                # batch norms read running statistics and the chunked result is the
                # same numbers the whole-batch pass would have produced.
                candidate = torch.cat(
                    [
                        (
                            held_out[start : start + VALIDATION_CHUNK]
                            + encoder(
                                held_out[start : start + VALIDATION_CHUNK],
                                probe[start : start + VALIDATION_CHUNK],
                            )
                            * args.res_scale
                        ).clamp(-1, 1)
                        for start in range(0, VALIDATION, VALIDATION_CHUNK)
                    ]
                )
                accuracies = []
                for level in (0.0, 0.5, 1.0):
                    hits = []
                    for start in range(0, VALIDATION, VALIDATION_CHUNK):
                        piece = candidate[start : start + VALIDATION_CHUNK]
                        read = (decoder(distort(piece, level)) > 0).float()
                        hits.append(
                            (read == probe[start : start + VALIDATION_CHUNK]).float().mean().item()
                        )
                    accuracies.append(float(np.mean(hits)))
                vae_note = ""
                if vae is not None:
                    hits = []
                    for start in range(0, VALIDATION, VALIDATION_CHUNK):
                        piece = vae_roundtrip(
                            vae, candidate[start : start + VALIDATION_CHUNK]
                        )
                        read = (decoder(piece) > 0).float()
                        hits.append(
                            (read == probe[start : start + VALIDATION_CHUNK]).float().mean().item()
                        )
                    vae_note = f" vae {float(np.mean(hits)):.3f}"
            print(
                f"step {step:5d} loss {loss.item():.4f} msg {message_loss.item():.4f} "
                f"psnr {peak_snr(candidate, held_out):.1f} "
                f"acc[clean/mid/full] {accuracies[0]:.3f}/{accuracies[1]:.3f}/{accuracies[2]:.3f}"
                f"{vae_note} warm {warmed:.2f} {time.time() - started:.0f}s",
                flush=True,
            )
            encoder.train()
            decoder.train()

        if step % args.checkpoint_every == 0:
            save(step)

    save(args.steps)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
