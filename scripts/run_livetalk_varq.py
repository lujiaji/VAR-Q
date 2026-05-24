#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import importlib.util
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from VAR_Q.hooks import collect_livetalk_memory_breakdown, install_livetalk_hooks
from VAR_Q.paths import prepend_sys_path, require_third_party_repo

MARKER = "VARQ_VIDEO_BENCH_RESULT="


def _install_xfuser_stub_if_missing() -> None:
    if importlib.util.find_spec("xfuser") is not None:
        return
    xfuser = types.ModuleType("xfuser")
    core = types.ModuleType("xfuser.core")
    distributed = types.ModuleType("xfuser.core.distributed")
    distributed.get_sequence_parallel_rank = lambda: 0
    distributed.get_sequence_parallel_world_size = lambda: 1

    class _SingleProcessGroup:
        device_group = None

        def all_gather(self, tensor, dim=0):
            return tensor

    distributed.get_sp_group = lambda: _SingleProcessGroup()
    sys.modules["xfuser"] = xfuser
    sys.modules["xfuser.core"] = core
    sys.modules["xfuser.core.distributed"] = distributed


def _parse_args() -> argparse.Namespace:
    livetalk_root = require_third_party_repo("LiveTalk", "https://github.com/ChenhongyiYang/LiveTalk")
    checkpoint_root = Path(os.environ.get("LIVETALK_CHECKPOINT_ROOT", livetalk_root / "pretrained_checkpoints"))
    parser = argparse.ArgumentParser(description="Run LiveTalk inference with VAR-Q runtime hooks.")
    parser.add_argument("--livetalk_root", type=Path, default=livetalk_root)
    parser.add_argument("--config", type=Path, default=livetalk_root / "configs" / "causal_inference.yaml")
    parser.add_argument("--checkpoint_root", type=Path, default=checkpoint_root)
    parser.add_argument("--wan_base_dir", type=Path, default=None)
    parser.add_argument("--livetalk_ckpt", type=Path, default=None)
    parser.add_argument("--wav2vec_path", type=Path, default=None)
    parser.add_argument("--image_path", type=Path, default=livetalk_root / "examples" / "inference" / "example1.jpg")
    parser.add_argument("--audio_path", type=Path, default=livetalk_root / "examples" / "inference" / "example1.wav")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "scripts" / "output" / "livetalk_varq_demo.mp4")
    parser.add_argument("--duration", type=int, default=2, help="Video duration in seconds. LiveTalk expects 3n+2.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--bits", type=int, default=4, choices=(2, 3, 4, 6, 8))
    parser.add_argument("--baseline", action="store_true", help="Run dense BF16 KV cache without VAR-Q hooks.")
    parser.add_argument("--quant_method", default="VARQ")
    parser.add_argument("--max_scale_seq_len", type=int, default=1560)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no_save", action="store_true")
    args = parser.parse_args()
    args.wan_base_dir = args.wan_base_dir or args.checkpoint_root / "Wan2.1-T2V-1.3B"
    args.livetalk_ckpt = args.livetalk_ckpt or args.checkpoint_root / "LiveTalk-1.3B-V0.1" / "model.safetensors"
    args.wav2vec_path = args.wav2vec_path or args.checkpoint_root / "wav2vec2"
    return args


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _prepare_livetalk_conditions(pipeline, args, noise: torch.Tensor, text_prompts):
    """Build LiveTalk conditions with explicit batch broadcasting.

    LiveTalk's public example constructs image/audio conditions with batch=1.
    For max-BS sweeps we keep third-party code untouched and broadcast the
    single reference image/audio to the requested noise batch here.
    """
    import librosa
    import numpy as np
    from PIL import Image

    batch_size, num_frames, _, _, _ = noise.shape
    video_duration = (num_frames * 4 - 4) / args.fps
    audio_len = int(num_frames * 4 - 3)

    image = Image.open(args.image_path).convert("RGB")
    image = pipeline.transform(image).unsqueeze(0).to(pipeline.device)
    _, _, h, w = image.shape
    select_size = pipeline.__class__.__module__
    from scripts.inference import match_size, resize_pad

    select_size = match_size(getattr(args, f"image_sizes_{args.max_hw}"), h, w)
    image = resize_pad(image, (h, w), select_size)
    image = image * 2.0 - 1.0
    image = image[:, :, None]
    img_lat = pipeline.vae.encode_to_latent(image.to(dtype=pipeline.dtype))
    img_lat = img_lat.repeat(batch_size, num_frames, 1, 1, 1)
    img_lat = img_lat.permute(0, 2, 1, 3, 4)
    msk = torch.zeros_like(img_lat)[:, :1]
    msk[:, :, 1:] = 1
    img_lat = torch.cat([img_lat, msk], dim=1)
    print("img_lat:", img_lat.shape)

    audio, sr = librosa.load(args.audio_path, sr=args.sample_rate)
    max_samples = int(video_duration * sr)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
        print(f"Audio trimmed to {video_duration} seconds")

    input_values = np.squeeze(pipeline.wav_feature_extractor(audio, sampling_rate=16000).input_values)
    input_values = torch.from_numpy(input_values).float().to(device=pipeline.device, dtype=pipeline.dtype).unsqueeze(0)
    with torch.no_grad():
        hidden_states = pipeline.audio_encoder(input_values, seq_len=audio_len, output_hidden_states=True)
        audio_embeddings = hidden_states.last_hidden_state
        for mid_hidden_states in hidden_states.hidden_states:
            audio_embeddings = torch.cat((audio_embeddings, mid_hidden_states), -1)
        audio_emb = audio_embeddings.permute(0, 2, 1)[:, :, :, None, None]
        audio_emb = torch.cat([audio_emb[:, :, :1].repeat(1, 1, 3, 1, 1), audio_emb], 2)
        audio_emb = pipeline.generator.audio_proj(audio_emb.to(pipeline.dtype))
        audio_emb = torch.concat([audio_cond_proj(audio_emb) for audio_cond_proj in pipeline.generator.audio_cond_projs], 0)
        if batch_size > 1:
            audio_emb = audio_emb.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1, 1).flatten(0, 1)
        print("audio_shape:", audio_emb.shape)

    return pipeline.inference(
        noise=noise,
        text_prompts=text_prompts,
        img_lat=img_lat,
        audio_embed=audio_emb,
        initial_latent=None,
        return_latents=False,
    )


def main() -> None:
    cli = _parse_args()
    _require_file(cli.config, "LiveTalk config")
    _require_file(cli.wan_base_dir / "models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1 text encoder")
    _require_file(cli.wan_base_dir / "Wan2.1_VAE.pth", "Wan2.1 VAE")
    _require_file(cli.livetalk_ckpt, "LiveTalk checkpoint")
    _require_file(cli.wav2vec_path / "config.json", "wav2vec2 checkpoint")
    _require_file(cli.image_path, "reference image")
    _require_file(cli.audio_path, "reference audio")

    _install_xfuser_stub_if_missing()
    prepend_sys_path([REPO_ROOT, cli.livetalk_root, cli.livetalk_root / "OmniAvatar"])
    os.chdir(cli.livetalk_root)

    hparams = {
        "text_encoder_path": cli.wan_base_dir / "models_t5_umt5-xxl-enc-bf16.pth",
        "dit_path": cli.livetalk_ckpt,
        "vae_path": cli.wan_base_dir / "Wan2.1_VAE.pth",
        "wav2vec_path": cli.wav2vec_path,
        "image_path": cli.image_path,
        "audio_path": cli.audio_path,
        "output_path": cli.output,
        "video_duration": cli.duration,
    }
    old_argv = sys.argv[:]
    sys.argv = [
        "run_livetalk_varq.py",
        "--config",
        str(cli.config),
        "-hp",
        ",".join(f"{key}={value}" for key, value in hparams.items()),
    ]
    try:
        from scripts import inference_example as livetalk
    finally:
        sys.argv = old_argv

    device = torch.device(cli.device)
    torch.set_grad_enabled(False)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    load_start = time.perf_counter()
    pipeline = livetalk.CausalInferencePipeline.from_pretrained(args=livetalk.args, device=device)
    handle = None
    if not cli.baseline:
        handle = install_livetalk_hooks(
            pipeline,
            {
                "enable": True,
                "quant_method": cli.quant_method,
                "q_bits": cli.bits,
                "qkv_format": "BLHc",
                "pack_to_int32": True,
                "compression_ratio": 1.0,
                "max_scale_seq_len": cli.max_scale_seq_len,
                "dequant_dtype": "native",
                "quant_compute_dtype": "native",
                "dequant_workspace_policy": "release",
            },
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    if handle is None:
        print("[LiveTalk] Running baseline dense BF16 KV cache.")
    else:
        print(f"[LiveTalk] Installed VAR-Q hooks on {len(handle.modules)} attention modules.")
    print(f"[LiveTalk] Load + hook time: {time.perf_counter() - load_start:.2f}s")

    args = livetalk.args
    num_frames = (int(args.video_duration) * int(args.fps) + 4) // 4
    noise = torch.randn([cli.batch_size, num_frames, 16, 64, 64], device=device, dtype=pipeline.dtype)
    gen_start = time.perf_counter()
    video = _prepare_livetalk_conditions(pipeline, args, noise, [args.prompt] * cli.batch_size)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - gen_start
    print(f"[LiveTalk] Generated shape: {tuple(video.shape)}")
    print(f"[LiveTalk] Generation time: {elapsed:.2f}s")
    stats = collect_livetalk_memory_breakdown(pipeline) if handle is not None else {}
    if device.type == "cuda":
        stats.update(
            {
                "cuda_memory_allocated": int(torch.cuda.memory_allocated()),
                "cuda_max_memory_allocated": int(torch.cuda.max_memory_allocated()),
                "cuda_memory_reserved": int(torch.cuda.memory_reserved()),
                "cuda_max_memory_reserved": int(torch.cuda.max_memory_reserved()),
            }
        )
    print(f"[LiveTalk] Memory breakdown: {stats}")
    print(
        MARKER
        + json.dumps(
            {
                "ok": True,
                "model": "LiveTalk",
                "method": "baseline" if cli.baseline else "varq",
                "bits": None if cli.baseline else cli.bits,
                "batch_size": cli.batch_size,
                "elapsed_sec": elapsed,
                "throughput_items_per_sec": cli.batch_size / elapsed,
                "shape": list(video.shape),
                **stats,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    if cli.no_save:
        return

    import imageio
    import numpy as np

    cli.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = cli.output.with_suffix(".tmp.mp4")
    video_np = (video.squeeze(0).permute(0, 2, 3, 1).cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)
    imageio.mimsave(
        tmp_output,
        video_np,
        fps=args.fps,
        codec="libx264",
        macro_block_size=None,
        ffmpeg_params=["-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p"],
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(tmp_output),
            "-i",
            str(args.audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-ac",
            "1",
            "-b:a",
            "96k",
            "-movflags",
            "+faststart",
            "-shortest",
            str(cli.output),
        ],
        check=True,
    )
    tmp_output.unlink(missing_ok=True)
    print(f"[LiveTalk] Saved: {cli.output}")


if __name__ == "__main__":
    main()
