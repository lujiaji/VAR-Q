cd /home/jiaji_lu/AR/VAR-Q/InfinityStar && \
GPU_ID=0 \
PROMPTS_JSON="/home/jiaji_lu/AR/VAR-Q/InfinityStar/evaluation/VBench_rewrited_prompt.json" \
CHECKPOINTS_DIR="/data/jiaji_lu/WM/infinitystar/checkpoints/InfinityStar" \
CONFIG_FILE="/home/jiaji_lu/AR/VAR-Q/VAR_Q/Infinity-VAR_Q-8.json" \
OUT_DIR="/home/jiaji_lu/AR/VAR-Q/Benchmark/outputs/vbench_eval_varq_480p/gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S)" \
GEN_DURATION=5 \
BASE_SEED=42 \
CUDA_VISIBLE_DEVICES="${GPU_ID}" ENABLE_VARQ=1 PYTHONPATH="/home/jiaji_lu/AR/VAR-Q:${PYTHONPATH}" \
nohup python3 - <<'PY' > "${OUT_DIR}.log" 2>&1 &
import os, json
from pathlib import Path
from tqdm import tqdm
from infinity.utils.arg_util import Args
from tools.infer_video_480p import InferencePipe, load_varq_quant_config, perform_inference
from tools.run_infinity import save_video

prompts_json = os.environ["PROMPTS_JSON"]
checkpoints_dir = os.environ["CHECKPOINTS_DIR"]
config_file = os.environ["CONFIG_FILE"]
out_dir = Path(os.environ["OUT_DIR"])
gen_duration = int(os.environ.get("GEN_DURATION", "5"))
base_seed = int(os.environ.get("BASE_SEED", "42"))

video_dir = out_dir / "videos"
video_dir.mkdir(parents=True, exist_ok=True)

with open(prompts_json, "r") as f:
    prompts = json.load(f)

quant_config = load_varq_quant_config(config_file)

args = Args()
args.pn = "0.40M"
args.fps = 16
args.video_frames = gen_duration * 16 + 1
args.model_path = os.path.join(checkpoints_dir, "infinitystar_8b_480p_weights")
args.checkpoint_type = "torch_shard"
args.vae_path = os.path.join(checkpoints_dir, "infinitystar_videovae.pth")
args.text_encoder_ckpt = os.path.join(checkpoints_dir, "text_encoder/flan-t5-xl-official/")
args.videovae = 10
args.model_type = "infinity_qwen8b"
args.text_channels = 2048
args.dynamic_scale_schedule = os.environ.get("INFINITY_SCHEDULE", "infinity_elegant_clip20frames_v2")
args.mask_type = args.dynamic_scale_schedule if "infinity_star" in args.dynamic_scale_schedule else "infinity_elegant_clip20frames_v2"
args.enable_quantization = int(quant_config.get("enable", False))
args.q_bits = quant_config.get("q_bits", 8)
args.quant_method = quant_config.get("quant_method", "G_SCALE_HEAD_DIM")
args.qkv_format = quant_config.get("qkv_format", "BHLc")
args.rescale_qk = int(quant_config.get("rescale_qk", False))
args.enable_sageattn = int(quant_config.get("enable_sageattn", False))
args.sageattn_type = str(quant_config.get("sageattn_type", "sageattn"))
args.bf16 = 1
args.use_apg = 1
args.use_cfg = 0
args.cfg = 34
args.tau_image = 1
args.tau_video = 0.4
args.apg_norm_threshold = 0.05
args.image_scale_repetition = "[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]"
args.video_scale_repetition = "[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1]"
args.append_duration2caption = 1
args.use_two_stage_lfq = 1
args.detail_scale_min_tokens = 350
args.semantic_scales = 11
args.max_repeat_times = 10000
args.enable_rewriter = 0

pipe = InferencePipe(args)
results = []

for i, item in enumerate(tqdm(prompts, desc="Generating 480p VBench")):
    prompt_idx = item.get("original_idx", i)
    prompt_en = item.get("prompt_en", "")
    refined_prompt = item.get("refined_prompt", prompt_en)
    seed = base_seed + prompt_idx * 1000  # 每个prompt仅1个sample
    data = {"seed": seed, "prompt": refined_prompt, "duration": gen_duration}

    try:
        out = perform_inference(pipe, data, args)
        fname = f"{prompt_idx:03d}-00.mp4"
        vpath = video_dir / fname
        save_video(out["output"], fps=args.fps, save_filepath=str(vpath))
        results.append({
            "prompt_idx": prompt_idx,
            "video_filename": fname,
            "video_path": str(vpath),
            "seed": seed,
            "prompt_en": prompt_en,
            "refined_prompt": refined_prompt,
            "elapsed_time": out.get("elapsed_time")
        })
    except Exception as e:
        results.append({
            "prompt_idx": prompt_idx,
            "video_path": None,
            "seed": seed,
            "prompt_en": prompt_en,
            "refined_prompt": refined_prompt,
            "error": str(e)
        })

meta = out_dir / "metadata.json"
with open(meta, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

ok = sum(1 for r in results if r.get("video_path"))
print(f"[Done] generated {ok}/{len(prompts)} videos")
print(f"[Done] videos dir: {video_dir}")
print(f"[Done] metadata: {meta}")
PY

echo "已启动。日志: ${OUT_DIR}.log"
echo "视频目录: ${OUT_DIR}/videos"