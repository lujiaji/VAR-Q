# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
import sys
import json
import os
import os.path as osp
from tqdm import tqdm
import sys
import time
from contextlib import nullcontext
import numpy as np
import torch
import cv2
import numpy as np
import argparse
from PIL import Image
sys.path.append(osp.dirname(osp.dirname(__file__)))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

_workspace_root = osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)

from tools.run_infinity import load_tokenizer, load_transformer, load_visual_tokenizer, gen_one_example, save_video, transform
from infinity.models.self_correction import SelfCorrection
from infinity.schedules.dynamic_resolution import get_dynamic_resolution_meta, get_first_full_spatial_size_scale_index
from infinity.schedules import get_encode_decode_func
from infinity.utils.arg_util import Args
from RuntimeProfiler.utils import (
    MIB as _MIB,
    cuda_memory_stats_mib,
    emit_profile_memory_marker,
    emit_runtime_mem_trace_markers,
    get_activation_peak_mb,
    mark_mem,
    mirror_infinitystar_artifacts_to_runtime_profiler,
    reset_peak_and_get_baseline_alloc_mb,
)


def _emit_memory_marker(tracer, marker_name: str, value_mb=None, marker_ts_ns=None, **extra):
    # Backward-compatible internal alias.
    component = str(extra.pop("component", "custom"))
    is_peak = bool(extra.pop("is_peak", False))
    emit_profile_memory_marker(
        tracer=tracer,
        marker_name=marker_name,
        component=component,
        value_mb=value_mb,
        marker_ts_ns=marker_ts_ns,
        is_peak=is_peak,
        **extra,
    )


def _cuda_memory_stats_mib():
    return cuda_memory_stats_mib()


def load_varq_quant_config(config_file: str):
    repo_root = osp.dirname(osp.dirname(osp.dirname(__file__)))
    if config_file is None:
        config_file = osp.join(repo_root, "VAR_Q", "Infinity-VAR_Q-8.json")
    if not osp.exists(config_file):
        raise FileNotFoundError(f"VARQ config not found: {config_file}")

    sys.path.append(repo_root)
    from VAR_Q.config_loader import VARQConfig

    config = VARQConfig(config_file)
    quant_config = config.get_quantization_config()
    print(f"[Config] Loading configuration from {config_file}")
    print(f"[Config]   - enable: {quant_config.get('enable', False)}")
    print(f"[Config]   - q_bits: {quant_config.get('q_bits', 8)}")
    print(f"[Config]   - quant_method: {quant_config.get('quant_method', 'G_SCALE_HEAD_DIM')}")
    print(f"[Config]   - qkv_format: {quant_config.get('qkv_format', 'BHLc')}")
    print(f"[Config]   - rescale_qk: {quant_config.get('rescale_qk', False)}")
    print(f"[Config]   - enable_sageattn: {quant_config.get('enable_sageattn', False)}")
    print(f"[Config]   - sageattn_type: {quant_config.get('sageattn_type', 'sageattn')}")
    return quant_config


def _init_prompt_rewriter():
    from tools.prompt_rewriter import OpenAIGPTModel
    """Initialize the OpenAI GPT model."""
    # Initialize the OpenAI GPT model
    model_name = 'gpt-4o-2024-08-06'
    ak = os.environ.get("OPEN_API_KEY", "")
    if len(ak) == 0:
        raise ValueError("Please provide your OpenAI API key in the OPEN_API_KEY environment variable.")
    model = OpenAIGPTModel(model_name, ak, if_global=True)
    system_prompt = (
        "You are a large language model specialized in rewriting video descriptions. Your task is to modify the input description to make the video more realistic and beautiful. 0. Preserve ALL information, including style words and technical terms. 1. If the subject is related to person, you need to provide a detailed description focusing on basic visual characteristics of the person, such as appearance, clothing, expression, posture, etc. You need to make the person as beautiful and handsome as possible. When the subject is only one person or object, do not use they to describe him/her/it to avoid confusion with multiple subjects. 2. If the input does not include style, lighting, atmosphere, you can make reasonable associations. 3. We only generate a four-second video based on your descriptions. So do not generate descriptions that are too long, too complex or contain too many activities. 4. You can add some descriptions of camera movements with regards to the scenes and allow the scenes to have very natural and coherent movements. 6. If the input is in Chinese, translate the entire description to English. 7. Output ALL must be in English. 8. Here are some expanded descriptions that can serve as examples: 1. The video begins with a distant aerial view of a winding river cutting through a rocky landscape, with the sun casting a soft glow over the scene. As the camera moves closer, the river's flow becomes more visible, and the surrounding terrain appears more defined. The camera continues to approach, revealing a steep cliff with a person sitting on its edge. The person is positioned near the top of the cliff, overlooking the river below. The camera finally reaches a close-up view, showing the person sitting calmly on the cliff, with the river and landscape fully visible in the background. 2. In a laboratory setting, a machine with a metallic structure and a green platform is seen. A small, clear plastic bottle is positioned on the green platform. The machine has a control panel with red and green lights on the right side. A nozzle is positioned above the bottle, and it begins to dispense liquid into the bottle. The liquid is dispensed in small droplets, and the nozzle moves slightly between each droplet. The background includes other laboratory equipment and a mesh-like structure. 3. The video shows a panoramic view of a cityscape with a prominent building featuring a green dome and ornate architecture in the center. Surrounding the main building are several other structures, including a white building with balconies on the left and a taller building with multiple windows on the right. In the background, there are hills with scattered buildings and greenery. The camera remains stationary, capturing the scene from a fixed position, with no noticeable changes in the environment or the buildings throughout the frames. 4. In a dimly lit room with red and blue lighting, a person holds up a smartphone to record a video of a band performing. The band members are seated, with one holding a guitar and another playing a double bass. The smartphone screen shows the band members being recorded, with the camera capturing their movements and expressions. The background includes a lamp and some furniture, adding to the cozy atmosphere of the scene. 5. In a grassy area with scattered trees, a large tree stands prominently in the center. A lion is perched on a thick branch of this tree, looking out into the distance. The sky is overcast, adding a somber tone to the scene. 6. A man in a green sweater holding a paper turns around and speaks to a group of people seated in a theater. He then points at a man in a yellow sweater sitting in the front row. The man in the yellow sweater looks at the paper in his hand and begins to speak. The man in the green sweater lowers his head and then looks up at the man in the yellow sweater again. 7. An elderly man, wearing a beige sweater over a yellow shirt, is sitting in front of a laptop. He holds a pair of glasses in his right hand and appears to be deep in thought, resting his head on his hand. He then raises the glasses and rubs his eyes with his fingers, showing signs of fatigue. After rubbing his eyes, he places the glasses on his sweater and looks down at the laptop screen. 8. A woman and a child are sitting at a table, each holding a pencil and coloring on a piece of paper. The woman is coloring a green leafy plant, while the child is coloring a red and blue object. The table has several colored pencils, a container filled with more pencils, and a few small colorful blocks. The woman is wearing a striped shirt, and the child is focused on their drawing. 9. A person wearing teal running shoes and colorful socks is running on a wet, sandy surface. The camera captures the movement of their legs and feet as they lift off the ground and land back, creating a clear shadow on the wet sand. The shadow elongates and shifts with each step, indicating the person's motion. The background remains consistent with the wet, textured sand, and the focus is solely on the runner's feet and their shadow. 10. A man is running along the shoreline of a beach, with the ocean waves gently crashing onto the shore. The sun is setting in the background, casting a warm glow over the scene. The man is wearing a light-colored jacket and shorts, and his hair is blowing in the wind as he runs. The water splashes around his legs as he moves forward, and his reflection is visible on the wet sand. The waves create a dynamic and lively atmosphere as they roll in and out."
    )
    gpt_model = OpenAIGPTModel(model_name, ak, if_global=True)
    return gpt_model, system_prompt

class InferencePipe:
    def __init__(self, args):
        # load text encoder
        self.text_tokenizer, self.text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
        # load vae
        self.vae = load_visual_tokenizer(args)
        self.vae = self.vae.float().to('cuda')
        # load infinity
        self.infinity = load_transformer(self.vae, args)
        self.self_correction = SelfCorrection(self.vae, args)
        
        self._models = [self.text_tokenizer, self.text_encoder, self.vae, self.infinity, self.self_correction]

        self.video_encode, self.video_decode, self.get_visual_rope_embeds, self.get_scale_pack_info = get_encode_decode_func(args.dynamic_scale_schedule)

        if args.enable_rewriter:
            self.gpt_model, self.system_prompt = _init_prompt_rewriter()   


def perform_inference(pipe, data, args, tracer=None):
    prompt = data["prompt"]
    seed = data["seed"]
    mapped_duration=5
    num_frames=81

    # If an image_path is provided, perform image-to-video generation.
    image_path = data.get("image_path", None)
    negative_prompt = data.get("negative_prompt", "")

    if isinstance(prompt, str):
        batch_size = int(data.get("batch_size", 1))
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        prompt_list = [prompt] * batch_size
    else:
        prompt_list = list(prompt)
        if not prompt_list:
            raise ValueError("prompt list is empty")
        batch_size = len(prompt_list)

    dynamic_resolution_h_w, h_div_w_templates = get_dynamic_resolution_meta(args.dynamic_scale_schedule, args.video_frames)
    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-0.571))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['pt2scale_schedule'][(num_frames-1)//4+1]
    args.first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule)
    args.tower_split_index = args.first_full_spatial_size_scale_index + 1
    context_info = pipe.get_scale_pack_info(scale_schedule, args.first_full_spatial_size_scale_index, args)    
    # for si, info in context_info.items():
    #     print(f"scale {si}: left_ref = {info['left_ref']}, right_ref = {info['right_ref']}")
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['pt2scale_schedule'][(num_frames-1)//4+1]
    tau = [args.tau_image] * args.tower_split_index + [args.tau_video] * (len(scale_schedule) - args.tower_split_index)
    tgt_h, tgt_w = scale_schedule[-1][1] * 16, scale_schedule[-1][2] * 16
    gt_leak, gt_ls_Bl = -1, None

    if image_path is not None:
        ref_image = [cv2.imread(image_path)[:,:,::-1]]
        ref_img_T3HW = [transform(Image.fromarray(frame).convert("RGB"), tgt_h, tgt_w) for frame in ref_image]
        ref_img_T3HW = torch.stack(ref_img_T3HW, 0) # [t,3,h,w]
        ref_img_bcthw = ref_img_T3HW.permute(1,0,2,3).unsqueeze(0) # [c,t,h,w] -> [b,c,t,h,w]
        _, _, gt_ls_Bl, _, _, _ = pipe.video_encode(pipe.vae, ref_img_bcthw.cuda(), vae_features=None, self_correction=pipe.self_correction, args=args, infer_mode=True, dynamic_resolution_h_w=dynamic_resolution_h_w)
        gt_leak=len(scale_schedule)//2

    generated_image_list = []
    prompt_list = [f'{item}, Close-up on big objects, emphasize scale and detail' for item in prompt_list]
    if args.append_duration2caption:
        prompt_list = [f'<<<t={mapped_duration}s>>>{item}' for item in prompt_list]
    
    start_time = time.time()
    baseline_alloc_mb = reset_peak_and_get_baseline_alloc_mb()
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True), torch.no_grad():
        generated_image, _ = gen_one_example(
            pipe.infinity,
            pipe.vae,
            pipe.text_tokenizer,
            pipe.text_encoder,
            prompt_list,
            negative_prompt=negative_prompt,
            g_seed=seed,
            gt_leak=gt_leak,
            gt_ls_Bl=gt_ls_Bl,
            cfg_list=args.cfg, 
            tau_list=tau,
            scale_schedule=scale_schedule,
            cfg_insertion_layer=[0],
            vae_type=args.vae_type,
            sampling_per_bits=1,
            enable_positive_prompt=0,
            low_vram_mode=True,
            args=args,
            get_visual_rope_embeds=pipe.get_visual_rope_embeds,
            context_info=context_info,
            noise_list=None,
        )
        if len(generated_image.shape) == 3:
            generated_image = generated_image.unsqueeze(0)
        print(generated_image.shape)
        generated_image_list.append(generated_image)
            
    generated_image = torch.cat(generated_image_list, 2)
    end_time = time.time()
    elapsed_time = end_time - start_time    

    activation_peak_mb = get_activation_peak_mb(baseline_alloc_mb)

    mem_trace = list(getattr(pipe.infinity, "runtime_mem_trace", []) or [])
    kv_cache_live_mb = float(getattr(pipe.infinity, "last_kv_cache_mb", 0.0) or 0.0)
    emit_runtime_mem_trace_markers(
        tracer, mem_trace, baseline_alloc_mb, activation_peak_mb, kv_cache_live_mb
    )
    
    return {
            "output": generated_image.cpu().numpy(),
            "elapsed_time": elapsed_time,
            "weights_baseline_mb": baseline_alloc_mb or 0.0,
        }


def plot_infinitystar_profiles(latency_trace, weights_baseline_mb, save_dir):
    """Generate InfinityStar profiling charts.

    Chart 1: Pie chart — end-to-end latency breakdown (Attention / FFN / Others).
    Chart 2: Two subplots — per-scale memory (stacked bar) and per-scale latency (line).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from collections import defaultdict

    if not latency_trace:
        return None, None

    all_steps = latency_trace
    visual_steps = [s for s in all_steps if isinstance(s['scale_ind'], int)]

    total_attn = sum(s['attn_time_s'] for s in all_steps)
    total_ffn = sum(s['ffn_time_s'] for s in all_steps)
    total_others = sum(s['others_time_s'] for s in all_steps)
    grand_total = total_attn + total_ffn + total_others

    # ---- Chart 1: Latency breakdown pie ----
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sizes = [total_attn, total_ffn, total_others]
    labels = [
        f'Attention\n{total_attn:.2f}s ({total_attn/grand_total*100:.1f}%)',
        f'FFN\n{total_ffn:.2f}s ({total_ffn/grand_total*100:.1f}%)',
        f'Others\n{total_others:.2f}s ({total_others/grand_total*100:.1f}%)',
    ]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    wedges, texts = ax1.pie(sizes, labels=labels, colors=colors, startangle=90,
                            textprops={'fontsize': 11}, labeldistance=1.15)
    ax1.set_title(
        f'InfinityStar End-to-End Latency Breakdown\nTotal: {grand_total:.2f}s',
        fontsize=14, fontweight='bold',
    )
    fig1.tight_layout()
    chart1_path = osp.join(save_dir, 'infinitystar_latency_breakdown.png')
    fig1.savefig(chart1_path, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"[InfinityStar Profile] Latency breakdown chart saved: {chart1_path}")

    if not visual_steps:
        return chart1_path, None

    # ---- Chart 2: Per-scale memory & latency (two subplots) ----
    scale_agg = defaultdict(lambda: {
        'attn_time': 0.0, 'ffn_time': 0.0, 'others_time': 0.0, 'total_time': 0.0,
        'allocated_mb': 0.0, 'reserved_mb': 0.0, 'kv_cache_mb': 0.0,
    })
    for s in visual_steps:
        si = s['scale_ind']
        scale_agg[si]['attn_time'] += s['attn_time_s']
        scale_agg[si]['ffn_time'] += s['ffn_time_s']
        scale_agg[si]['others_time'] += s['others_time_s']
        scale_agg[si]['total_time'] += s['total_time_s']
        scale_agg[si]['allocated_mb'] = s['allocated_mb']
        scale_agg[si]['reserved_mb'] = s['reserved_mb']
        scale_agg[si]['kv_cache_mb'] = s['kv_cache_mb']

    scale_inds = sorted(scale_agg.keys())
    n = len(scale_inds)

    attn_mem, ffn_mem, others_mem, reserved_mem = [], [], [], []
    latency_vals = []
    for si in scale_inds:
        d = scale_agg[si]
        kv = d['kv_cache_mb']
        alloc = d['allocated_mb']
        resv = d['reserved_mb']
        act = max(0.0, alloc - kv - weights_baseline_mb)
        attn_mem.append(kv)
        ffn_mem.append(act)
        others_mem.append(weights_baseline_mb)
        reserved_mem.append(resv)
        latency_vals.append(d['total_time'])

    x = list(range(n))
    fig2, (ax_mem, ax_lat) = plt.subplots(2, 1, figsize=(max(14, n * 0.45), 10), sharex=True)

    # Memory subplot — stacked bar
    ax_mem.bar(x, others_mem, width=0.7, label='Others (Weights)', color='#45B7D1')
    bottom1 = [o for o in others_mem]
    ax_mem.bar(x, attn_mem, width=0.7, bottom=bottom1, label='Attention (KV Cache)', color='#FF6B6B')
    bottom2 = [o + a for o, a in zip(others_mem, attn_mem)]
    ax_mem.bar(x, ffn_mem, width=0.7, bottom=bottom2, label='FFN (Activations)', color='#4ECDC4')
    ax_mem.plot(x, reserved_mem, 'k--', linewidth=1.5, label='CUDA Reserved')
    ax_mem.set_ylabel('Memory (MiB)', fontsize=12)
    ax_mem.set_title('Per-Scale Memory Breakdown', fontsize=13, fontweight='bold')
    ax_mem.legend(loc='upper left', fontsize=9)
    ax_mem.grid(axis='y', alpha=0.3)

    # Latency subplot — line chart
    ax_lat.plot(x, latency_vals, 'o-', color='#FF6B6B', linewidth=1.5, markersize=4, label='Total Latency')
    ax_lat.set_ylabel('Latency (s)', fontsize=12)
    ax_lat.set_xlabel('Scale Index', fontsize=12)
    ax_lat.set_title('Per-Scale Latency', fontsize=13, fontweight='bold')
    ax_lat.set_xticks(x)
    ax_lat.set_xticklabels([str(si) for si in scale_inds], fontsize=8, rotation=45 if n > 20 else 0)
    ax_lat.legend(loc='upper left', fontsize=9)
    ax_lat.grid(axis='y', alpha=0.3)

    fig2.tight_layout()
    chart2_path = osp.join(save_dir, 'infinitystar_per_scale_profile.png')
    fig2.savefig(chart2_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"[InfinityStar Profile] Per-scale profile chart saved: {chart2_path}")

    return chart1_path, chart2_path


def _profiler_span(profiler, name: str):
    if profiler is not None:
        return profiler.trace(name)
    return nullcontext()


def execute_infinitystar_profile(
    checkpoints_dir="/data/jiaji_lu/WM/infinitystar/checkpoints/InfinityStar",
    prompt="Iron Man fights a giant slime monster in a ruined city at night, flying fast, firing repulsor blasts, cinematic explosions, dynamic camera, realistic lighting, wet streets, smoke, sparks, epic superhero movie style, photorealistic, 4K.",
    seed=41,
    save_video_path="/home/jiaji_lu/AR/VAR-Q/Benchmark/outputs/gen_videos/demo.mp4",
    config_file=None,
    enable_rewriter=0,
    image_path=None,
    tracer=None,
):
    """
    端到端入口，供 RuntimeProfiler 的 profile_any_model.py 调用（通过 --target-kwargs-json 传参，tracer 由 --inject-profiler-kwarg 注入）。
    """
    if config_file is None:
        config_file = os.environ.get("CONFIG_FILE", None)

    args = Args()
    args.pn = "0.90M"
    args.fps = 16
    args.video_frames = 81
    args.model_path = os.path.join(checkpoints_dir, "infinitystar_8b_720p_weights")
    args.checkpoint_type = "torch_shard"
    args.vae_path = os.path.join(checkpoints_dir, "infinitystar_videovae.pth")
    args.text_encoder_ckpt = os.path.join(checkpoints_dir, "text_encoder/flan-t5-xl-official/")
    args.model_type = "infinity_qwen8b"
    args.text_channels = 2048
    args.dynamic_scale_schedule = "infinity_elegant_clip20frames_v2"
    args.bf16 = 1
    args.use_apg = 1
    args.use_cfg = 0
    args.cfg = 34
    args.tau_image = 1
    args.tau_video = 0.4
    args.apg_norm_threshold = 0.05
    args.image_scale_repetition = "[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]"
    args.video_scale_repetition = "[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 1]"
    args.append_duration2caption = 1
    args.use_two_stage_lfq = 1
    args.detail_scale_min_tokens = 750
    args.semantic_scales = 12
    args.max_repeat_times = 10000
    args.enable_rewriter = enable_rewriter

    quant_config = load_varq_quant_config(config_file)
    args.enable_quantization = int(quant_config.get("enable", False))
    args.q_bits = quant_config.get("q_bits", 8)
    args.quant_method = quant_config.get("quant_method", "G_SCALE_HEAD_DIM")
    args.qkv_format = quant_config.get("qkv_format", "BHLc")
    args.rescale_qk = int(quant_config.get("rescale_qk", False))
    args.enable_sageattn = int(quant_config.get("enable_sageattn", False))
    args.sageattn_type = str(quant_config.get("sageattn_type", "sageattn"))

    with _profiler_span(tracer, "load_models"):
        pipe = InferencePipe(args)
    ws = _cuda_memory_stats_mib()
    _emit_memory_marker(
        tracer,
        "weights_loaded",
        component="weights",
        value_mb=ws.get("alloc_mem_mb"),
        alloc_mem_mb=ws.get("alloc_mem_mb"),
        reserved_mem_mb=ws.get("reserved_mem_mb"),
    )

    data = {"seed": seed, "prompt": prompt}
    if image_path is not None:
        data["image_path"] = image_path

    if args.enable_rewriter:
        with _profiler_span(tracer, "prompt_rewrite"):
            rewritten_prompt = pipe.gpt_model(
                prompt=(
                    "Rewrite the following video descriptions, add more details of the subject and the camera movement to enhance the quality of the video. Do not use the word 'they' to refer to a single person or object. Concatenate all sentences together, not present them in paragraphs. Please rewrite with concise and clear language: "
                    + prompt
                ),
                system_prompt=pipe.system_prompt,
            )
            print(f"Rewritten prompt: {rewritten_prompt}")
            prompt = rewritten_prompt
        data["prompt"] = prompt

    with _profiler_span(tracer, "perform_inference"):
        output_dict = perform_inference(pipe, data, args, tracer=tracer)

    os.makedirs(osp.dirname(save_video_path) or ".", exist_ok=True)
    with _profiler_span(tracer, "save_video"):
        save_video(output_dict["output"], fps=args.fps, save_filepath=save_video_path)

    latency_trace = list(getattr(pipe.infinity, 'runtime_latency_trace', []) or [])
    charts_dir = osp.dirname(save_video_path) or "."
    if latency_trace:
        weights_baseline = getattr(pipe.infinity, '_weights_baseline_mb', 0.0)
        try:
            chart1, chart2 = plot_infinitystar_profiles(latency_trace, weights_baseline, charts_dir)
            output_dict['latency_chart'] = chart1
            output_dict['per_scale_chart'] = chart2
        except Exception as e:
            print(f"[InfinityStar Profile] Chart generation failed: {e}")
    mirror_infinitystar_artifacts_to_runtime_profiler(
        tracer,
        charts_dir,
        latency_trace=latency_trace or None,
        perform_inference_wall_s=output_dict.get("elapsed_time"),
    )

    print(f"Video genernation done: {save_video_path=}")
    return output_dict


if __name__ == '__main__':
    # For optimal performance, enabling the prompt rewriter is recommended.
    # To utilize the GPT model, ensure the following environment variables are set:
    # export OPEN_API_KEY="YOUR_API_KEY"
    # export GLOBAL_AZURE_ENDPOINT="YOUR_ENDPOINT"
    checkpoints_dir = "/data/jiaji_lu/WM/infinitystar/checkpoints/InfinityStar"
    prompt = "Iron Man fights a giant slime monster in a ruined city at night, flying fast, firing repulsor blasts, cinematic explosions, dynamic camera, realistic lighting, wet streets, smoke, sparks, epic superhero movie style, photorealistic, 4K."
    save_dir = "/home/jiaji_lu/AR/VAR-Q/Benchmark/outputs"
    gen_video_path = osp.join(osp.join(save_dir, "gen_videos"), "demo.mp4")
    execute_infinitystar_profile(
        checkpoints_dir=checkpoints_dir,
        prompt=prompt,
        seed=41,
        save_video_path=gen_video_path,
        enable_rewriter=0,
    )
