import os
import os.path as osp
import time
import argparse
import json
import cv2
import numpy as np
import csv
from pytorch_lightning import seed_everything

from infinity.utils.csv_util import load_csv_as_dicts, write_dicts2csv_file
from tools.run_infinity import *
from conf import HF_TOKEN, HF_HOME

# set environment variables
os.environ['HF_TOKEN'] = HF_TOKEN
os.environ['HF_HOME'] = HF_HOME
os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = '1'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument('--outdir', type=str, default='')
    parser.add_argument('--n_samples', type=int, default=4)
    parser.add_argument('--metadata_file', type=str, default='DPG/DPG_prompts.jsonl')
    parser.add_argument("--q_bits", type=int, default=8)
    parser.add_argument("--q_dim", type=str, default="per-head+per-dim")
    parser.add_argument("--use_diff_bits", type=bool, default=False)
    args = parser.parse_args()

    os.makedirs(args.outdir,exist_ok=True)
    grid_dir=osp.join(args.outdir,'dpg_images')
    os.makedirs(grid_dir,exist_ok=True)

    # parse cfg
    args.cfg = list(map(float, args.cfg.split(',')))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]
    
    with open(args.metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]
    total=len(metadatas);print(f'total {total} prompts')

    if 'infinity' in args.model_type:
        # load text encoder
        text_tokenizer, text_encoder = load_tokenizer(t5_path =args.text_encoder_ckpt)
        # load vae
        vae = load_visual_tokenizer(args)
        # load infinity
        infinity = load_transformer(vae, args)

    rows=[]

    for index, metadata in enumerate(metadatas):
        seed_everything(args.seed)
        prompt = metadata['prompt']
        print(f"Prompt ({index: >3}/{len(metadatas)}): '{prompt}'")

        tau = args.tau
        cfg = args.cfg
        images = []
        for sample_j in range(args.n_samples):
            print(f"Generating {sample_j+1} of {args.n_samples}, prompt={prompt}")
            t1 = time.time()
            h_div_w_template = 1.000
            scale_schedule = dynamic_resolution_h_w[h_div_w_template][args.pn]['scales']
            scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
            image = gen_one_img(infinity, vae, text_tokenizer, text_encoder, prompt, tau_list=tau, cfg_sc=3, cfg_list=cfg, scale_schedule=scale_schedule, cfg_insertion_layer=[args.cfg_insertion_layer], vae_type=args.vae_type)
            t2 = time.time()
            print(f'{args.model_type} infer one image takes {t2-t1:.2f}s')
            images.append(image)

        #now prepare the DPG images
        arrs=[im.cpu().numpy() if hasattr(im, 'cpu') else np.array(im) for im in images]
        h, w = arrs[0].shape[:2]
        grid = np.zeros((2*h, 2*w, 3), dtype=arrs[0].dtype)
        grid[0:h,   0:w  ] = arrs[0]
        grid[0:h,   w:2*w] = arrs[1]
        grid[h:2*h, 0:w  ] = arrs[2]
        grid[h:2*h, w:2*w] = arrs[3]
        #save images
        item_id = f"{index:05}"
        out_img = osp.join(grid_dir, f"{item_id}.png")
        cv2.imwrite(out_img, grid)

        rows.append({
            'item_id': item_id,
            'text': prompt,
            'keywords': '',
            'proposition_id': '0',
            'dependency': '0',
            'category_broad': '',
            'category_detailed': '',
            'tuple': '',
            'question_natural_language':''
        })
    csv_path ='Benchmark/DPG/dpg_metadata.csv'
    fieldnames = [
        'item_id','text','keywords','proposition_id',
        'dependency','category_broad','category_detailed',
        'tuple','question_natural_language'
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"save csv as: {csv_path}")

