#!/bin/bash

infer_eval_image_reward() {
    ${pip_ext} install image-reward pytorch_lightning
    ${pip_ext} install -U timm diffusers
    ${pip_ext} install openai==1.34.0 
    ${pip_ext} install httpx==0.20.0 

    # step 1, infer images
    ${python_ext} ImageReward/infer4eval.py \
    --cfg ${cfg} \
    --tau ${tau} \
    --pn ${pn} \
    --model_path ${infinity_model_path} \
    --vae_type ${vae_type} \
    --vae_path ${vae_path} \
    --add_lvl_embeding_only_first_block ${add_lvl_embeding_only_first_block} \
    --use_bit_label ${use_bit_label} \
    --model_type ${model_type} \
    --rope2d_each_sa_layer ${rope2d_each_sa_layer} \
    --rope2d_normalized_by_hw ${rope2d_normalized_by_hw} \
    --use_scale_schedule_embedding ${use_scale_schedule_embedding} \
    --cfg ${cfg} \
    --tau ${tau} \
    --checkpoint_type ${checkpoint_type} \
    --text_encoder_ckpt ${text_encoder_ckpt} \
    --text_channels ${text_channels} \
    --apply_spatial_patchify ${apply_spatial_patchify} \
    --cfg_insertion_layer ${cfg_insertion_layer} \
    --outdir  ${out_dir} \

    # step 2, compute image reward
    ${pip_ext} install "transformers==4.28.1" "diffusers==0.16.0" "huggingface_hub==0.16.4" -U
    ${pip_ext} install "datasets==1.18.4" "timm==0.6.13" "accelerate==0.18.0" -U
    ${pip_ext} install git+https://github.com/openai/CLIP.git ftfy
    ${python_ext} ImageReward/cal_imagereward.py \
    --meta_file ${out_dir}/metadata.jsonl
}

test_gen_eval() {
    ${pip_ext} install -U openmim
    mim install mmengine mmcv-full==1.7.2
    ${pip_ext} install mmdet==2.28.2 pytorch_lightning clip_benchmark open-clip-torch==2.20.0
    ${pip_ext} install -U diffusers
    sudo apt install libgl1
    ${pip_ext} install openai
    ${pip_ext} install httpx==0.20.0

    # run inference
    ${python_ext} GenEval/infer4eval.py \
    --cfg ${cfg} \
    --tau ${tau} \
    --pn ${pn} \
    --model_path ${infinity_model_path} \
    --vae_type ${vae_type} \
    --vae_path ${vae_path} \
    --add_lvl_embeding_only_first_block ${add_lvl_embeding_only_first_block} \
    --use_bit_label ${use_bit_label} \
    --model_type ${model_type} \
    --rope2d_each_sa_layer ${rope2d_each_sa_layer} \
    --rope2d_normalized_by_hw ${rope2d_normalized_by_hw} \
    --use_scale_schedule_embedding ${use_scale_schedule_embedding} \
    --cfg ${cfg} \
    --tau ${tau} \
    --checkpoint_type ${checkpoint_type} \
    --text_encoder_ckpt ${text_encoder_ckpt} \
    --text_channels ${text_channels} \
    --apply_spatial_patchify ${apply_spatial_patchify} \
    --cfg_insertion_layer ${cfg_insertion_layer} \
    --outdir ${out_dir}/images \
    --rewrite_prompt ${rewrite_prompt} \

    # detect objects
    ${python_ext} GenEval/evaluate_images.py ${out_dir}/images \
    --outfile ${out_dir}/results/det.jsonl \
    --model-config GenEval/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py \
    --model-path weights/mask2former

    # accumulate results
    ${python_ext} GenEval/summary_scores.py ${out_dir}/results/det.jsonl > ${out_dir}/results/res.txt
    cat ${out_dir}/results/res.txt
}

test_DPG() {
    # generate combined imgs
    ${python_ext} DPG/infer4eval.py \
    --cfg ${cfg} \
    --tau ${tau} \
    --pn ${pn} \
    --model_path ${infinity_model_path} \
    --vae_type ${vae_type} \
    --vae_path ${vae_path} \
    --add_lvl_embeding_only_first_block ${add_lvl_embeding_only_first_block} \
    --use_bit_label ${use_bit_label} \
    --model_type ${model_type} \
    --rope2d_each_sa_layer ${rope2d_each_sa_layer} \
    --rope2d_normalized_by_hw ${rope2d_normalized_by_hw} \
    --use_scale_schedule_embedding ${use_scale_schedule_embedding} \
    --cfg ${cfg} \
    --tau ${tau} \
    --checkpoint_type ${checkpoint_type} \
    --text_encoder_ckpt ${text_encoder_ckpt} \
    --text_channels ${text_channels} \
    --apply_spatial_patchify ${apply_spatial_patchify} \
    --cfg_insertion_layer ${cfg_insertion_layer} \
    --outdir ${out_dir}/images \

    #run DPG
    ${pip_ext} install "huggingface_hub==0.24.7" -U
    ${pip_ext} install "datasets==2.18.0" "timm==0.6.13" -U
    bash DPG/dist_eval.sh ${img_fold} 1024
}

########## Workpath should be set to VAR-Q/scripts ##########
cd ../Benchmark 

########## Select a model size ##########
MODEL_SIZE="2b"

python_ext=python3
pip_ext=pip3

# set arguments for inference
pn=1M
use_scale_schedule_embedding=0
use_bit_label=1
cfg=3
tau=1
rope2d_normalized_by_hw=2
add_lvl_embeding_only_first_block=1
rope2d_each_sa_layer=1
text_encoder_ckpt=YOUR_PATH/flan-t5-xl
text_channels=2048
cfg_insertion_layer=0
sub_fix=cfg${cfg}_tau${tau}_cfg_insertion_layer${cfg_insertion_layer}

if [[ "$MODEL_SIZE" == "2b" ]]; then
    echo "[running 2b Infinity]"
    model_type=infinity_2b
    checkpoint_type='torch'
    infinity_model_path=YOUR_PATH/infinity_2b_reg.pth
    out_dir_root=output/Infinity/eval/2b_eval
    vae_type=32
    vae_path=YOUR_PATH/infinity_vae_d32reg.pth
    apply_spatial_patchify=0
else
    echo "[running 8b Infinity]"
    model_type=infinity_8b
    checkpoint_type='torch_shard'
    infinity_model_path=YOUR_PATH/infinity_8b_weights
    out_dir_root=output/Infinity/eval/8b_eval
    vae_type=14
    vae_path=YOUR_PATH/infinity_vae_d56_f8_14_patchify.pth
    apply_spatial_patchify=1
fi

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0  #please set GPU

# ImageReward
out_dir=${out_dir_root}/image_reward_${sub_fix}
infer_eval_image_reward > results_txt/${MODEL_SIZE}/IR.txt 2>&1

# GenEval
rewrite_prompt=0
out_dir=${out_dir_root}/gen_eval_${sub_fix}_rewrite_prompt${rewrite_prompt}_round2_real_rewrite
test_gen_eval > results_txt/${MODEL_SIZE}/gen_eval.txt 2>&1

#DPG
out_dir=${out_dir_root}/DPG_${sub_fix}
img_fold=${out_dir_root}/DPG_cfg3_tau1_cfg_insertion_layer0/images/dpg_images/
test_DPG > results_txt/${MODEL_SIZE}/DPG.txt 2>&1